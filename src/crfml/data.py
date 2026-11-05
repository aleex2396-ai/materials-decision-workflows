"""Local-only workbook intake with a frozen chronological reservation.

This is an accidental-leakage guard, not an access-control/security boundary.
Source paths, hashes, values, and intake logs belong in the ignored private tree.
Date blocks are a conservative observable grouping, not proof of independence.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dates(path: str) -> set[str]:
    values = set()
    for pattern, order in ((r"(?<!\d)(20\d{2})[_-](\d{2})[_-](\d{2})(?!\d)", (0, 1, 2)),
                           (r"(?<!\d)(\d{2})[_-](\d{2})[_-](20\d{2})(?!\d)", (2, 1, 0))):
        for match in re.finditer(pattern, path):
            parts = match.groups()
            try:
                values.add(date(*(int(parts[i]) for i in order)).isoformat())
            except ValueError:
                pass
    return values


def build_manifest(source_root: str | Path, private_dir: str | Path,
                   sealed_fraction: float = 0.70,
                   prior_knowledge: str = "Historical narrative summaries may have been read; no claim of human blinding.") -> dict[str, Any]:
    """Freeze file metadata before opening cells; refuse to replace a manifest.

    Files sharing any date in their relative paths form one connected date block.
    Byte-identical workbooks also share a block, so copies cannot cross partitions.
    Latest blocks are sealed. Undated workbooks and non-workbooks are quarantined.
    No cell, worksheet, drawing, or outcome is read during this operation.
    """
    if not 0.60 <= sealed_fraction < 1:
        raise ValueError("Reserve at least 60%, and less than 100%, of dated blocks")
    root, destination = Path(source_root).resolve(), Path(private_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "manifest.json"
    if output.exists() or (destination / "manifest.sha256").exists():
        raise FileExistsError("Reservation already exists; do not regenerate after observing outcomes")
    records = []
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    seen_hash: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("Source contains an external symlink")
        relative = path.relative_to(root).as_posix()
        dates = sorted(_dates(relative))
        workbook = path.suffix.lower() == ".xlsx" and not path.name.startswith("~$")
        sha = _digest(path) if workbook else None
        record = dict(path=relative, bytes=path.stat().st_size, dates=dates,
                      sha256=sha, workbook=workbook)
        if workbook and dates:
            for item in dates:
                union(dates[0], item)
            if sha in seen_hash:
                union(dates[0], seen_hash[sha])
            else:
                seen_hash[sha] = dates[0]
        records.append(record)
    blocks: dict[str, set[str]] = {}
    for value in parent:
        blocks.setdefault(find(value), set()).add(value)
    ordered = sorted(blocks.values(), key=lambda values: (max(values), min(values)))
    if len(ordered) < 3:
        raise ValueError("Need at least three dated blocks for a reservation")
    n_sealed = math.ceil(sealed_fraction * len(ordered))
    development_dates = set().union(*ordered[:len(ordered) - n_sealed]) if n_sealed < len(ordered) else set()
    block_for_date = {day: f"date_block_{i:04d}" for i, group in enumerate(ordered) for day in group}
    for row in records:
        if not row["workbook"] or not row["dates"]:
            row.update(partition="quarantine", group=None)
        else:
            row.update(partition="development" if row["dates"][0] in development_dates else "sealed_test",
                       group=block_for_date[row["dates"][0]])
    # An undated duplicate of a dated file is kept quarantined, never opened.
    manifest = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                    source_root=str(root), policy="chronological_date_connected_components_v1",
                    requested_sealed_fraction=sealed_fraction,
                    dated_blocks=len(ordered), sealed_blocks=n_sealed,
                    grouping_limit="Metadata date blocks are not independent experiments; repeated formulations may cross dates. Sealed data are not a certified independent test until lineage review.",
                    prior_knowledge=prior_knowledge,
                    metadata_read_scope="Relative paths, sizes, SHA256 bytes only; no cell/XML content interpreted.",
                    files=records)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (destination / "manifest.sha256").write_text(_digest(output) + "\n", encoding="ascii")
    (destination / "access.jsonl").write_text(json.dumps({"event": "reservation_frozen", "manifest_sha256": _digest(output)}) + "\n", encoding="utf-8")
    return manifest


class PrivateStore:
    """Readers expose development data only. There is deliberately no test unlock."""

    def __init__(self, private_dir: str | Path):
        self.directory = Path(private_dir).resolve()
        manifest_path = self.directory / "manifest.json"
        self.manifest_hash = (self.directory / "manifest.sha256").read_text(encoding="ascii").strip()
        if _digest(manifest_path) != self.manifest_hash:
            raise ValueError("Manifest integrity check failed")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.root = Path(self.manifest["source_root"]).resolve()
        self.records = {item["path"]: dict(item) for item in self.manifest["files"]}
        self.amendment_hash = None
        amendment_path = self.directory / "development_amendment.json"
        if amendment_path.exists():
            self.amendment_hash = (self.directory / "development_amendment.sha256").read_text(encoding="ascii").strip()
            if _digest(amendment_path) != self.amendment_hash:
                raise ValueError("Amendment integrity check failed")
            amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
            if amendment["original_manifest_sha256"] != self.manifest_hash:
                raise ValueError("Amendment refers to another original manifest")
            expected = _earliest_returned_groups(self.manifest, amendment["count"])
            if amendment["development_groups"] != expected:
                raise ValueError("Amendment violates frozen metadata selection rule")
            if (self.manifest["sealed_blocks"] - len(expected)) / self.manifest["dated_blocks"] < 0.60:
                raise ValueError("Amendment leaves less than 60% of date blocks sealed")
            for row in self.records.values():
                if row["group"] in expected:
                    row["partition"] = "development"

    def _log(self, event: str, relative: str, **details: Any) -> None:
        row = dict(event=event, path=relative, utc=datetime.now(timezone.utc).isoformat(), **details)
        with (self.directory / "access.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")

    def development_paths(self) -> list[str]:
        return [key for key, row in self.records.items() if row["partition"] == "development"]

    def _check(self, relative: str) -> Path:
        if _digest(self.directory / "manifest.json") != self.manifest_hash:
            raise ValueError("Manifest changed after reader was created")
        amendment_path = self.directory / "development_amendment.json"
        if amendment_path.exists() != (self.amendment_hash is not None):
            raise ValueError("Reservation amendment changed; create a fresh reader")
        if self.amendment_hash is not None and _digest(amendment_path) != self.amendment_hash:
            raise ValueError("Amendment changed after reader was created")
        row = self.records.get(relative)
        if row is None or row["partition"] != "development":
            self._log("denied", relative, reason="not_development")
            raise PermissionError("Only frozen development workbooks may be opened")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            self._log("denied", relative, reason="outside_source")
            raise PermissionError("Path escapes source root")
        if _digest(path) != row["sha256"]:
            self._log("denied", relative, reason="source_changed")
            raise ValueError("Source differs from frozen reservation")
        return path

    def read_development_workbook(self, relative: str) -> dict[str, Any]:
        """Return cells with coordinates plus drawing presence, without guessing labels.

        Embedded pictures are inventoried but not interpreted. A photographic
        outcome remains unassessed until a qualified local review sees it.
        Formula cells are recorded as formulas, not silently treated as outcomes.
        """
        import openpyxl
        import zipfile
        path = self._check(relative)
        self._log("open_development_workbook", relative)
        book = openpyxl.load_workbook(path, read_only=True, data_only=False)
        sheets = []
        for sheet in book:
            cells = [{"coordinate": cell.coordinate, "value": cell.value,
                      "data_type": cell.data_type}
                     for row in sheet.iter_rows() for cell in row if cell.value is not None]
            sheets.append(dict(name=sheet.title, cells=cells))
        book.close()
        with zipfile.ZipFile(path) as archive:
            media = [name for name in archive.namelist() if name.startswith("xl/media/")]
            drawings = [name for name in archive.namelist() if re.match(r"xl/drawings/drawing\d+\.xml$", name)]
        return dict(source=relative, sha256=self.records[relative]["sha256"],
                    group=self.records[relative]["group"], sheets=sheets,
                    embedded_media_count=len(media), drawing_parts=drawings,
                    photo_interpretation="not_assessed", execution_fidelity="not_verified")


def _earliest_returned_groups(manifest: dict[str, Any], count: int) -> list[str]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in manifest["files"]:
        if row["partition"] == "sealed_test":
            groups.setdefault(row["group"], []).append(row)
    ordered = sorted(groups, key=lambda group: max(day for row in groups[group] for day in row["dates"]))
    eligible = [group for group in ordered if any("returns" in Path(row["path"]).parts for row in groups[group])]
    if count <= 0 or len(eligible) < count:
        raise ValueError("Insufficient returned components for the declared rule")
    return eligible[:count]


def freeze_returned_development_amendment(private_dir: str | Path, *, count: int,
                                         independent_review: str) -> dict[str, Any]:
    """One documented metadata-only development amendment; never overwrite.

    The original reservation remains immutable. This selects returned-file
    provenance, never outcomes, and does not restore confirmatory independence.
    An independent review must be recorded before calling this function.
    """
    directory = Path(private_dir).resolve()
    output = directory / "development_amendment.json"
    if output.exists() or (directory / "development_amendment.sha256").exists():
        raise FileExistsError("Only one amendment is permitted; no further reallocations")
    if not independent_review.strip():
        raise ValueError("Record the independent review before amendment")
    store = PrivateStore(directory)
    groups = _earliest_returned_groups(store.manifest, count)
    remaining = store.manifest["sealed_blocks"] - len(groups)
    if remaining / store.manifest["dated_blocks"] < 0.60:
        raise ValueError("Must preserve at least 60% of date components")
    amendment = dict(schema_version=1, original_manifest_sha256=store.manifest_hash,
                     utc=datetime.now(timezone.utc).isoformat(), count=count,
                     rule="Earliest originally sealed date components containing a returns directory; all associated files move together.",
                     reason="Metadata-based development allocation; inspect the recorded independent review for the caller's rationale.",
                     development_groups=groups, remaining_sealed_blocks=remaining,
                     dated_blocks=store.manifest["dated_blocks"],
                     sealed_component_fraction=remaining / store.manifest["dated_blocks"],
                     independent_review=independent_review,
                     limitation="Early-return provenance selection is exploratory and may induce selection bias. Component fraction is not independent-sample fraction. Historical summaries may be known.",
                     no_further_reallocation=True)
    output.write_text(json.dumps(amendment, indent=2), encoding="utf-8")
    digest = _digest(output)
    (directory / "development_amendment.sha256").write_text(digest + "\n", encoding="ascii")
    store._log("development_amendment_frozen", "", amendment_sha256=digest, groups=groups)
    return amendment


def validate_observation(row: dict[str, Any]) -> None:
    """Minimal public interchange contract; no inferred units or missing targets."""
    required = {"experiment_id", "group_id", "features", "target", "target_unit", "provenance", "flags"}
    if required - row.keys():
        raise ValueError(f"Missing fields: {sorted(required - row.keys())}")
    if not isinstance(row["features"], dict) or not row["features"]:
        raise ValueError("Features must be a nonempty name/value mapping")
    if not row["target_unit"] or not row["provenance"]:
        raise ValueError("Explicit target unit and provenance are required")
    numbers = list(row["features"].values()) + [row["target"]]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in numbers):
        raise ValueError("Features and target must be finite numbers; missing is not zero")
    if not isinstance(row["flags"], list):
        raise ValueError("Flags must be a list")


def extract_numeric_table(workbook: dict[str, Any], *, sheet_name: str,
                          header_row: int, id_header: str,
                          feature_headers: dict[str, str],
                          feature_units: dict[str, str], target_header: str,
                          target_unit: str, time_features: tuple[str, ...] = ()) -> dict[str, Any]:
    """Extract a declared table; preserve exclusions instead of inventing labels.

    Feature units are explicitly declared by the reviewed schema. Time conversion
    is opt-in for elapsed HH:MM:SS text, not a guess about clock time. Formula
    targets require a separately verified calculation and are rejected here.
    This produces *candidate* observations: execution fidelity remains unverified.
    """
    from openpyxl.utils.cell import coordinate_from_string
    if not feature_headers or set(feature_headers) != set(feature_units) or any(not unit for unit in feature_units.values()):
        raise ValueError("Each feature requires an explicit unit")
    if not target_unit:
        raise ValueError("Target unit is required")
    sheets = [sheet for sheet in workbook["sheets"] if sheet["name"] == sheet_name]
    if len(sheets) != 1:
        raise ValueError("Expected exactly one named worksheet")
    cells = {item["coordinate"]: item for item in sheets[0]["cells"]}
    headers = {str(item["value"]): coordinate_from_string(position)[0]
               for position, item in cells.items()
               if coordinate_from_string(position)[1] == header_row}
    expected = {id_header, target_header, *feature_headers.values()}
    header_values = [str(item["value"]) for position, item in cells.items()
                     if coordinate_from_string(position)[1] == header_row]
    if any(header_values.count(value) > 1 for value in expected):
        raise ValueError("Ambiguous duplicate table header")
    if expected - headers.keys():
        raise ValueError(f"Schema mismatch; missing headers: {sorted(expected - headers.keys())}")
    row_numbers = sorted({coordinate_from_string(position)[1] for position in cells
                          if coordinate_from_string(position)[1] > header_row})
    accepted, rejected = [], []

    def numeric(cell: dict[str, Any] | None, elapsed_time: bool = False) -> float:
        if cell is None:
            raise ValueError("missing")
        if cell["data_type"] == "f":
            raise ValueError("unverified_formula")
        value = cell["value"]
        if elapsed_time and isinstance(value, time):
            value = value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6
        if elapsed_time and isinstance(value, timedelta):
            value = value.total_seconds()
        if elapsed_time and isinstance(value, str):
            match = re.fullmatch(r"(\d+):([0-5]\d):([0-5]\d)", value.strip())
            if not match:
                raise ValueError("invalid_elapsed_time")
            hours, minutes, seconds = map(int, match.groups())
            value = hours * 3600 + minutes * 60 + seconds
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("not_finite_numeric")
        return float(value)

    for row_number in row_numbers:
        identifier = cells.get(f"{headers[id_header]}{row_number}")
        target_coordinate = f"{headers[target_header]}{row_number}"
        if identifier is None:
            target = cells.get(target_coordinate)
            if target is not None and isinstance(target["value"], (int, float)):
                rejected.append(dict(row=row_number, reason="missing_experiment_id"))
            continue
        try:
            target = numeric(cells.get(target_coordinate))
        except ValueError as error:
            rejected.append(dict(row=row_number, reason=f"target_{error}"))
            continue
        try:
            features = {name: numeric(cells.get(f"{headers[header]}{row_number}"), name in time_features)
                        for name, header in feature_headers.items()}
        except ValueError as error:
            rejected.append(dict(row=row_number, reason=f"feature_{error}"))
            continue
        record = dict(experiment_id=str(identifier["value"]), group_id=workbook["group"],
                      features=features, feature_units=feature_units, target=target,
                      target_unit=target_unit,
                      provenance=dict(source=workbook["source"], sha256=workbook["sha256"],
                                      sheet=sheet_name, row=row_number, target_cell=target_coordinate),
                      flags=["execution_fidelity_not_verified", "candidate_requires_agenda_review"])
        validate_observation(record)
        accepted.append(record)
    return dict(records=accepted, rejected=rejected,
                status="candidate_observations" if accepted else "abstain_no_observed_targets")
