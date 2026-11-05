"""Create a fresh, allowlisted local publication tree; never read private inputs.

This guards accidental inclusion. Provenance labels and hashes are evidence of
reproducibility, not a proof that arbitrary maliciously relabeled data are public.
Review the final manifest and content before publishing. Export never changes Git;
the optional tracked-file check uses read-only Git commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT_FILES = {"README.md", "LICENSE", "CONTRIBUTING.md", "pyproject.toml", "requirements-lock.txt", ".gitignore", ".gitattributes"}
CI_FILE = ".github/workflows/ci.yml"
REQUIRED_ROOT_FILES = ROOT_FILES | {CI_FILE}
PUBLIC_SCRIPTS = {"scripts/export_public.py", "scripts/reproduce.py", "scripts/reproduce.ps1"}
PUBLIC_DOCS = {
    "DATA_POLICY.md", "DECISION_FOCUSED_THEORY.md", "DECISION_LEARNING.md",
    "DUAL_CONTROL_THEORY.md", "FINDINGS.md", "OPTIMIZER_DIAGNOSTICS.md",
    "PIPELINE.md", "REPRODUCIBILITY.md", "RELEASE_VERIFICATION.md",
    "TECHNICAL_VALIDATION.md", "THEORY.md",
}
RESEARCH_FILES = {
    "LITERATURE_REVIEW.md", "GITHUB_REVIEW.md", "THEORY_NOTES.md", "SEARCH_LOG.md",
    "references.bib", "bibliography.json", "github_evidence.json",
    "retrieve_openalex.ps1", "build_bibliography.py",
    "DR_REGRET_FORMULATION.md", "DUAL_CONTROL_REVIEW.md", "DECISION_FOCUSED_REVIEW.md",
    "decision_github_evidence.json",
}
OPENALEX_FILES = {
    "safe_linear_bandits.json", "uncertainty_sets.json",
    "nonstationary_chance_constraints.json", "wasserstein_regret_control.json",
    "wasserstein_dro.json", "conformal_covariate_shift.json", "safe_control_gym.json",
    "botorch.json", "moment_regret_control.json", "regret_minimization.json",
    "materials_bax.json", "safeopt_title_search.json",
    "decision_control_policies.json", "decision_convex_layers.json",
    "decision_covariance_cleaning.json", "decision_covariance_gmvp.json",
    "decision_loss_aware_dro.json", "decision_spo.json",
    "decision_task_exploration.json", "decision_task_model.json",
    "decision_uncertainty_sets.json",
}
RESULT_FILES = {
    "provenance.json", "summary.json", "predictions.json", "discovery.json",
    "control.json", "REPORT.md", "control_cost.png", "trajectory.png",
}
RESULT_GROUPS = {
    "public": {"config": "configs/study.json", "files": RESULT_FILES},
    "decision": {"config": "configs/decision_study.json", "files": {
        "summary.json", "records.json", "fits.json", "failures.json", "policies.json",
        "provenance.json", "REPORT.md", "primary_comparison.png", "covariance_tradeoff.png"}},
    "mechanism": {"config": "configs/mechanism_study.json", "files": {
        "summary.json", "records.json", "fits.json", "failures.json", "policies.json",
        "provenance.json", "REPORT.md"}},
    "reference": {"config": "configs/dual_study.json", "files": {
        "summary.json", "provenance.json", "REPORT.md"}},
}
CONFIG_FILES = {specification["config"] for specification in RESULT_GROUPS.values()}
TEXT_SUFFIXES = {".py", ".json", ".md", ".toml", ".txt", ".bib", ".ps1", ".yml"}
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_EXPORT_BYTES = 64 * 1024 * 1024


class ExportError(ValueError):
    pass


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def allowed(relative: str) -> bool:
    path = PurePosixPath(relative)
    if not path.parts or path.is_absolute():
        return False
    if relative == CI_FILE:
        return True
    if any(part.startswith(".") for part in path.parts) and relative not in {".gitignore", ".gitattributes"}:
        return False
    if relative in ROOT_FILES or relative in PUBLIC_SCRIPTS:
        return True
    if path.parts[0] in {"src", "tests"} and path.suffix == ".py":
        return True
    if relative in CONFIG_FILES:
        return True
    if len(path.parts) == 2 and path.parts[0] == "docs" and path.suffix == ".md":
        return path.name in PUBLIC_DOCS
    if len(path.parts) == 2 and path.parts[0] == "research":
        return path.name in RESEARCH_FILES
    if len(path.parts) == 3 and path.parts[:2] == ("research", "openalex"):
        return path.name in OPENALEX_FILES
    if len(path.parts) == 3 and path.parts[0] == "results" and path.parts[1] in RESULT_GROUPS:
        return path.name in RESULT_GROUPS[path.parts[1]]["files"]
    return False


def json_object(data: bytes, label: str):
    def invalid_constant(value):
        raise ExportError(f"Non-standard numeric constant in {label}")
    try:
        return json.loads(data.decode("utf-8-sig"), parse_constant=invalid_constant)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ExportError(f"Invalid UTF-8 JSON in {label}") from error


def scan_text(data: bytes, relative: str, source_root: Path) -> None:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeError as error:
        raise ExportError(f"Unexpected non-text content: {relative}") from error
    if "\x00" in text:
        raise ExportError(f"Binary/null payload in text file: {relative}")
    # Do not print matching private values or credentials into the console.
    forbidden = [
        r"(?<![A-Za-z0-9])(?i:[a-z]):[\\/]",
        r"/(?:Users|home)/[^\s/]+/",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"\bgh[pousr]_[A-Za-z0-9]{20,}\b",
        r"\bgithub_pat_[A-Za-z0-9_]{20,}\b",
        r"\bAKIA[A-Z0-9]{16}\b",
        r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b",
        r"(?i)(?:api[_-]?key|access[_-]?token|password)\s*[:=]\s*[\"'][A-Za-z0-9_./+\-=]{16,}[\"']",
    ]
    if any(re.search(pattern, text) for pattern in forbidden):
        raise ExportError(f"Private path or credential pattern detected: {relative}")
    for root_string in {str(source_root), source_root.as_posix()}:
        if root_string.lower() in text.lower():
            raise ExportError(f"Workspace path detected: {relative}")


def validate_results(files: dict[str, bytes]) -> None:
    code = {PurePosixPath(name).name: digest(data) for name, data in files.items()
            if PurePosixPath(name).parent == PurePosixPath("src/crfml") and name.endswith(".py")}
    if {name for name in files if name.startswith("src/")} != {"src/crfml/" + name for name in code}:
        raise ExportError("Source tree contains modules outside the producer's hashed source set")
    for group, specification in RESULT_GROUPS.items():
        prefix = f"results/{group}/"
        expected = specification["files"]
        config_path = specification["config"]
        if {name[len(prefix):] for name in files if name.startswith(prefix)} != expected:
            raise ExportError(f"Required synthetic result set is incomplete: {group}")
        if config_path not in files:
            raise ExportError(f"Required synthetic configuration missing: {group}")
        provenance = json_object(files[prefix + "provenance.json"], prefix + "provenance")
        summary = json_object(files[prefix + "summary.json"], prefix + "summary")
        config = json_object(files[config_path], config_path)
        for label, value in (("provenance", provenance), ("summary", summary), ("config", config)):
            if not isinstance(value, dict) or value.get("origin") != "public_synthetic_only":
                raise ExportError(f"Missing public synthetic origin: {group}/{label}")
        if provenance.get("no_private_data_loaded") is not True:
            raise ExportError(f"Provenance must explicitly declare no private data loaded: {group}")
        if summary.get("config") != config:
            raise ExportError(f"Summary configuration differs from current study configuration: {group}")
        if provenance.get("config_sha256") != digest(files[config_path]):
            raise ExportError(f"Configuration hash is stale: {group}")
        if not code or provenance.get("source_sha256") != code:
            raise ExportError(f"Source hashes are stale or incomplete; reproduce the study: {group}")
        # A producer that records start hashes must prove its sources stayed
        # unchanged during the run, not merely stamp the final source tree.
        if "source_start_sha256" in provenance and (
                provenance["source_start_sha256"] != code
                or provenance.get("sources_unchanged_during_run") is not True):
            raise ExportError(f"Source tree changed during the study: {group}")
        if group == "mechanism":
            decision_path = RESULT_GROUPS["decision"]["config"]
            decision_config = json_object(files[decision_path], decision_path)
            if provenance.get("decision_config_sha256") != digest(files[decision_path]):
                raise ExportError("Primary decision configuration hash is stale: mechanism")
            if summary.get("primary_decision_config") != decision_config:
                raise ExportError("Primary decision configuration differs: mechanism")
        artifacts = {name: digest(files[prefix + name]) for name in expected if name != "provenance.json"}
        if provenance.get("artifact_sha256") != artifacts:
            raise ExportError(f"Result artifact hashes are missing or stale; regenerate the final report: {group}")
        for name in expected:
            if name.endswith(".json"):
                json_object(files[prefix + name], prefix + name)
            elif name.endswith(".png") and not files[prefix + name].startswith(b"\x89PNG\r\n\x1a\n"):
                raise ExportError(f"Invalid PNG signature: {prefix + name}")


def collect_public_files(source_root: str | Path) -> dict[str, bytes]:
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise ExportError("Source project directory does not exist")
    files: dict[str, bytes] = {}
    # Enumerate allowed roots only: no traversal or reading of private/, runtime,
    # local caches, raw audits, outputs, model weights, or Git metadata.
    candidates = [root / name for name in ROOT_FILES if (root / name).is_file()]
    if (root / CI_FILE).is_file():
        candidates.append(root / CI_FILE)
    directories = ("src", "tests", "configs", "docs", "research", "scripts") + tuple(
        "results/" + group for group in RESULT_GROUPS)
    for directory in directories:
        subtree = root / directory
        if subtree.exists():
            if subtree.is_symlink() or not subtree.resolve().is_relative_to(root):
                raise ExportError(f"Linked export subtree rejected: {directory}")
            subtree_files = [p for p in subtree.rglob("*") if p.is_file()]
            if directory.startswith("results/") and any(
                    not allowed(p.relative_to(root).as_posix()) for p in subtree_files):
                raise ExportError(f"Unexpected artifact in required synthetic result set: {directory}")
            candidates.extend(subtree_files)
    for path in sorted(set(candidates)):
        relative = path.relative_to(root).as_posix()
        if not allowed(relative):
            continue
        if path.is_symlink() or path.resolve() != path.absolute() or not path.resolve().is_relative_to(root):
            raise ExportError(f"Linked source rejected: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ExportError(f"File exceeds publication budget: {relative}")
        data = path.read_bytes()
        if path.suffix in TEXT_SUFFIXES or path.name in ROOT_FILES:
            scan_text(data, relative, root)
        if relative.startswith("configs/"):
            configuration = json_object(data, relative)
            if not isinstance(configuration, dict) or configuration.get("origin") != "public_synthetic_only":
                raise ExportError(f"Non-public configuration rejected: {relative}")
        if relative.startswith("research/openalex/"):
            snapshot = json_object(data, relative)
            if not isinstance(snapshot, dict) or not str(snapshot.get("url", "")).startswith("https://api.openalex.org/"):
                raise ExportError(f"OpenAlex retrieval provenance missing: {relative}")
        files[relative] = data
    missing = REQUIRED_ROOT_FILES - files.keys()
    if missing:
        raise ExportError(f"Required publication files missing: {sorted(missing)}")
    if sum(map(len, files.values())) > MAX_EXPORT_BYTES:
        raise ExportError("Publication tree exceeds total byte budget")
    validate_results(files)
    return files


def export_public(source_root: str | Path, destination: str | Path | None = None) -> dict:
    root = Path(source_root).resolve()
    target = Path(destination).resolve() if destination is not None else root / "dist/public-repo"
    dist = root / "dist"
    if target == dist or not target.is_relative_to(dist) or not dist.resolve().is_relative_to(root):
        raise ExportError("Destination must be a fresh directory inside project dist/")
    if target.exists():
        raise FileExistsError("Destination already exists; choose a new directory, never overwrite")
    files = collect_public_files(root)
    manifest = dict(schema_version=3, status="local_review_copy_not_published",
                    publication_profile="neutral_synthetic_workflow",
                    origin="public_synthetic_only", no_private_tree_read=True,
                    files=[dict(path=name, bytes=len(data), sha256=digest(data)) for name, data in sorted(files.items())],
                    notes="Allowlisted copy only. No source Git history copied, no remote or push. Review content and provenance before publication.")
    # Validation finishes before any destination is created. The final manifest
    # marks a completed export; partial failures remain visible and are not erased.
    target.mkdir(parents=True, exist_ok=False)
    for relative, data in files.items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (target / "PUBLIC_EXPORT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for row in manifest["files"]:
        if digest((target / row["path"]).read_bytes()) != row["sha256"]:
            raise ExportError("Destination verification failed")
    return manifest


def check_tracked(source_root: str | Path) -> dict:
    """Check tracked paths and current public artifacts without modifying Git.

    This checks the repository root's current working tree, not historical blobs
    or staged content that differs from that tree. CI should run it on a checkout.
    Reject disallowed tracked paths before reading files; never echo those paths.
    """
    root = Path(source_root).resolve()
    try:
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                             capture_output=True, check=True)
        if Path(os.fsdecode(top.stdout).strip()).resolve() != root:
            raise ExportError("Tracked-file validation must run at the Git repository root")
        listing = subprocess.run(["git", "-C", str(root), "ls-files", "-z"],
                                 capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        # Git errors can contain local/private paths; keep them out of CI output.
        raise ExportError("Tracked-file validation requires an accessible Git repository") from None
    tracked = {os.fsdecode(name) for name in listing.stdout.split(b"\0") if name}
    manifest_name = "PUBLIC_EXPORT_MANIFEST.json"
    disallowed_count = sum(not allowed(name) and name != manifest_name for name in tracked)
    if disallowed_count:
        raise ExportError(f"Publication allowlist rejected {disallowed_count} tracked path(s); names withheld")
    files = collect_public_files(root)
    expected = set(files)
    if manifest_name in tracked:
        path = root / manifest_name
        if not path.is_file() or path.is_symlink() or path.resolve() != path.absolute():
            raise ExportError("Tracked publication manifest is missing or linked")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ExportError("Publication manifest exceeds byte budget")
        data = path.read_bytes()
        scan_text(data, manifest_name, root)
        manifest = json_object(data, "publication manifest")
        if (not isinstance(manifest, dict)
                or manifest.get("publication_profile") != "neutral_synthetic_workflow"
                or manifest.get("origin") != "public_synthetic_only"
                or not isinstance(manifest.get("files"), list)):
            raise ExportError("Publication manifest lacks the neutral synthetic profile")
        if any(not isinstance(row, dict) or not isinstance(row.get("path"), str)
               or not allowed(row["path"]) for row in manifest["files"]):
            raise ExportError("Publication manifest contains a non-allowlisted entry; names withheld")
        expected.add(manifest_name)
    if tracked != expected:
        raise ExportError("Tracked paths do not match the validated public working tree; names withheld")
    return dict(status="tracked_files_validated", publication_profile="neutral_synthetic_workflow",
                tracked_file_count=len(tracked), public_file_count=len(files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--check-tracked", action="store_true",
                        help="Read-only check of Git tracked paths and current public artifact provenance")
    args = parser.parse_args()
    if args.check_tracked:
        if args.destination is not None:
            parser.error("--destination cannot be combined with --check-tracked")
        print(json.dumps(check_tracked(args.source)))
    else:
        result = export_public(args.source, args.destination)
        print(json.dumps({"status": result["status"], "file_count": len(result["files"])}))
