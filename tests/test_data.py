"""Leakage and source-integrity tests use entirely artificial workbook fixtures."""
import json
from pathlib import Path

import openpyxl
import pytest

from crfml.data import PrivateStore, build_manifest, extract_numeric_table, validate_observation, freeze_returned_development_amendment


def make_sources(tmp_path):
    source = tmp_path / "source"
    for day in range(1, 11):
        folder = source / f"2020_01_{day:02d}"
        folder.mkdir(parents=True)
        book = openpyxl.Workbook()
        sheet = book.active
        sheet.title = "observations"
        sheet.append(["sample", "elapsed", "response"])
        sheet.append([f"artificial-{day}", "00:01:30", float(day)])
        book.save(folder / "observations.xlsx")
    return source


def test_holdout_cannot_be_opened_and_denial_logged(tmp_path):
    source = make_sources(tmp_path)
    manifest = build_manifest(source, tmp_path / "private")
    assert manifest["sealed_blocks"] == 7
    store = PrivateStore(tmp_path / "private")
    sealed = next(row["path"] for row in manifest["files"] if row["partition"] == "sealed_test")
    with pytest.raises(PermissionError):
        store.read_development_workbook(sealed)
    assert json.loads((tmp_path / "private/access.jsonl").read_text().splitlines()[-1])["event"] == "denied"
    with pytest.raises(PermissionError):
        store.read_development_workbook("../../outside.xlsx")


def test_changed_source_and_manifest_fail_closed(tmp_path):
    source = make_sources(tmp_path)
    build_manifest(source, tmp_path / "private")
    store = PrivateStore(tmp_path / "private")
    path = store.development_paths()[0]
    with (source / path).open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="Source differs"):
        store.read_development_workbook(path)
    (tmp_path / "private/manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="Manifest changed"):
        store.read_development_workbook(path)
    with pytest.raises(ValueError, match="integrity"):
        PrivateStore(tmp_path / "private")


def test_reservation_cannot_be_replaced_after_reading(tmp_path):
    source = make_sources(tmp_path)
    build_manifest(source, tmp_path / "private")
    with pytest.raises(FileExistsError):
        build_manifest(source, tmp_path / "private", sealed_fraction=0.60)


def test_cross_date_return_and_byte_duplicate_stay_grouped(tmp_path):
    source = make_sources(tmp_path)
    new = source / "2020_01_09" / "returned_01_01_2020.xlsx"
    new.write_bytes((source / "2020_01_01/observations.xlsx").read_bytes())
    manifest = build_manifest(source, tmp_path / "private")
    rows = {row["path"]: row for row in manifest["files"]}
    first, late = rows["2020_01_01/observations.xlsx"], rows["2020_01_09/observations.xlsx"]
    assert first["group"] == late["group"]
    assert first["partition"] == late["partition"] == "sealed_test"


def test_explicit_table_schema_and_missing_target_abstention(tmp_path):
    source = make_sources(tmp_path)
    build_manifest(source, tmp_path / "private")
    store = PrivateStore(tmp_path / "private")
    book = store.read_development_workbook(store.development_paths()[0])
    kwargs = dict(sheet_name="observations", header_row=1, id_header="sample",
                  feature_headers={"time": "elapsed"}, feature_units={"time": "s"},
                  time_features=("time",), target_header="response", target_unit="arbitrary_fixture_unit")
    result = extract_numeric_table(book, **kwargs)
    assert result["records"][0]["features"]["time"] == 90
    from datetime import time
    elapsed_cell = next(c for c in book["sheets"][0]["cells"] if c["coordinate"] == "B2")
    elapsed_cell["value"] = time(0, 2, 30)
    assert extract_numeric_table(book, **kwargs)["records"][0]["features"]["time"] == 150
    book["sheets"][0]["cells"] = [c for c in book["sheets"][0]["cells"] if c["coordinate"] != "C2"]
    result = extract_numeric_table(book, **kwargs)
    assert result["records"] == []
    assert result["status"] == "abstain_no_observed_targets"
    assert result["rejected"][0]["reason"] == "target_missing"
    with pytest.raises(ValueError, match="Schema mismatch"):
        extract_numeric_table(book, **{**kwargs, "target_header": "invented"})
    book["sheets"][0]["cells"].append(dict(coordinate="D1", value="response", data_type="s"))
    with pytest.raises(ValueError, match="duplicate"):
        extract_numeric_table(book, **kwargs)


def test_nonfinite_observation_is_not_training_data():
    row = dict(experiment_id="fixture", group_id="group", features={"a": 1.0},
               target=float("nan"), target_unit="u", provenance={"source": "fixture"}, flags=[])
    with pytest.raises(ValueError, match="finite"):
        validate_observation(row)


def test_single_metadata_amendment_preserves_original_and_minimum(tmp_path):
    source = make_sources(tmp_path)
    for day in (4, 5, 6, 7, 8, 9, 10):
        original = source / f"2020_01_{day:02d}/observations.xlsx"
        returned = original.parent / "returns/observations.xlsx"
        returned.parent.mkdir()
        returned.write_bytes(original.read_bytes())
    private = tmp_path / "private"
    build_manifest(source, private)
    before = (private / "manifest.json").read_bytes()
    stale = PrivateStore(private)
    amendment = freeze_returned_development_amendment(private, count=1, independent_review="fixture review")
    assert (private / "manifest.json").read_bytes() == before
    assert amendment["sealed_component_fraction"] == .6
    with pytest.raises(ValueError, match="fresh reader"):
        stale.read_development_workbook(stale.development_paths()[0])
    store = PrivateStore(private)
    store.read_development_workbook("2020_01_04/returns/observations.xlsx")
    with pytest.raises(PermissionError):
        store.read_development_workbook("2020_01_05/returns/observations.xlsx")
    with pytest.raises(FileExistsError):
        freeze_returned_development_amendment(private, count=1, independent_review="second")
    (private / "development_amendment.json").write_text("{}")
    with pytest.raises(ValueError, match="Amendment changed"):
        store.read_development_workbook(store.development_paths()[0])


def test_private_study_rejects_clock_reset_and_uses_chronological_prefix(tmp_path):
    from crfml.private_local import run_development_study
    source = make_sources(tmp_path)
    path = source / "2020_01_01/observations.xlsx"
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "observations"
    sheet.append(["sample", "elapsed", "response"])
    for minute in range(1, 11):
        sheet.append(["artificial-curve", f"00:{minute:02d}:00", minute * 0.1])
    sheet.append(["artificial-curve", "00:02:00", 50.0])
    sheet.append([None, "00:11:00", 60.0])
    book.save(path)
    private = tmp_path / "private"
    build_manifest(source, private)
    spec = dict(study_scope="development_trajectory_smoke_only", ordered_feature="time",
                train_fraction=.7, standardized_noise_assumption=.05,
                tables=[dict(source="2020_01_01/observations.xlsx", schema=dict(
                    sheet_name="observations", header_row=1, id_header="sample",
                    feature_headers={"time": "elapsed"}, feature_units={"time": "s"},
                    time_features=["time"], target_header="response", target_unit="fixture_unit"))])
    schema = private / "schema.json"
    schema.write_text(json.dumps(spec))
    report = run_development_study(private, schema)
    assert report["normalized_observations"] == 10
    assert {row["reason"] for row in report["exclusions"]} == {
        "missing_experiment_id", "nonincreasing_elapsed_time_ambiguous_do_not_repair"}
    trajectory = report["trajectories"][0]
    assert trajectory["train_observations"] == 7
    assert trajectory["forecast"][0]["time"] == 480
    assert trajectory["forecast"][-1]["time"] == 600
    assert report["material_generalization_status"].startswith("abstain")
