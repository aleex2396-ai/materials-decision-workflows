"""Publication guard tests use artificial files, never real private records."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts/export_public.py"
SPEC = importlib.util.spec_from_file_location("public_export", MODULE)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def write(root, path, content):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content if isinstance(content, bytes) else content.encode())


def synthetic_project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    for name in exporter.ROOT_FILES:
        write(root, name, "public fixture\n")
    write(root, exporter.CI_FILE, "name: Synthetic fixture CI\n")
    write(root, "src/crfml/example.py", "VALUE = 1\n")
    configs = {}
    for group, specification in exporter.RESULT_GROUPS.items():
        config = {"origin": "public_synthetic_only", "seeds": [0, 1], "study": group}
        configs[group] = config
        write(root, specification["config"], json.dumps(config, indent=2) + "\n")
    for group, specification in exporter.RESULT_GROUPS.items():
        prefix = f"results/{group}/"
        summary = {"origin": "public_synthetic_only", "config": configs[group]}
        if group == "mechanism":
            summary["primary_decision_config"] = configs["decision"]
        write(root, prefix + "summary.json", json.dumps(summary, indent=2) + "\n")
        for name in specification["files"] - {"summary.json", "provenance.json"}:
            if name.endswith(".json"):
                write(root, prefix + name, json.dumps([{"synthetic_fixture": group}], indent=2) + "\n")
            elif name.endswith(".md"):
                write(root, prefix + name, "Synthetic fixture study\n\nMultiline report.\n")
            else:
                write(root, prefix + name, b"\x89PNG\r\n\x1a\nsynthetic_test_payload")
        provenance = dict(origin="public_synthetic_only", no_private_data_loaded=True,
                          config_sha256=exporter.digest((root / specification["config"]).read_bytes()),
                          source_sha256={"example.py": exporter.digest((root / "src/crfml/example.py").read_bytes())},
                          artifact_sha256={name: exporter.digest((root / prefix / name).read_bytes())
                                           for name in specification["files"] if name != "provenance.json"})
        if group == "mechanism":
            provenance["decision_config_sha256"] = exporter.digest(
                (root / exporter.RESULT_GROUPS["decision"]["config"]).read_bytes())
            provenance["source_start_sha256"] = provenance["source_sha256"].copy()
            provenance["sources_unchanged_during_run"] = True
        write(root, prefix + "provenance.json", json.dumps(provenance, indent=2) + "\n")
    return root


def test_export_omits_private_raw_audits_runtime_and_git(tmp_path):
    root = synthetic_project(tmp_path)
    for name in ("private/data.json", "audits/raw.md", ".runtime/cache.py", ".git/config",
                 "docs/PRIVATE_NOTES.md", "docs/UNPUBLISHED_REVIEW.md",
                 "docs/INTERNAL_ASSESSMENT.md", "docs/unreviewed.md",
                 "research/openalex/author_catalog.json",
                 "research/unreviewed.json", "docs/secret.xlsx", "scripts/run_local.ps1",
                 "configs/unreviewed.json", "results/unreviewed/summary.json",
                 ".github/workflows/unreviewed.yml", ".github/hidden-secret.txt"):
        write(root, name, "PRIVATE_FIXTURE_CANARY")
    manifest = exporter.export_public(root)
    target = root / "dist/public-repo"
    paths = {row["path"] for row in manifest["files"]}
    assert "README.md" in paths
    assert exporter.CI_FILE in paths
    for group, specification in exporter.RESULT_GROUPS.items():
        assert {p.split("/", 2)[-1] for p in paths if p.startswith(f"results/{group}/")} == specification["files"]
    assert all("PRIVATE_FIXTURE_CANARY" not in path.read_text(errors="ignore")
               for path in target.rglob("*") if path.is_file())
    for row in manifest["files"]:
        assert exporter.digest((target / row["path"]).read_bytes()) == row["sha256"]
    assert not (target / "private").exists()
    with pytest.raises(FileExistsError):
        exporter.export_public(root)
    original_manifest = (target / "PUBLIC_EXPORT_MANIFEST.json").read_bytes()
    release = root / "dist/release-v2"
    exporter.export_public(root, release)
    assert (release / "PUBLIC_EXPORT_MANIFEST.json").is_file()
    assert (target / "PUBLIC_EXPORT_MANIFEST.json").read_bytes() == original_manifest


@pytest.mark.parametrize("mutation", ["source", "config", "result", "origin", "unknown_result"])
def test_stale_or_false_provenance_fails_before_creating_destination(tmp_path, mutation):
    root = synthetic_project(tmp_path)
    if mutation == "source":
        write(root, "src/crfml/example.py", "VALUE = 2\n")
    elif mutation == "config":
        write(root, "configs/study.json", '{"origin":"public_synthetic_only","seeds":[9]}')
    elif mutation == "result":
        write(root, "results/public/trajectory.png", b"\x89PNG\r\n\x1a\nreplaced")
    elif mutation == "origin":
        path = root / "results/public/provenance.json"
        record = json.loads(path.read_text())
        record["origin"] = "private"
        path.write_text(json.dumps(record))
    else:
        (root / "results/public/summary.json").rename(root / "results/public/unreviewed.json")
    with pytest.raises(exporter.ExportError):
        exporter.export_public(root)
    assert not (root / "dist/public-repo").exists()


def test_path_credential_and_workspace_strings_rejected(tmp_path):
    root = synthetic_project(tmp_path)
    for text in ("Z:" + chr(92) + "confidential" + chr(92) + "records", str(root),
                 "ghp_" + "a" * 30):
        write(root, "README.md", text)
        with pytest.raises(exporter.ExportError):
            exporter.collect_public_files(root)


def test_unknown_openalex_provenance_rejected(tmp_path):
    root = synthetic_project(tmp_path)
    write(root, "research/openalex/botorch.json", '{"url":"https://untrusted.invalid/","response":{}}')
    with pytest.raises(exporter.ExportError, match="OpenAlex"):
        exporter.collect_public_files(root)


def test_destination_must_be_fresh_local_dist_child(tmp_path):
    root = synthetic_project(tmp_path)
    with pytest.raises(exporter.ExportError, match="Destination"):
        exporter.export_public(root, tmp_path / "elsewhere")
    assert not (tmp_path / "elsewhere").exists()


def test_unhashed_nested_source_and_private_configuration_rejected(tmp_path):
    root = synthetic_project(tmp_path)
    write(root, "src/crfml/nested/unhashed.py", "VALUE = 1\n")
    with pytest.raises(exporter.ExportError, match="hashed source"):
        exporter.collect_public_files(root)
    write(root, "configs/mechanism_study.json", '{"origin":"private"}')
    with pytest.raises(exporter.ExportError, match="configuration"):
        exporter.collect_public_files(root)


@pytest.mark.parametrize("group", exporter.RESULT_GROUPS)
@pytest.mark.parametrize("mutation", ["source_hash", "config_hash", "artifact", "missing_artifact",
                                      "extra_artifact", "origin", "private_access"])
def test_each_result_group_fails_closed_before_destination_exists(tmp_path, group, mutation):
    root = synthetic_project(tmp_path)
    prefix = f"results/{group}/"
    path = root / prefix / "provenance.json"
    provenance = json.loads(path.read_text())
    if mutation == "source_hash":
        provenance["source_sha256"]["example.py"] = "0"*64
    elif mutation == "config_hash":
        provenance["config_sha256"] = "0"*64
    elif mutation == "artifact":
        # Relabeling an accidentally copied artifact does not restore its hash.
        summary = json.loads((root / prefix / "summary.json").read_text())
        summary["accidentally_copied_fixture"] = [13, 27]
        write(root, prefix + "summary.json", json.dumps(summary, indent=2))
    elif mutation == "missing_artifact":
        (root / prefix / "REPORT.md").unlink()
    elif mutation == "extra_artifact":
        write(root, prefix + "unreviewed.json", '{"origin":"public_synthetic_only"}')
    elif mutation == "origin":
        provenance["origin"] = "private"
    else:
        provenance["no_private_data_loaded"] = False
    path.write_text(json.dumps(provenance, indent=2))
    destination = root / "dist/release-v2"
    with pytest.raises(exporter.ExportError):
        exporter.export_public(root, destination)
    assert not destination.exists()


@pytest.mark.parametrize("mutation", ["decision_hash", "decision_summary", "source_start", "source_changed"])
def test_mechanism_dependency_and_run_source_hashes_are_required(tmp_path, mutation):
    root = synthetic_project(tmp_path)
    provenance_path = root / "results/mechanism/provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if mutation == "decision_hash":
        provenance["decision_config_sha256"] = "0"*64
    elif mutation == "decision_summary":
        summary_path = root / "results/mechanism/summary.json"
        summary = json.loads(summary_path.read_text())
        summary["primary_decision_config"]["seeds"] = [9]
        summary_path.write_text(json.dumps(summary, indent=2))
        provenance["artifact_sha256"]["summary.json"] = exporter.digest(summary_path.read_bytes())
    elif mutation == "source_start":
        provenance["source_start_sha256"]["example.py"] = "0"*64
    else:
        provenance["sources_unchanged_during_run"] = False
    provenance_path.write_text(json.dumps(provenance, indent=2))
    with pytest.raises(exporter.ExportError):
        exporter.collect_public_files(root)


def test_all_curated_decision_research_and_no_other_research_are_included(tmp_path):
    root = synthetic_project(tmp_path)
    for name in ("DR_REGRET_FORMULATION.md", "DUAL_CONTROL_REVIEW.md", "DECISION_FOCUSED_REVIEW.md"):
        write(root, "research/" + name, "Public primary-source research fixture\n")
    write(root, "research/decision_github_evidence.json", '{"public_fixture":true}')
    expected = {name for name in exporter.OPENALEX_FILES if name.startswith("decision_")}
    assert len(expected) == 9
    for name in expected:
        write(root, "research/openalex/" + name, json.dumps({
            "url": "https://api.openalex.org/works?search=public", "response": {}}, indent=2))
    write(root, "research/openalex/decision_unreviewed.json", "DO_NOT_EXPORT")
    files = exporter.collect_public_files(root)
    assert all("research/openalex/" + name in files for name in expected)
    assert "research/decision_github_evidence.json" in files
    assert "research/openalex/decision_unreviewed.json" not in files


def test_size_budgets_remain_enforced_for_new_result_groups(tmp_path, monkeypatch):
    root = synthetic_project(tmp_path)
    monkeypatch.setattr(exporter, "MAX_FILE_BYTES", 10)
    with pytest.raises(exporter.ExportError, match="File exceeds"):
        exporter.collect_public_files(root)
    monkeypatch.setattr(exporter, "MAX_FILE_BYTES", 20*1024*1024)
    monkeypatch.setattr(exporter, "MAX_EXPORT_BYTES", 10)
    with pytest.raises(exporter.ExportError, match="total byte budget"):
        exporter.collect_public_files(root)


def init_tracked_fixture(root):
    if shutil.which("git") is None:
        pytest.skip("Git is needed for tracked-file integration tests")
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "add", "--all"], check=True, capture_output=True)


def test_tracked_check_accepts_neutral_export_and_does_not_change_index(tmp_path):
    source = synthetic_project(tmp_path)
    exporter.export_public(source)
    root = source / "dist/public-repo"
    init_tracked_fixture(root)
    before = (root / ".git/index").read_bytes()
    result = exporter.check_tracked(root)
    assert result["status"] == "tracked_files_validated"
    assert result["tracked_file_count"] == len(exporter.collect_public_files(root)) + 1
    assert (root / ".git/index").read_bytes() == before


def test_tracked_check_rejects_force_added_ignored_private_file_without_reading_or_naming_it(tmp_path, monkeypatch):
    root = synthetic_project(tmp_path)
    write(root, ".gitignore", "private/\n")
    private = "private/customer_fixture_export.csv"
    write(root, private, "PRIVATE_FIXTURE_CANARY")
    init_tracked_fixture(root)
    subprocess.run(["git", "-C", str(root), "add", "--force", "--", private], check=True, capture_output=True)
    original_read = Path.read_bytes
    def guarded_read(path):
        if path == root / private:
            raise AssertionError("Private contents must not be read")
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    with pytest.raises(exporter.ExportError, match="rejected 1 tracked path") as error:
        exporter.check_tracked(root)
    assert private not in str(error.value)
    assert "customer_fixture" not in str(error.value)


@pytest.mark.parametrize("failure", ["missing_git", "not_repository"])
def test_tracked_git_errors_hide_raw_output(tmp_path, monkeypatch, failure):
    def fail(*args, **kwargs):
        if failure == "missing_git":
            raise FileNotFoundError("PRIVATE_FIXTURE_CANARY")
        raise subprocess.CalledProcessError(128, args[0], stderr=b"PRIVATE_FIXTURE_CANARY")
    monkeypatch.setattr(exporter.subprocess, "run", fail)
    with pytest.raises(exporter.ExportError, match="accessible Git repository") as error:
        exporter.check_tracked(tmp_path)
    assert "PRIVATE_FIXTURE_CANARY" not in str(error.value)


def test_tracked_check_keeps_current_artifact_validation(tmp_path):
    root = synthetic_project(tmp_path)
    init_tracked_fixture(root)
    write(root, "results/decision/records.json", '[{"changed_fixture":true}]')
    with pytest.raises(exporter.ExportError, match="artifact hashes"):
        exporter.check_tracked(root)


@pytest.mark.parametrize("entry", ["private/customer_fixture_export.csv", ""])
def test_tracked_check_rejects_manifest_with_private_entry_without_naming_it(tmp_path, entry):
    source = synthetic_project(tmp_path)
    exporter.export_public(source)
    root = source / "dist/public-repo"
    path = root / "PUBLIC_EXPORT_MANIFEST.json"
    manifest = json.loads(path.read_text())
    manifest["files"].append({"path": entry})
    path.write_text(json.dumps(manifest))
    init_tracked_fixture(root)
    with pytest.raises(exporter.ExportError, match="non-allowlisted entry") as error:
        exporter.check_tracked(root)
    assert "customer_fixture" not in str(error.value)
