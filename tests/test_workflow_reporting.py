import base64
import json
import re

from materials_workflow.reporting import write_report


def test_offline_report_preserves_evidence_and_safely_embeds_ids(tmp_path):
    design = {"a": .2, "b": .45, "c": .35, "cure": .6, "route": "press"}
    unsafe_id = "A</script><script>alert(1)</script>"
    candidate = {"id": unsafe_id, "design": design, "cost": 1.5}
    y = {"transmission": .6, "toughness": 7.5}
    row = {"run_id": "r01", "group_id": "g01", "design": design, "targets": y}
    front = {"design": design, "targets": y, "independent_groups": 1}
    plan = {"actions": [], "trace": [], "frontier": [front], "stop_reason": "no_positive_marginal_utility"}
    summary = {"initial_groups": 1, "initial_attempts": 1, "initial_rows": 1, "gate_counts": {"use": 1},
               "selected_ids": [], "next_ids": [], "final_groups": 1, "spent": 0., "budget": 1.}
    packet = {"initial_rows": [row], "audit": [], "candidates": [candidate], "initial_plan": plan,
              "next_plan": plan, "execution": [], "diagnostics": {}}
    write_report(tmp_path, summary, packet)

    assert {p.name for p in tmp_path.iterdir()} == {"explorer.html", "REPORT.md", "decision_map.png"}
    page = (tmp_path / "explorer.html").read_text(encoding="utf-8")
    img = re.search(r'src="data:image/png;base64,([^"]+)"', page).group(1)
    assert base64.b64decode(img) == (tmp_path / "decision_map.png").read_bytes()
    assert "<script>alert(1)</script>" not in page
    record = re.search(r'<script id="data" type="application/json">(.+?)</script>', page).group(1)
    assert json.loads(record)["candidates"][0]["id"] == unsafe_id
    assert "src=\"http" not in page and "href=\"http" not in page
    assert "[Decision packet](packet.json)" in (tmp_path / "REPORT.md").read_text(encoding="utf-8")
