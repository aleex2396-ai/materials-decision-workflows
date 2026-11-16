"""Reproducible figure, report and offline explorer for the synthetic membrane study."""
from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np


INK = "#19343b"
MUTED = "#455f63"
TEAL = "#116f67"
ORANGE = "#bd4a24"
PAPER = "#fbfaf6"
GRID = "#d3ded9"
MEASURED = "#788f95"


def _xy(design):
    """Left, right and top vertices are pure A, B and C."""
    return np.array([design["b"] + .5 * design["c"], math.sqrt(3) / 2 * design["c"]])


def _chart(summary, packet):
    rows = packet["initial_rows"]
    candidates = packet["candidates"]
    actions = packet["initial_plan"]["actions"]
    recorded = packet["execution"]
    # Use explicit axes so raster positioning and headers stay stable across backends.
    fig = Figure(figsize=(12.4, 6.25), dpi=145, facecolor=PAPER)
    FigureCanvasAgg(fig)
    fig.text(.046, .936, "Mixture design and measured performance", color=INK, weight="bold", size=23)
    fig.text(.047, .884, "Synthetic optical membranes", color=TEAL, weight="bold", size=11.5)
    fig.text(.047, .824, f"{summary['initial_groups']} initial preparation groups"
             f"     /     {len(candidates)} candidate designs"
             f"     /     {len(actions)} budgeted selections", color=MUTED, size=10.4)
    fig.add_artist(matplotlib.lines.Line2D([.047, .953], [.79, .79], transform=fig.transFigure,
                                            lw=1, color=GRID))
    for x, title in ((.047, "Mixture composition"), (.532, "Performance trade-off")):
        fig.text(x, .736, title, weight="bold", color=INK, size=14)
    triangle = fig.add_axes([.052, .222, .405, .457], facecolor=PAPER)
    ax = fig.add_axes([.555, .225, .387, .454], facecolor=PAPER)
    triangle.set_aspect("equal")
    triangle.set_xlim(-.09, 1.09)
    triangle.set_ylim(-.07, .94)
    triangle.axis("off")
    height = math.sqrt(3) / 2
    for t in (.25, .5, .75):
        for start, end in (((t, 0), (.5 + .5*t, height * (1-t))),
                           ((1-t, 0), (.5 - .5*t, height * (1-t))),
                           ((.5*t, height*t), (1-.5*t, height*t))):
            triangle.plot([start[0], end[0]], [start[1], end[1]], lw=.65, color=GRID, zorder=0)
    triangle.plot([0, 1, .5, 0], [0, 0, height, 0], color=MUTED, lw=1.1, zorder=1)
    for x, y, label in ((-.02, -.035, "A"), (1.02, -.035, "B"), (.5, .919, "C")):
        triangle.text(x, y, label, color=INK, weight="bold", ha="center", va="center", size=11)
    for source, label, marker, color, size in (
            (candidates, "Candidate", "o", "#becdca", 15),
            (rows, "Measured", ".", MEASURED, 26),
            (actions, "Selected", "D", ORANGE, 53)):
        xy = np.array([_xy(r["design"]) for r in source]).reshape(-1, 2)
        triangle.scatter(*xy.T, s=size, marker=marker, c=color,
                         edgecolors=PAPER if marker == "D" else "none", linewidths=.8, label=label)
    # Space direct labels by height; leader lines keep similarly constituted designs distinct.
    placements = []
    for act in sorted(actions, key=lambda a: _xy(a["design"])[1]):
        x, y = _xy(act["design"])
        side = 1 if x < .69 else -1
        y_text = y + .028
        for prior_x, prior_y in placements:
            if abs(x - prior_x) < .21 and abs(y_text - prior_y) < .065:
                y_text = prior_y + .065
        placements.append((x, y_text))
        triangle.annotate(act["id"], (x, y), (x + side * .07, y_text), textcoords="data",
                          ha="left" if side == 1 else "right", va="bottom", weight="bold",
                          color="#8a361e", size=8, bbox=dict(facecolor=PAPER, edgecolor="none", pad=.4),
                          arrowprops=dict(arrowstyle="-", color=ORANGE, lw=.7),
                          annotation_clip=False)
    triangle.legend(loc="upper center", bbox_to_anchor=(.5, -.19), ncol=3, frameon=False,
                    handletextpad=.35, columnspacing=1.1, fontsize=9, labelcolor=MUTED)

    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=3, labelsize=8.5)
    ax.grid(color=GRID, lw=.6, alpha=.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Light transmission (fraction)", size=9.5, color=INK, labelpad=5)
    ax.set_ylabel("Toughness (kJ/m²)", size=9.5, color=INK, labelpad=5)
    ax.scatter([r["targets"]["transmission"] for r in rows], [r["targets"]["toughness"] for r in rows],
               s=21, color=MEASURED, alpha=.66, linewidths=0, label="Initial measurements")
    points = sorted((p["targets"]["transmission"], p["targets"]["toughness"])
                    for p in packet["initial_plan"]["frontier"])
    if points:
        x, y = np.asarray(points).T
        ax.step(x, y, where="pre", color=TEAL, lw=1.35, alpha=.9)
        ax.scatter(x, y, s=24, facecolor=PAPER, edgecolor=TEAL, lw=1.3, label="Measured frontier")
    predictions, observed = [], []
    for r in recorded:
        means = r["pre_run_prediction"]["targets"]
        pred = [means[k]["mean"] for k in ("transmission", "toughness")]
        predictions.append(pred)
        if not r.get("observed"):
            continue
        obs = [r["observed"][k] for k in ("transmission", "toughness")]
        observed.append(obs)
        ax.plot(*np.asarray([pred, obs]).T, ls=(0, (2, 2)), lw=1.2, color=ORANGE, zorder=3)
    if predictions:
        ax.scatter(*np.asarray(predictions).T, marker="D", facecolor=PAPER, edgecolor=ORANGE, lw=1.6,
                   s=45, zorder=4, label="Predicted mean")
    if observed:
        ax.scatter(*np.asarray(observed).T, marker="X", facecolor=ORANGE, edgecolor=PAPER, lw=.5,
                   s=65, zorder=5, label="Returned result")
    ax.margins(.065, .10)
    ax.legend(loc="upper center", bbox_to_anchor=(.49, -.30), ncol=2, frameon=False, fontsize=8.5,
              labelcolor=MUTED, handletextpad=.35, columnspacing=1.15)
    fig.text(.047, .046, "Each marker in the triangle gives the A/B/C fractions. Cure setting and process route appear in the explorer.",
             color=MUTED, size=9)
    image = io.BytesIO()
    fig.savefig(image, format="png", dpi=145, metadata={"Software": "materials_workflow"})
    return image.getvalue()


def _metrics(packet):
    metrics = packet.get("diagnostics", {}).get("validation", {}).get("targets", {})
    return [(k, metrics[k]["model_mae"], metrics[k]["mean_baseline_mae"])
            for k in ("transmission", "toughness") if k in metrics]


def _markdown(summary, packet):
    selected = ", ".join(summary["selected_ids"]) or "none"
    following = ", ".join(summary["next_ids"]) or "none"
    lines = [
        "# Optical membrane campaign", "",
        "This public study uses independently generated synthetic optical composites. "
        "The simulator produces preparation groups, instrument returns and partially observed material descriptors.", "",
        "![Mixture compositions and the measured performance frontier](decision_map.png)", "",
        "[Offline campaign explorer](explorer.html) · [Summary](summary.json) · [Decision packet](packet.json)", "",
        "To use the explorer from GitHub, download the HTML file and open it in a browser. All evidence and figures are embedded.", "",
        "## From raw returns to training data", "",
        f"The first intake received {summary['initial_attempts']} attempts. The performance model used "
        f"{summary['initial_rows']} eligible measurements from {summary['initial_groups']} preparation groups. "
        "Measurements from a shared preparation stay together during validation.", "",
        "| Intake action | Initial receipts |", "|---|---:|",
    ]
    lines += [f"| {action.replace('_', ' ')} | {count} |" for action, count in sorted(summary["gate_counts"].items())]
    views = packet.get("descriptor_views")
    if views:
        counts = views["before"]["counts"]
        lines += ["", "## Completing supplier properties", "",
                  f"The first feature view contains {counts['observed']} measured descriptor entries and "
                  f"{counts['conditional']} estimates conditioned on other known supplier properties. "
                  "The database stores the estimates separately with their units, uncertainty and training-snapshot identity.", ""]
    example = packet.get("prediction_example", {})
    if "candidate" in example and "prediction" in example:
        descriptors = example["prediction"].get("descriptors", {})
        if descriptors:
            lines += [f"Supplier card for **{example['candidate']['id']}** before the experiment.", "",
                      "| Property | Source | Value or estimate | Imputation SD |", "|---|---|---:|---:|"]
            units = {"density": "g/cm³", "scattering": "1/mm", "viscosity": "Pa s"}
            for name, estimate in descriptors.items():
                value = "unavailable" if estimate["mean"] is None else f"{estimate['mean']:.3f}"
                sd = "—" if estimate["sd"] is None else f"{estimate['sd']:.3f}"
                lines.append(f"| {name.title()} ({units[name]}) | {estimate['source']} | {value} | {sd} |")
    if "descriptor_assay_decision" in packet:
        decision = packet["descriptor_assay_decision"]["recommended_action"]
        lines += ["", f"The separate assay-pricing decision chose **{decision['action']}** "
                  f"for **{decision.get('candidate_id') or 'no eligible candidate'}** "
                  f"with each proposed assay priced at {summary['config']['descriptor_assay_cost_in_utility_units']:.3f} utility units. "
                  "The decision packet gives the expected gains for the alternative assays."]
    lines += ["", "## Predicting both performance outcomes", "",
              "The model propagates uncertainty in missing stock descriptors into both endpoint predictions. "
              "The table reports mean absolute errors from whole-preparation-group cross-validation. "
              "Predictive standard deviations describe model-based uncertainty. Coverage on new physical preparations is unverified.", "",
              "| Outcome | Model MAE | Training-mean MAE |", "|---|---:|---:|"]
    labels = {"transmission": "Transmission (fraction)", "toughness": "Toughness (kJ/m²)"}
    lines += [f"| {labels[name]} | {model:.3f} | {base:.3f} |" for name, model, base in _metrics(packet)]
    lines += ["", "## Selecting experiments and learning from the returns", "",
              f"The first batch selected **{selected}** and spent **{summary['spent']:.2f} of {summary['budget']:.2f}** synthetic cost units. "
              "Each proposal has a record of its expected change in Pareto area, estimated probability of a successful make, "
              "distance from pending and selected designs, and cost.", "",
              "| Experiment | Predicted transmission | Returned transmission | Predicted toughness (kJ/m²) | Returned toughness (kJ/m²) |",
              "|---|---:|---:|---:|---:|"]
    for r in packet["execution"]:
        p, y = r["pre_run_prediction"]["targets"], r.get("observed")
        t = f"{y['transmission']:.3f}" if y else "withheld"
        k = f"{y['toughness']:.2f}" if y else "withheld"
        lines.append(f"| {r['candidate_id']} | {p['transmission']['mean']:.3f} | {t} | "
                     f"{p['toughness']['mean']:.2f} | {k} |")
    lines += ["", f"After simulated execution, the refit used {summary['final_groups']} preparation groups. "
              f"The next proposed batch contains **{following}**.", ""]
    replay = packet.get("gate_replay", {})
    if "settled" in replay:
        lines += ["## Checking decisions on doubtful returns", "",
                  f"On {replay['total']} labelled synthetic challenge cases, the fixed gate settled {replay['settled']}, "
                  f"with {replay['harmful_admissions']} wrongly admitted and {replay['harmful_exclusions']} wrongly withheld. "
                  "Each scenario was constructed. Its frequency in the challenge set does not estimate its frequency in a lab.", ""]
    shadow = packet.get("shadow_study") or {}
    if shadow.get("evaluation"):
        lines += ["The shadow classifier and fixed rules also see the same evidence on independent synthetic "
                  "preparations. Coverage is the share of all cases settled. Error rate is the share of settled "
                  "decisions with an incorrect action. The shifted generator challenges transfer.", "",
                  "| Evidence | Policy | Settled / total | Incorrect / settled |",
                  "|---|---|---:|---:|"]
        for condition, policies in shadow["evaluation"].items():
            for policy, met in policies.items():
                label = "Shadow classifier" if policy == "model" else "Fixed rules"
                lines.append(f"| {condition} | {label} | {met['settled']} / {met['groups']} "
                             f"| {met['settled_errors']} / {met['settled']} |")
        lines.append("")
    lines += ["Inputs, score components, training snapshots and gate evaluations are available in the decision packet. "
              "The results describe this simulator and candidate library.", ""]
    return "\n".join(lines)


_CSS = """
:root{--ink:#19343b;--muted:#4e686b;--line:#cfdbd5;--paper:#fbfaf6;--teal:#116f67;--orange:#9c3c1f;
font-family:Inter,system-ui,sans-serif;color:var(--ink);background:var(--paper)}
*{box-sizing:border-box}body{margin:0}main{max-width:1060px;margin:auto;padding:46px 30px 60px}
header{border-top:4px solid var(--teal);padding-top:22px}.eyebrow{margin:0;color:var(--teal);font-weight:700}
h1{font-size:clamp(2.2rem,5vw,3.8rem);line-height:1.05;letter-spacing:-.055em;margin:10px 0}
h2{font-size:1.45rem;margin:0 0 14px}h3{font-size:1rem;margin:20px 0 8px}
p{line-height:1.52}header p:last-child{max-width:63ch;color:var(--muted);margin-bottom:24px}
figure{margin:0 0 20px;border:1px solid var(--line);border-radius:6px;overflow:hidden}
figure img{width:100%;height:auto;display:block}nav{display:flex;gap:4px;flex-wrap:wrap;border-bottom:1px solid var(--line);margin:28px 0}
button,select{font:inherit;color:var(--ink)}nav button{padding:10px 14px;border:0;background:transparent;border-bottom:3px solid transparent;cursor:pointer}
nav button[aria-pressed=true]{font-weight:700;border-color:var(--teal);color:var(--teal)}
button:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #b6552e;outline-offset:3px}
.tools{display:flex;gap:14px;flex-wrap:wrap;align-items:end}.tools label{display:grid;gap:6px;font-size:.86rem;font-weight:650}
select{max-width:100%;background:white;border:1px solid #93aaa4;border-radius:4px;padding:8px 10px;min-height:40px}
.card{background:#fff;border:1px solid var(--line);border-radius:6px;padding:18px 20px;margin-top:15px}
.card h3:first-child{margin-top:0}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;text-align:left;font-variant-numeric:tabular-nums}
td,th{border-bottom:1px solid var(--line);padding:9px 10px 8px 0}th{font-weight:600;color:var(--muted);font-size:.85rem}
dl{display:grid;grid-template-columns:minmax(110px,1fr) 2fr;gap:8px 20px;margin:8px 0 0}dt{color:var(--muted)}dd{margin:0;overflow-wrap:anywhere}
details{margin-top:16px}summary{cursor:pointer;font-weight:600;color:var(--teal)}
pre{background:#f2f5f2;color:#233e43;white-space:pre-wrap;overflow-wrap:anywhere;padding:12px;border-radius:4px;font:12px/1.5 ui-monospace,monospace}
@media(max-width:560px){main{padding:20px 14px 40px}nav button{padding:9px 7px}.card{padding:13px}.tools>*{min-width:0}td,th{font-size:.81rem}figure{margin-left:-6px;margin-right:-6px}}
"""

_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Optical membrane campaign explorer</title><style>__CSS__</style></head><body><main>
<header><p class="eyebrow">Synthetic optical composites</p><h1>Campaign explorer</h1>
<p>Explore the evidence behind a formulation recommendation, inspect a return at intake, or check performance against a simple baseline.</p></header>
<figure><img alt="Mixture composition triangle and performance trade-off showing predictions and simulated returns" src="data:image/png;base64,__IMAGE__"></figure>
<nav aria-label="Explore campaign"><button data-view="plans" aria-pressed="true" aria-controls="plans">Batch decisions</button><button data-view="intake" aria-pressed="false" aria-controls="intake">Intake evidence</button><button data-view="checks" aria-pressed="false" aria-controls="checks">Model and gate checks</button></nav>
<section id="plans"><h2>Batch decisions</h2><div class="tools"><label>Planning round<select id="round"><option value="initial_plan">First batch</option><option value="next_plan">After feedback</option></select></label>
<label>Candidate<select id="candidate"></select></label></div><div id="candidate-detail" class="card"></div></section>
<section id="intake" hidden><h2>Intake evidence</h2><div class="tools"><label>Disposition<select id="action"></select></label>
<label>Submission<select id="run"></select></label></div><div id="run-detail" class="card"></div></section>
<section id="checks" hidden><h2>Model and gate checks</h2><div id="model-detail" class="card"></div></section>
<noscript>Enable JavaScript to inspect individual records. The figure above is available without JavaScript.</noscript>
<script id="data" type="application/json">__DATA__</script><script>__JS__</script></main></body></html>"""


_JS = """
'use strict';
const D = JSON.parse(document.getElementById('data').textContent);
const el = id => document.getElementById(id);
function node(tag, text) { const e=document.createElement(tag); e.textContent=text??''; return e; }
function options(id, entries) {
 const x=el(id); x.replaceChildren();
 for(const [value,label] of entries){ const o=node('option',label);o.value=value;x.add(o); }
}
function fields(parent, pairs) {
 const list=node('dl');for(const [k,v] of pairs){list.append(node('dt',k),node('dd',v??'Unavailable'));}parent.append(list);
}
function table(parent, headers, rows) {
 const t=node('table'), head=t.createTHead().insertRow();for(const h of headers)head.append(node('th', h));
 const body=t.createTBody();for(const row of rows){const tr=body.insertRow();for(const v of row)tr.append(node('td',v));}
 const wrap=node('div');wrap.className='scroll';wrap.append(t);parent.append(wrap);
}
function details(parent,title,value) {
 if(value==null) return;const w=node('details');w.append(node('summary',title),node('pre',JSON.stringify(value,null,2)));parent.append(w);
}
const nice=s=>String(s??'').replaceAll('_',' '), fmt=x=>Number.isFinite(x)?x.toFixed(3):'—';
function setupCandidates(){
 const p=D[el('round').value], selected=p.actions.map(x=>x.id), pool=[...new Set([...selected,...p.trace.map(x=>x.id)])];
 options('candidate',pool.map(id=>[id,id+(selected.includes(id)?' · selected':'')]));
 showCandidate();
}
function showCandidate(){
 const d=el('candidate-detail'), id=el('candidate').value, plan=D[el('round').value];d.replaceChildren();
 if(!id){d.append(node('p',nice(plan.stop_reason)));return;}
 const a=plan.actions.find(x=>x.id===id), c=D.candidates.find(x=>x.id===id), trace=plan.trace.filter(x=>x.id===id);
 d.append(node('h3',id+' · '+(a?nice(a.action):'Unselected')));
 if(a)d.append(node('p',a.rationale));
 if(c){const g=c.design;fields(d,[['Component fractions','A '+fmt(g.a)+'  /  B '+fmt(g.b)+'  /  C '+fmt(g.c)],
 ['Route and cure',g.route+'  /  '+fmt(g.cure)],['Cost',fmt(c.cost)+' units']]);}
 const scores=trace.filter(x=>x.score_terms);if(scores.length){
 d.append(node('h3','Scores by batch position'));
 if(el('round').value==='cold_start'){
 table(d,['Slot','Score basis','Reference designs','Score'],scores.map(x=>[x.slot,nice(x.score_terms.basis),x.score_terms.anchor_designs,fmt(x.score_terms.score)]));
 }else table(d,['Slot','Expected area utility','Makeability','Diversity','Utility per cost'],scores.map(x=>{
 const t=x.score_terms;return [x.slot,fmt(t.expected_area_utility),fmt(t.makeability_probability),fmt(t.diversity),fmt(t.score)];}));}
 for(const t of trace.filter(x=>x.reason))d.append(node('p',nice(t.stage)+' · '+nice(t.reason)+(t.details?.length?' · '+t.details.join(', '):'')));
 const ret=D.execution.find(x=>x.candidate_id===id);
 if(ret&&el('round').value==='initial_plan'){
 d.append(node('h3','Prediction recorded before simulated execution'));
 table(d,['Outcome','Predicted mean','Predictive SD','Measured'],['transmission','toughness'].map(k=>{
 const p=ret.pre_run_prediction.targets[k];return [k==='transmission'?'Transmission (fraction)':'Toughness (kJ/m²)',fmt(p.mean),fmt(p.sd),fmt(ret.observed?.[k])];}));}
 details(d,'Candidate inputs',c);details(d,'Score and eligibility trace',trace);
}
function setupRuns(){
 const filter=el('action').value, list=D.audit.map((r,index)=>[String(index),r]).filter(([,r])=>!filter||r.action===filter);
 options('run',list.map(([i,r])=>[i,(r.run_id||'Unidentified')+' · '+nice(r.action)+' · receipt '+r.submission_id]));showRun();
}
function showRun(){
 const d=el('run-detail');d.replaceChildren();if(el('run').value===''){d.append(node('p','No matching receipts.'));return;}
 const r=D.audit[Number(el('run').value)], accepted=D.initial_rows.find(x=>x.run_id===r.run_id), returned=D.execution.find(x=>x.run_id===r.run_id);
 d.append(node('h3',(r.run_id||'Unidentified')+' · '+nice(r.action)));
 fields(d,[['Disposition',nice(r.status)],['Decision reasons',(r.reasons??[]).map(nice).join(', ')],['Source',r.source_id],['Revision',r.revision],
 ['Current receipt',r.is_current?'Yes':'No'],['Review required',r.review_required?'Yes':'No']]);
 if(r.is_current&&!r.review_required&&r.action==='use'&&(accepted||returned)){
 const t=accepted?accepted.targets:returned.observed;
 d.append(node('h3','Eligible measurements'));
 fields(d,[['Preparation group',(accepted||returned).group_id],['Transmission',fmt(t.transmission)+' fraction'],['Toughness',fmt(t.toughness)+' kJ/m²']]);}
 const shadow=D.shadow_study?.decisions?.[r.run_id];
 if(shadow){d.append(node('h3','Shadow review'));fields(d,[['Suggested action',nice(shadow.action)],['Reason',nice(shadow.reason)],['Predicted cause',nice(shadow.predicted_cause)]]);}
 details(d,'Audit receipt',r);
}
function setupChecks(){
 const d=el('model-detail'), v=D.diagnostics?.validation;
 const example=D.prediction_example;
 if(example?.prediction?.descriptors){
 d.append(node('h3','Supplier card for '+example.candidate.id));
 table(d,['Property','Source','Value or estimate','Imputation SD'],Object.entries(example.prediction.descriptors)
 .map(([name,x])=>[name,nice(x.source),fmt(x.mean),fmt(x.sd)]));}
 const assay=D.descriptor_assay_decision;
 if(assay){const choice=assay.recommended_action;
 d.append(node('h3','Measure a supplier field or choose a recipe'),node('p',nice(choice.action)+' · '+(choice.candidate_id??'No eligible candidate')));
 table(d,['Candidate','Property to measure','Expected gain','Chance of switching','Net gain'],
 assay.candidates.flatMap(c=>(c.assays??[]).map(a=>[c.id,a.descriptor,fmt(a.expected_gain),fmt(a.switch_probability),fmt(a.expected_net_value)])));}
 if(v){d.append(node('h3','Whole-preparation validation'),node('p',v.n_groups+' preparation groups contributed equally to the mean absolute error.'));
 table(d,['Outcome','Model MAE','Training-mean MAE'],Object.entries(v.targets).map(([k,e])=>[k==='transmission'?'Transmission (fraction)':'Toughness (kJ/m²)',fmt(e.model_mae),fmt(e.mean_baseline_mae)]));}
 const gate=D.gate_replay;
 if(gate){d.append(node('h3','Fixed gate challenge'),node('p',gate.settled+' of '+gate.total+' constructed cases settled. '+gate.harmful_admissions+' harmful admissions and '+gate.harmful_exclusions+' harmful exclusions.'));
 table(d,['Scenario','Action','Comparison with known truth'],gate.cases.map(c=>[c.scenario,nice(c.action),nice(c.initial_routing)]));}
 const shadow=D.shadow_study;
 if(shadow?.evaluation){
 d.append(node('h3','Shadow classifier and fixed rules'),node('p','Each policy saw the same evidence. Coverage is the share of cases settled. Error is the share of incorrect actions among those settled.'));
 const rows=[];for(const [condition,policies] of Object.entries(shadow.evaluation)){
 for(const [name,m] of Object.entries(policies))rows.push([condition,name==='model'?'Shadow classifier':'Fixed rules',m.settled+' / '+m.groups,m.settled_errors+' / '+m.settled]);}
 table(d,['Evidence','Policy','Settled / total','Incorrect / settled'],rows);}
 details(d,'Prediction and missing-input uncertainty',D.prediction_example);
 details(d,'Shadow decision study',D.shadow_study);details(d,'Training snapshots',D.training_snapshot);
}
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{
 for(const b of document.querySelectorAll('[data-view]')){const on=b===button;b.setAttribute('aria-pressed',String(on));el(b.dataset.view).hidden=!on;}
}));
el('round').addEventListener('change',setupCandidates);el('candidate').addEventListener('change',showCandidate);
el('action').addEventListener('change',setupRuns);el('run').addEventListener('change',showRun);
options('action',[['','All dispositions'],...[...new Set(D.audit.map(r=>r.action))].sort().map(s=>[s,nice(s)])]);
if(D.cold_start){const cold=node('option','Cold start without measurements');cold.value='cold_start';el('round').add(cold);}
setupCandidates();setupRuns();setupChecks();
"""


def write_report(output: Path, summary: dict, packet: dict) -> None:
    """Write three presentation artifacts from the recorded packet; never rerun the decision."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "axes.unicode_minus": True}):
        image = _chart(summary, packet)
    (output / "decision_map.png").write_bytes(image)
    (output / "REPORT.md").write_text(_markdown(summary, packet), encoding="utf-8", newline="\n")
    keys = ("initial_rows", "audit", "candidates", "cold_start", "initial_plan", "next_plan", "prediction_example",
            "execution", "diagnostics", "gate_replay", "shadow_study", "training_snapshot", "descriptor_assay_decision")
    data = json.dumps({key: packet.get(key) for key in keys}, separators=(",", ":"), allow_nan=False)
    # Keep arbitrary run IDs and evidence strings inert inside the JSON script element.
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    page = _PAGE.replace("__CSS__", _CSS).replace("__JS__", _JS)
    page = page.replace("__IMAGE__", base64.b64encode(image).decode("ascii")).replace("__DATA__", data)
    (output / "explorer.html").write_text(page, encoding="utf-8", newline="\n")
