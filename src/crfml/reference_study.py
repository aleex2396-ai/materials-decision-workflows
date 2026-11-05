"""Controlled dual-control and estimator/adaptation reference experiments."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .control import ControlConfig,GainSet,solve_mpc,rollout
from .dual import (Belief,two_step_action,bayes_one_step,safe_cap,evaluate_two_step_action,
                   indistinguishable_disturbances,_one_step_array)
from .simulator import uniform_gain_episodes
from .discovery import discover
from .study import paired_summary,write_json
from .decision_report import finalize_artifacts


def run_reference_study(config_path,output):
    config_path,output = Path(config_path),Path(output)
    config = json.loads(config_path.read_text())
    if config['origin'] != 'public_synthetic_only':
        raise ValueError('Public simulation required')
    output.mkdir(parents=True,exist_ok=True)
    regimes = []
    for index,regime in enumerate(config['regimes']):
        cfg = ControlConfig(wbar=regime['wbar'])
        belief = Belief(regime['low'],regime['high'])
        x,previous = .2,0.
        now,later = regime['reference_now'],regime['reference_next']
        common = dict(cfg=cfg,reference_now=now,reference_next=later,
                      grid_size=config['grid_size'],quadrature_order=config['quadrature_order'])
        informed,informed_diag = two_step_action(x,belief,previous,**common)
        frozen,frozen_diag = two_step_action(x,belief,previous,frozen_belief=True,**common)
        refined = dict(common,grid_size=config['refinement_grid_size'],quadrature_order=config['refinement_quadrature_order'])
        ui,di = two_step_action(x,belief,previous,**refined)
        uf,df = two_step_action(x,belief,previous,frozen_belief=True,**refined)
        actions = {'informed':informed,'frozen':frozen,
                   'myopic':bayes_one_step(x,belief,previous,cfg,now)[0],
                   'maximal_probe':safe_cap(x,belief,cfg)}
        b,noise = uniform_gain_episodes(config['seed']+index,belief.low,belief.high,cfg.wbar,config['monte_carlo_episodes'])
        evaluations,samples = {},{}
        for policy,u in actions.items():
            evidence = evaluate_two_step_action(u,x,belief,previous,cfg,now,later,config['refinement_quadrature_order'])
            x1 = cfg.a*x+b*u+noise[:,0]
            if policy == 'frozen' or u == 0:
                low,high = belief.low,belief.high
            else:
                low = np.maximum(belief.low,(x1-cfg.a*x-cfg.wbar)/u)
                high = np.minimum(belief.high,(x1-cfg.a*x+cfg.wbar)/u)
            u1 = _one_step_array(x1,low,high,u,cfg,later)
            x2 = cfg.a*x1+b*u1+noise[:,1]
            costs = ((x1-now)**2+cfg.effort*u*u+cfg.slew*(u-previous)**2+
                     (x2-later)**2+cfg.effort*u1*u1+cfg.slew*(u1-u)**2)
            samples[policy] = costs
            evaluations[policy] = {'first_action':u,'quadrature_total_cost':evidence['frozen_cost' if policy=='frozen' else 'adapted_cost'],
                                   'sample_mean_cost':float(costs.mean()),'sample_mean_se':float(costs.std(ddof=1)/np.sqrt(len(costs))),
                                   'upper_constraint_violations':int(np.sum((x1>cfg.xmax+1e-9)|(x2>cfg.xmax+1e-9))),
                                   'expected_posterior_width':evidence['expected_posterior_width'],
                                   'same_first_action_information_benefit':evidence['information_benefit']}
        differences = samples['informed']-samples['frozen']
        regimes.append({'regime':regime,'policies':evaluations,
                        'paired_informed_minus_frozen_mean':float(differences.mean()),
                        'paired_mean_se':float(differences.std(ddof=1)/np.sqrt(len(differences))),
                        'informed_coarse_grid':informed_diag,'frozen_coarse_grid':frozen_diag,
                        'refinement':{'informed_action':ui,'frozen_action':uf,
                                      'informed_cost_change':di['minimum_value']-informed_diag['minimum_value'],
                                      'frozen_cost_change':df['minimum_value']-frozen_diag['minimum_value']}})
    cfg = ControlConfig()
    separation = 2*cfg.wbar/cfg.umax
    b_low,b_high = .36-separation/2,.36+separation/2
    sets = [GainSet(cfg.bmin,cfg.bmax),GainSet(cfg.bmin,cfg.bmax)]
    states = [.2,.2]
    last_u = 0.
    identical = []
    for t in range(30):
        u,_ = solve_mpc(states[0],.36,sets[0].high,cfg,last_u)
        noise = indistinguishable_disturbances(u,b_low,b_high,cfg.wbar)
        next_states = [cfg.a*x+b*u+w for x,b,w in zip(states,(b_low,b_high),noise)]
        for gain,x,nx in zip(sets,states,next_states):
            gain.update(x,u,nx,cfg)
        identical.append({'t':t,'input':u,'low_world_state':next_states[0],'high_world_state':next_states[1],
                          'interval_width':sets[0].high-sets[0].low,'matched_disturbances':list(noise)})
        states,last_u = next_states,u
    factorial = []
    for seed in range(12):
        for acquisition in ('random','greedy','constrained_ei'):
            material = discover(seed,acquisition,steps=12)
            base = np.random.default_rng(30000+seed).uniform(-1,1,36)
            for scenario in ('iid','bounded_shift'):
                w = cfg.wbar*base if scenario=='iid' else cfg.wbar*(base+1)/2
                results = {
                    'gp_adaptive':rollout(material['true_gain'],material['predicted_gain'],w,'robust_adaptive'),
                    'constant_adaptive':rollout(material['true_gain'],material['predicted_gain'],w,'adaptive_constant_center'),
                    'measured_adaptive':rollout(material['true_gain'],material['observed_gain'],w,'robust_adaptive')}
                factorial.append({'seed':seed,'acquisition':acquisition,'scenario':scenario,
                                  'true_gain':material['true_gain'],'gp_gain':material['predicted_gain'],
                                  'observed_gain':material['observed_gain'],
                                  'policies':{p:{k:v for k,v in r.items() if k!='trajectory'} for p,r in results.items()}})
    rng = np.random.default_rng(8866)
    comparisons = {}
    for acquisition in ('random','greedy','constrained_ei'):
        for scenario in ('iid','bounded_shift'):
            group = [r for r in factorial if r['acquisition']==acquisition and r['scenario']==scenario]
            for baseline in ('constant_adaptive','measured_adaptive'):
                comparisons[f'{acquisition}/{scenario}/{baseline}-gp'] = paired_summary([
                    r['policies'][baseline]['total_cost']-r['policies']['gp_adaptive']['total_cost'] for r in group],rng)
    summary = {'origin':'public_synthetic_only','config':config,'dual_regimes':regimes,
               'adversarial_identifiability':{'fixed_gain_separation':separation,
                    'max_state_difference':max(abs(r['low_world_state']-r['high_world_state']) for r in identical),
                    'minimum_compatible_interval_width':min(r['interval_width'] for r in identical),'trajectory':identical},
               'factorial_records':factorial,'factorial_paired':comparisons}
    write_json(output/'summary.json',summary)
    lines = ['# Controlled reference experiments','',
             'These supplementary references isolate information acquisition and the nominal estimator from online adaptation. They accompany the multi-state covariance-learning study; no laboratory measurements are used.','',
             '## Two-stage value of information','',
             'Both informed and frozen-belief policies observe the same next physical state. The frozen continuation keeps its old belief for action selection, but its performance is integrated under the actual conditional posterior. The first action is exhaustively searched on a finite grid; the expectation uses piecewise quadrature. Refinement is a numerical diagnostic, not a proof of continuous global optimality.','',
             '| Regime | Policy | First action | Quadrature cost | MC cost | MC mean SE | Violations |','|---|---|---:|---:|---:|---:|---:|']
    for r in regimes:
        for policy,v in r['policies'].items():
            lines.append(f"| {r['regime']['name']} | {policy} | {v['first_action']:.5f} | {v['quadrature_total_cost']:.6f} | {v['sample_mean_cost']:.6f} | {v['sample_mean_se']:.6f} | {v['upper_constraint_violations']} |")
    lines += ['', '## Adversarial identification limit','',
              f"Two fixed gains separated by {separation:.5f} generated transition histories differing by at most {summary['adversarial_identifiability']['max_state_difference']:.3g}, under matched admissible causal disturbances. The compatible interval never became narrower than {summary['adversarial_identifiability']['minimum_compatible_interval_width']:.5f}.", '',
              'This is an action-adapted bounded-disturbance construction, not an IID-noise experiment. Its induction argument prevents a uniform vanishing-error claim over that adversarial class. IID uniform disturbances provide different identification information.','',
              '## Matched estimator/adaptation ablation','',
              'Every compared policy uses identical adaptive gain-set updates, robust constraints, cost weights and disturbances. Only the nominal centre changes: GP, initial physical midpoint clipped to the current set, or the directly observed noisy gain. Positive cost difference favours GP; negative favours the comparator.','',
              '| Acquisition / scenario / comparator minus GP | Seeds | Cost difference | Bootstrap 95% interval |','|---|---:|---:|---|']
    for key,v in comparisons.items():
        lines.append(f"| {key} | {v['n']} | {v['mean']:.6f} | [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}] |")
    lines += ['', 'All reference parameters and seed conventions are fixed in code/configuration; raw grid values, refinement deltas and paired seed values are retained in summary.json.','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(output/'provenance.json',{'origin':'public_synthetic_only','no_private_data_loaded':True,
               'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
               'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}})
    finalize_artifacts(output)
    return summary
