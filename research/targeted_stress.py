"""Targeted local diagnostics for guarded hybrid; never evaluates locked data.

A: blocked held-out periods, gap/SOC and telemetry sensitivity.
B: short-quote forward runs and exact action equivalence to 50%/k16 control.
C: deliberately biased forecasts in DIAGNOSTIC ONLY policy copies.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .evaluate import REPO, load_data, run
    from .stress import aggregate_runs
    from .build_agent import build
except ImportError:
    from evaluate import REPO, load_data, run
    from stress import aggregate_runs
    from build_agent import build


GUARDED = 'hybrid16_guarded'
BASELINE = 'lgb_direct_empirical'


def save(name, result):
    (REPO/'research/results'/name).write_text(json.dumps(result, indent=2))


def config():
    return json.loads((REPO/'research/guarded_config.json').read_text())[GUARDED]


def part_a(data):
    split = json.loads((REPO/'research/folds/blocked/blocked_split.json').read_text())
    subset = data.iloc[split['evaluation_indices']]
    assert subset.Fecha.max() < pd.Timestamp(split['locked_final_start'])
    build(REPO/'research/folds/blocked', REPO/'research/agents/blocked'/GUARDED, config())
    labels = [BASELINE, 'analog_residual_quantiles', 'blend_market_0.5', GUARDED]
    result = dict(method='Blocked held-out 1512h; explicit SOC gap semantics', split=split, results={})
    for label in labels:
        result['results'][label] = {}
        for mode in ['carry', 'reset']:
            r = run(subset, REPO/'research/agents/blocked'/label, gap_soc=mode, label=label)
            result['results'][label][mode] = r
            print('A', label, mode, r['score'], r['errors'], flush=True)
            save('targeted_gap_stress.json', result)
    for telemetry in ['missing_at_starts', 'current_at_starts']:
        r = run(subset, REPO/'research/agents/blocked'/GUARDED, telemetry_mode=telemetry, label=GUARDED)
        result['results'][GUARDED][telemetry] = r
        print('A', GUARDED, telemetry, r['score'], r['errors'], flush=True)
    result['gains'] = {mode: {name: result['results'][name][mode]['score']-result['results'][BASELINE][mode]['score']
                              for name in labels} for mode in ['carry', 'reset']}
    save('targeted_gap_stress.json', result)


def development_parts(data):
    split = json.loads((REPO/'research/split_manifest.json').read_text())
    for part in split['development']:
        subset = data.loc[(data.Fecha >= part['start']) & (data.Fecha < part['end'])]
        assert subset.Fecha.max() < pd.Timestamp(split['locked_final']['start'])
        yield part['name'], subset


def part_b(data):
    result = dict(method='Guarded short-quote policy vs exact 50%/k16 control; no final data',
                  note='Existing blend_market_0.5 and the independent control both have k16; exact equality is expected', folds={}, aggregate={})
    for fold, subset in development_parts(data):
        control = build(REPO/'research/folds'/fold, REPO/'research/stress_copies'/fold/'control_market50_k16',
                        {**config(), 'market_analog_weight': .5})
        result['folds'][fold] = {}
        for length in [1, 2]:
            result['folds'][fold][str(length)] = {}
            traces = []
            for label, directory in [(GUARDED, REPO/'research/agents'/fold/GUARDED), ('control_market50_k16', control)]:
                trace = REPO/'research/results'/f'short_guard_{fold}_{length}_{label}.csv'
                r = run(subset, directory, market_len=length, trace_path=trace, label=label)
                result['folds'][fold][str(length)][label] = r
                traces.append(pd.read_csv(trace)['action_kw'].to_numpy())
                print('B', fold, length, label, r['score'], flush=True)
            difference = float(np.max(np.abs(traces[0]-traces[1])))
            result['folds'][fold][str(length)]['max_action_difference_kw'] = difference
            assert difference == 0., f'Guard/control mismatch {fold}/{length}: {difference}'
            save('targeted_short_guard.json', result)
    historical = json.loads((REPO/'research/results/development_stress.json').read_text())
    for length, scenario in [(1,'one_known_market_hour'), (2,'two_known_market_hours')]:
        agg = aggregate_runs([f[str(length)][GUARDED] for f in result['folds'].values()])
        old = historical['aggregate'][scenario]
        agg['delta_vs_baseline'] = agg['score']-old[BASELINE]['score']
        agg['delta_vs_blend_market_0.5_k16'] = agg['score']-old['blend_market_0.5']['score']
        result['aggregate'][str(length)] = agg
    save('targeted_short_guard.json', result)


def diagnostic_copy(source, destination, bias, factor):
    shutil.copytree(source, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    instruction = dict(diagnostic_only=True, net_bias_kw=bias, unknown_market_weight_factor=factor,
                       altered_horizons='net: all24 including current; market weights: h>=3 only',
                       deployable=False)
    text = (source/'agent.py').read_text()
    text += '\n\n# DIAGNOSTIC ONLY: deliberately injected forecast error; do not submit.\n'
    text += '_unperturbed_forecast = BatteryAgent._forecast\n'
    text += 'def _diagnostic_forecast(self, obs):\n'
    text += '    weights, nets, detail = _unperturbed_forecast(self, obs)\n'
    text += '    weights = np.asarray(weights, dtype=float).copy()\n'
    text += f'    weights[3:] *= {factor!r}\n'
    text += f'    nets = [[float(value) + {bias!r} for value in row] for row in nets]\n'
    text += '    return weights, nets, detail\n'
    text += 'BatteryAgent._forecast = _diagnostic_forecast\n'
    (destination/'agent.py').write_text(text)
    (destination/'DIAGNOSTIC_ONLY.json').write_text(json.dumps(instruction, indent=2))
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.iterdir())
              if p.is_file() and p.name!='manifest.json'}
    (destination/'manifest.json').write_text(json.dumps(hashes, indent=2))
    return destination


def part_c(data):
    scenarios = {'net_plus100': (100., 1.), 'net_minus100': (-100., 1.),
                 'unknown_market_80pct': (0., .8), 'unknown_market_120pct': (0., 1.2)}
    result = dict(method='DIAGNOSTIC ONLY: synthetic errors after forecasting; no future labels; original copies unchanged',
                  scenarios={k:dict(net_bias_kw=v[0], unknown_market_weight_factor=v[1]) for k,v in scenarios.items()},
                  folds={}, aggregate={})
    for fold, subset in development_parts(data):
        result['folds'][fold] = {}
        for label in [BASELINE, GUARDED]:
            result['folds'][fold][label] = {}
            for name, (bias,factor) in scenarios.items():
                directory = diagnostic_copy(REPO/'research/agents'/fold/label,
                    REPO/'research/stress_copies'/fold/f'{label}__{name}', bias, factor)
                r = run(subset, directory, label=label+'__'+name)
                r['diagnostic_only'] = True
                r['forecast_stress'] = result['scenarios'][name]
                result['folds'][fold][label][name] = r
                print('C', fold, label, name, r['score'], r['errors'], flush=True)
                save('targeted_forecast_stress.json', result)
    for name in scenarios:
        result['aggregate'][name] = {}
        for label in [BASELINE, GUARDED]:
            result['aggregate'][name][label] = aggregate_runs([f[label][name] for f in result['folds'].values()])
        result['aggregate'][name]['gain_vs_baseline'] = (result['aggregate'][name][GUARDED]['score']
                                                       -result['aggregate'][name][BASELINE]['score'])
    save('targeted_forecast_stress.json', result)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=['A','B','C','all'], default='all')
    args=parser.parse_args()
    data=load_data(cache=REPO/'research/data_cache.pkl')
    for letter, function in [('A',part_a),('B',part_b),('C',part_c)]:
        if args.part in [letter,'all']:
            function(data)
