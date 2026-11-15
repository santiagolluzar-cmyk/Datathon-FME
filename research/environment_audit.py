"""Reproduce the data/environment audit; no network or submissions.

Run from any directory: python /path/to/research/environment_audit.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_dificil import carregar_dades, seleccionar
from eval_local import seleccionar_blocs
from starter_kit.environment import CampusBatteryEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_suffix('.json'))
    args = parser.parse_args()
    data = carregar_dades(args.repo)
    data['base_grid'] = np.maximum(data.load_kw-data.pv_kw, 0)
    data['cost'] = data.base_grid * data.precio_eur_per_kwh
    data['carbon'] = data.base_grid * data.co2_kg_per_kwh
    data['surplus'] = np.maximum(data.pv_kw-data.load_kw, 0)
    data['surplus_h'] = (data.surplus > 0).astype(int)
    monthly = data.groupby(data.Fecha.dt.to_period('M')).agg(
        hours=('load_kw','size'), mean_load_kw=('load_kw','mean'),
        mean_scaled_pv_kw=('pv_kw','mean'), cost_base_eur=('cost','sum'),
        co2_base_kg=('carbon','sum'), surplus_kwh=('surplus','sum'),
        surplus_hours=('surplus_h','sum'))
    monthly.index = monthly.index.astype(str)
    blocks = seleccionar_blocs(data)
    raw_reports = {}
    for file in sorted((args.repo/'data/train').glob('*.csv')):
        raw = pd.read_csv(file)
        ts = pd.to_datetime(raw.Fecha)
        numeric = raw.drop(columns='Fecha')
        raw_reports[file.name] = {
            'rows':len(raw), 'columns':len(raw.columns),
            'start':str(ts.min()), 'end':str(ts.max()),
            'duplicate_timestamps':int(ts.duplicated().sum()),
            'null_cells':int(numeric.isna().sum().sum()),
            'all_null_columns':numeric.columns[numeric.isna().all()].tolist(),
            'time_deltas':{str(k):int(v) for k,v in ts.diff().value_counts().items()},
        }
    inv_cols = [f'Inversor {i} - POTENCIA ACTIVA (kW)' for i in (1,2,3)]
    inv = pd.read_csv(args.repo/'data/train/inversor.csv',parse_dates=['Fecha']).set_index('Fecha')[inv_cols]
    strict = inv.sum(axis=1,min_count=3).clip(lower=0).resample('1h').mean()
    tutorial = inv.fillna(0).sum(axis=1).clip(lower=0).resample('1h').mean()
    differences = (strict-tutorial).abs().gt(1e-8) | (strict.isna() & tutorial.notna())
    diff_table = pd.DataFrame({'strict_pv_kw':strict,'tutorial_pv_kw':tutorial})[differences]
    diff_table.index = diff_table.index.astype(str)
    sample_counts = inv.notna().all(axis=1).resample('1h').sum()
    # Exact deterministic physical/scoring smoke check with a timestamp gap.
    smoke_data = pd.DataFrame({
        'Fecha':pd.to_datetime(['2026-01-01 00:00','2026-02-01 00:00']),
        'load_kw':[1000.,1000.], 'pv_kw':[0.,0.],
        'precio_eur_per_kwh':[.1,.2], 'co2_kg_per_kwh':[.05,.1],
    })
    env = CampusBatteryEnv(smoke_data,spec=dict(capacity_kwh=6000,power_kw=1500,
        efficiency=.9,soc_init=.5,soc_min=0,soc_max=1,dt_hours=1))
    initial_obs = env.reset()
    second_obs,_ = env.step(1000)
    _,_ = env.step(-1000)
    eta = np.sqrt(.9)
    expected_soc = 3000+1000*eta-1000/eta
    assert np.isclose(env.soc, expected_soc)
    assert np.isclose(second_obs['soc_kwh'], 3000+1000*eta)
    assert env.evaluate_results()['score'] == 33.33
    assert env.evaluate_results()['infraccions'] == 0
    result = {
        'hours':len(data),'start':str(data.Fecha.min()),'end':str(data.Fecha.max()),
        'missing_joined_values':{k:int(v) for k,v in data.isna().sum().items()},
        'monthly':monthly.to_dict(orient='index'),
        'continuous_blocks':[{'start':str(b.Fecha.iloc[0]),'end':str(b.Fecha.iloc[-1]),'hours':len(b)} for b in blocks],
        'historical_week_validation_hours':len(seleccionar(data,'setmanes')),
        'legacy_1008_suite_start':str(seleccionar(data,'1008h').Fecha.min()),
        'legacy_1008_suite_end':str(seleccionar(data,'1008h').Fecha.max()),
        'raw_sources':raw_reports,
        'pv_samples_per_nonempty_hour':{str(k):int(v) for k,v in sample_counts[sample_counts>0].value_counts().sort_index().items()},
        'tutorial_pv_differences':diff_table.where(pd.notnull(diff_table),None).to_dict(orient='index'),
        'numeric_ranges':{c:{'min':float(data[c].min()),'max':float(data[c].max())} for c in ['load_kw','pv_kw','precio_eur_per_kwh','co2_kg_per_kwh','tamb','tpan','irr','wind']},
        'physical_smoke_check':{'initial_telemetry_current_row':initial_obs['noves_dades_5min']['ultim_consum_kw']==1000.,
          'soc_persists_across_gap':True,'final_soc_kwh':float(env.soc),'expected_soc_kwh':float(expected_soc),
          'metrics':env.evaluate_results()},
    }
    # Use strict JSON for absent PV values in the tutorial difference.
    for row in result['tutorial_pv_differences'].values():
        for key, val in row.items():
            if pd.isna(val):row[key]=None
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'hours':len(data),'blocks':len(blocks),'output':str(args.output),
                      'physical_smoke_score':env.evaluate_results()['score']}))


if __name__=='__main__':
    main()
