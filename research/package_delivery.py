"""Package and verify both the holdout-tested and all-train refit agents locally."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd

from build_agent import build,DEFAULT
from evaluate import run,load_data

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'research'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pack(directory,path):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(Path(directory).iterdir()):
            if p.is_file() and p.suffix in ('.py','.txt','.json','.npz'):
                z.write(p,p.name)


def verify_zip(path,source,data,reference_trace,outdir,label):
    from starter_kit.check_submission import check_ast
    with tempfile.TemporaryDirectory(prefix='battery_zip_') as temp:
        directory=Path(temp)
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if Path(name).name!=name:
                    raise ValueError('Delivery must have flat root files')
            z.extractall(directory)
        manifest=json.loads((directory/'manifest.json').read_text())
        assert all(sha(directory/name)==value for name,value in manifest.items())
        assert all(sha(directory/name)==sha(source/name) for name in manifest)
        assert 'qtab_original' not in json.loads((directory/'prior.json').read_text())
        for p in directory.glob('*.py'):
            error=check_ast(p)
            if error: raise ValueError(error)
        trace=outdir/f'{label}_zip_trace.csv'
        expected=['direct','price','co2']+[f'q{j}' for j in range(7)]
        audited=run(data,directory,label=label,trace_path=trace,expected_models=expected)
        original=pd.read_csv(reference_trace)
        replay=pd.read_csv(trace)
        action_diff=float(np.max(abs(original.action_kw-replay.action_kw)))
        soc_diff=float(np.max(abs(original.soc_after_kwh-replay.soc_after_kwh)))
        assert action_diff<1e-9 and soc_diff<1e-9
        benchmark=run(data,directory,label=label,audit_exceptions=False,expected_models=expected)
        assert benchmark['errors']==0 and benchmark['violations']==0
        return dict(filename=path.name,sha256=sha(path),bytes=path.stat().st_size,
                    models_expected=expected,models_loaded=list(audited['model_status']),
                    model_count=len(audited['model_status']),model_status=audited['model_status'],
                    market_loaded=audited['market_loaded'],AST='passed',manifest='all hashes match',
                    no_original_qtab=True,
                    parity=dict(max_action_diff=action_diff,max_soc_diff=soc_diff),
                    errors=audited['errors'],violations=audited['violations'],hours=len(data),
                    runtime_1008_s=benchmark['runtime']['total_seconds'],
                    runtime_init_s=benchmark['runtime']['init_seconds'],
                    runtime_import_s=benchmark['runtime']['import_seconds'],
                    runtime_act_mean_ms=benchmark['runtime']['act_mean_ms'],
                    runtime_act_max_ms=benchmark['runtime']['act_max_ms'],
                    local_margin_to_60_seconds=60/benchmark['runtime']['total_seconds'],
                    audit=audited,benchmark=benchmark)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    selection=json.loads((HERE/'selection.json').read_text())
    frozen=json.loads((HERE/'results/locked_final.json').read_text())
    assert frozen['selection_sha256']==sha(HERE/'selection.json')
    config={**DEFAULT,**selection['config']}
    data=load_data(cache=HERE/'data_cache.pkl')
    split=json.loads((HERE/'split_manifest.json').read_text())
    final=data.loc[(data.Fecha>=split['locked_final']['start']) & (data.Fecha<split['locked_final']['end'])].copy()
    assert len(final)==1008
    locked=HERE/'agents/locked/selected'
    full=build(HERE/'folds/fulltrain',HERE/'agents/delivery_fulltrain',config)
    verified_zip=out/'agent_validat_prefinal.zip'
    full_zip=out/'submission_hybrid16_fulltrain.zip'
    pack(locked,verified_zip);pack(full,full_zip)
    # Replaying a frozen identical package verifies parity; it is not a new
    # architecture search on the locked data.
    verified=verify_zip(verified_zip,locked,final,HERE/'results/locked_selected_carry.csv',out,'prefinal')
    verified['score_diff']=abs(verified['audit']['score']-frozen['results']['selected']['carry']['score'])
    assert verified['score_diff']<1e-9
    verified['trained_until_exclusive']=split['locked_final']['start']
    verified['evidence']='The exact model artifacts evaluated on the declared1008-hour confirmation.'
    # Full-data refit gets a separately labelled, in-sample execution/parity
    # suite. Its score is not an out-of-sample estimate or used for selection.
    mask=np.zeros(len(data),bool)
    for f in split['development']:
        mask |= ((data.Fecha>=f['start']) & (data.Fecha<f['end'])).to_numpy()
    smoke=data.loc[mask].copy();assert len(smoke)==1008
    reftrace=out/'fulltrain_source_trace.csv'
    fullrun=run(smoke,full,label='fulltrain_in_sample_execution_only',trace_path=reftrace)
    refit=verify_zip(full_zip,full,smoke,reftrace,out,'fulltrain')
    refit['score_diff']=abs(refit['audit']['score']-fullrun['score'])
    assert refit['score_diff']<1e-9
    refit['training_hours']=len(data)
    refit['evidence']='All-train fixed-recipe refit. Execution suite is IN-SAMPLE; no independent generalization score for these exact weights.'
    for item,path in [(verified,verified_zip),(refit,full_zip)]:
        check=subprocess.run([sys.executable,str(ROOT/'starter_kit/check_submission.py'),str(path)],capture_output=True,text=True)
        (out/(path.stem+'_checker.log')).write_text(check.stdout+'\n'+check.stderr)
        item['check_submission']='passed' if check.returncode==0 else 'FAILED'
        if check.returncode!=0: raise RuntimeError(check.stdout+check.stderr)
    before=json.loads((HERE/'folds/locked/training_metadata.json').read_text())
    after=json.loads((HERE/'folds/fulltrain/training_metadata.json').read_text())
    comparison={}
    for key in ['direct','price','co2']+[f'q{j}' for j in range(7)]:
        b,c=before['models'][key],after['models'][key]
        comparison[key]=dict(before_rows=b['rows'],after_rows=c['rows'],before_trees=b['trees'],after_trees=c['trees'],recipe_identical=b['recipe']==c['recipe'])
        assert comparison[key]['recipe_identical'] and b['trees']==c['trees']
    priors=[json.loads((p/'prior.json').read_text()) for p in [locked,full]]
    shifts={}
    for key in ('load','pv','price','co2','irr'):
        a,b=priors[0][key]['mth'],priors[1][key]['mth'];keys=set(a)&set(b)
        delta=np.array([b[k]-a[k] for k in keys])
        shifts[key]=dict(common_cells=len(keys),rmse_change=float(np.sqrt(np.mean(delta**2))),max_abs_change=float(np.max(abs(delta))),added_cells=len(set(b)-set(a)))
    eligible=all(frozen['results']['selected'][g]['score']>frozen['results']['baseline_clean'][g]['score'] and frozen['results']['selected'][g]['errors']==0 and frozen['results']['selected'][g]['violations']==0 for g in ('carry','reset'))
    recommendation=('Recomano un únic intent oficial exploratori amb el refit complet, mantenint el baseline oficial de 23,45. El guany sobre el baseline reentrenat es repeteix en els dos tractaments del SOC, però no hi ha evidència suficient de top 1; el prototip encara supera lleugerament el candidat en aquesta reserva.' if eligible else 'No recomano consumir un intent oficial: el candidat no compleix la regla predeclarada de millora robusta sobre el baseline.')
    result=dict(selection_sha256=sha(HERE/'selection.json'),artifacts=[verified,refit],
                refit_audit=dict(training_hours_before=before['training_hours'],training_hours_after=after['training_hours'],
                    model_comparison=comparison,config_identical=json.loads((locked/'config.json').read_text())==json.loads((full/'config.json').read_text()),
                    code_identical=sha(locked/'agent.py')==sha(full/'agent.py'),profile_shifts=shifts,
                    analog_before=before['analog_trajectories'],analog_after=after['analog_trajectories'],
                    inner_residual_weeks_before=before['inner_calibration_spans'],inner_residual_weeks_after=after['inner_calibration_spans'],
                    caution='Refit changes learned parameters, profiles, analog pool and residual calibration weeks. Fixed recipes/code do not guarantee preserved hidden-test performance.'),
                recommendation=recommendation,
                external_actions='No server validation, submissions, commits, pushes or external project changes performed.',
                timing_caution='Local CPU timing only. Server hardware/runtime not measured.')
    assert result['refit_audit']['config_identical'] and result['refit_audit']['code_identical']
    (HERE/'results/delivery_verification.json').write_text(json.dumps(result,indent=2))
    (out/'delivery_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({i['filename']:{k:i[k] for k in ('sha256','bytes','errors','violations','runtime_1008_s','check_submission')} for i in result['artifacts']},indent=2))


if __name__=='__main__':main()
