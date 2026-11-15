"""End-to-end heldout validation with real agent observations and history resets."""
import json
import shutil
import sys
from training import *
import eval_dificil


def main():
    baseline=OUT/'oof_baseline';baseline.mkdir(exist_ok=True)
    shutil.copy2(OUT/'original/agent.py',baseline/'agent.py')
    for name in ('model_net.txt','model_price.txt','model_co2.txt','prior.json'):
        shutil.copy2(OUT/'temporal'/name,baseline/name)
    sys.path.insert(0,str(REPO))
    data=eval_dificil.carregar_dades(REPO)
    mask,weeks=validation_mask(data);cal,con=calibration_masks(data,mask)
    results={'method':'End-to-end CampusBatteryEnv hard-mode observations; single SOC trajectory across time gaps; agent history resets on timestamp gaps; no retuning',
             'validation_weeks':weeks,'results':{}}
    for label,m in [('calibration',cal),('confirmation',con),('all',mask)]:
        selected=data.loc[m].reset_index(drop=True)
        results['results'][label]={}
        for name,path in [('baseline',baseline),('candidate',OUT/'oof_delivery')]:
            res=eval_dificil.simular(selected,path,REPO)
            results['results'][label][name]=res
            print(label,name,res['score'],res['errors_planificacio'],res['temps_s'],flush=True)
            (OUT/'authoritative_validation.json').write_text(json.dumps(results,indent=2))
        results['results'][label]['gain']=results['results'][label]['candidate']['score']-results['results'][label]['baseline']['score']
        (OUT/'authoritative_validation.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
