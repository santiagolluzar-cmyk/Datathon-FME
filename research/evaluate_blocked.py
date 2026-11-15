"""Evaluate declared policies on the separate blocked seasonal development set."""
import argparse
import json
from pathlib import Path
from build_agent import build
from evaluate import run,load_data

HERE=Path(__file__).resolve().parent

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs',type=Path,default=HERE/'shortlist_configs.json')
    p.add_argument('--out',type=Path,default=HERE/'results/blocked_reproduced.json')
    a=p.parse_args()
    data=load_data(cache=HERE/'data_cache.pkl')
    split=json.loads((HERE/'folds/blocked/blocked_split.json').read_text())
    data=data.iloc[split['evaluation_indices']].copy()
    assert len(data)==1512
    result={'method':split['method'],'results':{}}
    for name,c in json.loads(a.configs.read_text()).items():
        directory=build(HERE/'folds/blocked',HERE/'agents/blocked'/name,c)
        row=run(data,directory,label=name,trace_path=a.out.parent/('blocked_'+name+'.csv'))
        result['results'][name]={'config':c,'result':row}
        a.out.write_text(json.dumps(result,indent=2))
        print(name,row['score'],row['errors'],flush=True)
