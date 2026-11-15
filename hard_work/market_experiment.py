"""Analog market trajectories, trained outside temporal holdout weeks."""
import json
import importlib.util
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import control

OUT = Path(__file__).parent
REPO = Path(__file__).resolve().parents[1]


def market_features(f):
    x = f['Xm'][:,0]
    hour,month = x[:,5],x[:,4]
    # Current3 market levels and slopes, profile anomalies, rolling levels.
    return np.column_stack([x[:,8]/.04,x[:,9]/.04,x[:,10]/.04,
                            x[:,11]/.04,x[:,12]/.04,x[:,13]/.04,
                            (x[:,10]-x[:,8])/.015,(x[:,13]-x[:,11])/.015,
                            x[:,20]/.04,x[:,21]/.04,x[:,18]/.06,x[:,19]/.06,
                            np.sin(month*2*np.pi/12)*.7,np.cos(month*2*np.pi/12)*.7,
                            (x[:,2]>=5)*.5])


def analog(trainf, queryf, cache, k=12):
    n = len(trainf['load'])
    origins=np.arange(n)
    hold=cache['validation']
    usable=~hold.copy()
    for h in range(24):
        idx=np.minimum(origins+h,n-1)
        usable &= (origins+h<n)&~hold[idx]&(trainf['block_id']==trainf['block_id'][idx])
    tr=market_features(trainf)
    qu=market_features(queryf)
    # Missing cold-start lag features are ignored by filling their training medians.
    med=np.nanmedian(tr[usable],axis=0)
    tr=np.where(np.isfinite(tr),tr,med)
    qu=np.where(np.isfinite(qu),qu,med)
    prices=np.zeros((n,24));co2=np.zeros((n,24))
    for hour in range(24):
        candidates=np.flatnonzero(usable&(trainf['Xm'][:,0,5]==hour))
        queries=np.flatnonzero(queryf['Xm'][:,0,5]==hour)
        tree=cKDTree(tr[candidates])
        dist,idx=tree.query(qu[queries],k=min(k,len(candidates)))
        if dist.ndim==1:
            dist,idx=dist[:,None],idx[:,None]
        neighbor=candidates[idx]
        weight=1/(.2+dist**2)
        weight/=weight.sum(axis=1,keepdims=True)
        for key,dest,baseidx,anchoridx in [('price',prices,27,10),('carbon',co2,28,13)]:
            true=trainf[key]
            # Trajectory differences after known +2h; preserves observed current level.
            known=queryf['Xm'][queries,0,anchoridx-2:anchoridx+1]
            last=np.max(np.where(np.isfinite(known),np.arange(3),-1),axis=1)
            last=np.maximum(last,0)
            residual=true[neighbor[:,:,None]+np.arange(24)]-true[neighbor+last[:,None]][:,:,None]
            delta=np.sum(weight[:,:,None]*residual,axis=1)
            anchor=known[np.arange(len(queries)),last]
            dest[queries]=anchor[:,None]+delta
    known_p=queryf['Xm'][:,0,8:11]
    known_c=queryf['Xm'][:,0,11:14]
    prices[:,:3]=np.where(np.isfinite(known_p),known_p,prices[:,:3])
    co2[:,:3]=np.where(np.isfinite(known_c),known_c,co2[:,:3])
    return prices,np.maximum(co2,0.)


def main():
    trainf=dict(np.load(OUT/'temporal/features.npz'))
    queryf=dict(np.load(OUT/'temporal/features_cold.npz'))
    cache=dict(np.load(OUT/'temporal/cache_cold.npz'))
    spec=importlib.util.spec_from_file_location('hard_agent',REPO/'mode_dificil/agent.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[]
    calibration=cache['calibration']
    selected={k:v[calibration] for k,v in cache.items()}
    base=control.run(selected,module)
    rows.append({'policy':'baseline',**{k:float(base[k]) for k in ('score','cost_pct','co2_pct')}})
    print(rows[-1],flush=True)
    for k in (8,24):
        prices,co2=analog(trainf,queryf,cache,k=k)
        for blend in (.5,1.):
            candidate=dict(cache)
            candidate['prices']=(1-blend)*cache['prices']+blend*prices
            candidate['co2']=(1-blend)*cache['co2']+blend*co2
            selected={key:v[calibration] for key,v in candidate.items()}
            result=control.run(selected,module)
            row={'policy':f'analog_k{k}_blend{blend}',**{key:float(result[key]) for key in ('score','cost_pct','co2_pct')}}
            rows.append(row)
            print(row,flush=True)
            np.savez_compressed(OUT/f'market_analog_k{k}_b{blend}.npz',prices=candidate['prices'],co2=candidate['co2'])
    (OUT/'market_results.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
