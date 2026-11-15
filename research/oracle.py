"""NONDEPLOYABLE perfect-information LP diagnostic on fixed development blocks.

Future demand, solar and market labels are used ONLY to give an optimistic
reference. Never import this module in the submitted agent. Final holdout is not
evaluated. Requires numpy, pandas and scipy (HiGHS LP).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from eval_dificil import carregar_dades

BLOCKS = [('winter','2025-12-08','2025-12-22'),
          ('spring','2026-03-09','2026-03-23'),
          ('summer','2026-06-01','2026-06-15')]
ETA = float(np.sqrt(.9))
CAP = 6000.
PMAX = 1500.
INITIAL = 3000.


def metrics(cost_base,co2_base,cost_agent,co2_agent):
    pc = 100*(cost_base-cost_agent)/cost_base
    pe = 100*(co2_base-co2_agent)/co2_base
    return {'cost_base_eur':float(cost_base),'co2_base_kg':float(co2_base),
            'cost_agent_eur':float(cost_agent),'co2_agent_kg':float(co2_agent),
            'cost_saved_pct':float(pc),'co2_saved_pct':float(pe),
            'score':float(max(0,.3*pc+.7*pe))}


def solve(data, cost_base_total, co2_base_total, terminal, initial=INITIAL):
    """LP variables: charge, discharge, end-of-hour SOC, grid imports.

    Nonnegative import marginal costs mean simultaneous charge/discharge never
    improves the optimum. Check returned solution and physical replay anyway.
    A tiny throughput tiebreaker chooses no gratuitous zero-cost cycling.
    """
    n=len(data)
    net=(data.load_kw-data.pv_kw).to_numpy()
    price=data.precio_eur_per_kwh.to_numpy()
    co2=data.co2_kg_per_kwh.to_numpy()
    marginal=.3*price/cost_base_total+.7*co2/co2_base_total
    assert np.min(marginal)>0
    # Main objective is fractional score loss, not rounded per-block scores.
    objective=np.r_[np.full(n,1e-13),np.full(n,1e-13),np.zeros(n),marginal]
    rows=[]; cols=[]; vals=[]
    for i in range(n):
        rows.extend([i,i,i]);cols.extend([i,n+i,2*n+i]);vals.extend([-ETA,1/ETA,1.])
        if i:
            rows.append(i);cols.append(2*n+i-1);vals.append(-1.)
    rhs=np.zeros(n);rhs[0]=initial
    neq=n
    if terminal is not None:
        rows.append(n);cols.append(3*n-1);vals.append(1.)
        rhs=np.r_[rhs,terminal];neq+=1
    Aeq=coo_matrix((vals,(rows,cols)),shape=(neq,4*n)).tocsr()
    idx=np.arange(n)
    Aub=coo_matrix((np.r_[np.ones(n),-np.ones(n),-np.ones(n)],
                       (np.tile(idx,3),np.r_[idx,n+idx,3*n+idx])),shape=(n,4*n)).tocsr()
    bounds=[(0,PMAX)]*(2*n)+[(0,CAP)]*n+[(0,None)]*n
    started=time.perf_counter()
    result=linprog(objective,A_ub=Aub,b_ub=-net,A_eq=Aeq,b_eq=rhs,bounds=bounds,
                   method='highs',options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
    elapsed=time.perf_counter()-started
    if not result.success:raise RuntimeError(result.message)
    charge,discharge,soc,grid=result.x.reshape(4,n)
    # Verify primal constraints, simultaneous flows and exact signed-action physics.
    action=charge-discharge
    physical_delta=np.where(action>=0,action*ETA,action/ETA)
    physical_soc=initial+np.cumsum(physical_delta)
    physical_grid=np.maximum(net+action,0)
    simultaneous=np.minimum(charge,discharge)
    phys_ok=(np.max(simultaneous)<1e-6 and physical_soc.min()>=-1e-5 and
             physical_soc.max()<=CAP+1e-5 and np.max(np.abs(physical_soc-soc))<1e-4)
    if terminal is not None:phys_ok=phys_ok and abs(physical_soc[-1]-terminal)<1e-4
    if not phys_ok:
        raise AssertionError(f'LP relaxation differs from physical signed-action replay: {simultaneous.max()}')
    base=np.maximum(net,0)
    detail=metrics(float(base@price),float(base@co2),float(physical_grid@price),float(physical_grid@co2))
    detail.update(hours=n,initial_soc_kwh=float(initial),final_soc_kwh=float(physical_soc[-1]),
                  solve_time_s=elapsed,max_simultaneous_kw=float(simultaneous.max()),
                  max_soc_replay_error_kwh=float(np.max(abs(physical_soc-soc))),
                  physical_action_replay_verified=True,infraccions=0,
                  information='perfect future labels; nondeployable')
    return detail,physical_soc,physical_grid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=Path(__file__).with_suffix('.json'))
    args=parser.parse_args()
    all_data=carregar_dades(args.repo)
    blocks=[]
    for label,start,end in BLOCKS:
        block=all_data.loc[(all_data.Fecha>=start)&(all_data.Fecha<end)].copy()
        assert len(block)==336 and block.Fecha.diff().dropna().eq(pd.Timedelta(hours=1)).all()
        blocks.append((label,block))
    selected=pd.concat([b for _,b in blocks],ignore_index=True)
    baseline=np.maximum(selected.load_kw-selected.pv_kw,0)
    bc=float(baseline@selected.precio_eur_per_kwh)
    be=float(baseline@selected.co2_kg_per_kwh)
    output={'purpose':'Nondeployable perfect-information upper-bound diagnosis, development periods only',
      'final_holdout_evaluated':False,
      'periods':[{'name':label,'start':str(b.Fecha.iloc[0]),'end':str(b.Fecha.iloc[-1]),'hours':len(b)} for label,b in blocks],
      'objective':'common total baseline denominators across all three periods; no mean of block scores',
      'physics':{'capacity_kwh':CAP,'power_kw':PMAX,'eta_charge':ETA,'eta_discharge':ETA,'initial_soc_kwh':INITIAL},
      'results':{}}
    for gap in ['reset','carry']:
        for terminal in [None,INITIAL]:
            key=f'{gap}_terminal_'+('free' if terminal is None else '3000')
            t=time.perf_counter()
            if gap=='reset':
                per_block=[]
                for label,block in blocks:
                    record,_,_=solve(block,bc,be,terminal)
                    per_block.append({'name':label,**record})
                ca=sum(b['cost_agent_eur'] for b in per_block)
                ce=sum(b['co2_agent_kg'] for b in per_block)
                total=metrics(bc,be,ca,ce)
                total.update(hours=len(selected),infraccions=0,blocks=per_block,
                    final_soc_kwh=[b['final_soc_kwh'] for b in per_block],
                    physical_action_replay_verified=True)
            else:
                total,soc,grid=solve(selected,bc,be,terminal)
                per_block=[];offset=0
                for label,block in blocks:
                    m=len(block);seg=slice(offset,offset+m)
                    bbase=np.maximum(block.load_kw-block.pv_kw,0)
                    record=metrics(float(bbase@block.precio_eur_per_kwh),float(bbase@block.co2_kg_per_kwh),
                                   float(grid[seg]@block.precio_eur_per_kwh),float(grid[seg]@block.co2_kg_per_kwh))
                    record.update(name=label,hours=m,initial_soc_kwh=float(INITIAL if offset==0 else soc[offset-1]),
                                  final_soc_kwh=float(soc[offset+m-1]))
                    per_block.append(record);offset+=m
                total['blocks']=per_block
            total['elapsed_s']=time.perf_counter()-t
            total['gap_soc']=gap;total['terminal_soc_constraint']=terminal
            output['results'][key]=total
            print(key,round(total['score'],6),'SOC',total['final_soc_kwh'],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
