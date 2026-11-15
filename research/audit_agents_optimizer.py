"""Independent optimizer checks; no training labels or models are used."""
from pathlib import Path
import importlib.util,json,math
import numpy as np
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('audited_agent',ROOT/'hard_work/original/agent.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def lp(ws,netss,soc,cap,pmax,first=None):
 H=len(ws);K=len(netss[0]);N=3*H+K*H
 obj=np.zeros(N)
 for t,w in enumerate(ws):obj[3*H+t*K:3*H+(t+1)*K]=w/K
 eq=[];eb=[]
 for t in range(H):
  row=np.zeros(N);row[t]=-a.ETA;row[H+t]=1/a.ETA;row[2*H+t]=1
  if t:row[2*H+t-1]=-1
  eq.append(row);eb.append(soc if t==0 else 0)
 if first is not None:
  row=np.zeros(N);row[0]=1;row[H]=-1;eq.append(row);eb.append(first)
 ub=[];bb=[]
 for t,nets in enumerate(netss):
  for k,n in enumerate(nets):
   row=np.zeros(N);row[t]=1;row[H+t]=-1;row[3*H+t*K+k]=-1
   ub.append(row);bb.append(-n)
 bounds=[(0,pmax)]*(2*H)+[(0,cap)]*H+[(0,None)]*(H*K)
 return linprog(obj,A_ub=ub,b_ub=bb,A_eq=eq,b_eq=eb,bounds=bounds,method='highs')

rng=np.random.default_rng(20261115)
rows=[]
for i in range(100):
 H=int(rng.integers(1,25));K=int(rng.integers(1,8));cap=float(rng.uniform(1000,9000));pmax=float(rng.uniform(100,2500));soc=float(rng.uniform(0,cap))
 ws=rng.uniform(0,.3,H);nets=np.sort(rng.normal(300,1000,(H,K)),axis=1).tolist()
 action=a.decide(a.backward(ws[1:],nets[1:],cap,pmax),ws[0],nets[0],soc,cap,pmax)
 best=lp(ws,nets,soc,cap,pmax);fixed=lp(ws,nets,soc,cap,pmax,action)
 if not(best.success and fixed.success):raise RuntimeError((i,best.message,fixed.message))
 rows.append({'case':i,'horizon':H,'scenarios':K,'gap':float(fixed.fun-best.fun),'action_kw':float(action)})

# For negative marginal costs convexity fails. A dense one-dimensional search
# for a 2-hour deterministic horizon demonstrates it without relying on an LP
# relaxation that is invalid for negative import prices.
negative=None
for i in range(1000):
 ws=rng.uniform(-1,1,2);nets=rng.uniform(-2,2,(2,1));cap=6.;pmax=1.5;soc=3.
 action=a.decide(a.backward(ws[1:],nets[1:],cap,pmax),ws[0],nets[0],soc,cap,pmax)
 lo=max(0,soc-pmax/a.ETA);hi=min(cap,soc+pmax*a.ETA)
 ss=np.linspace(lo,hi,10001)
 def total(s):
  d=s-soc;p=np.where(d>=0,d/a.ETA,d*a.ETA)
  p1=np.minimum(pmax,(cap-s)/a.ETA) if ws[1]<0 else -np.minimum(pmax,s*a.ETA)
  return ws[0]*np.maximum(nets[0,0]+p,0)+ws[1]*np.maximum(nets[1,0]+p1,0)
 chosen_soc=soc+(action*a.ETA if action>=0 else action/a.ETA)
 values=total(ss);gap=float(total(chosen_soc)-values.min())
 if gap>1e-3:
  negative={'cost_weights':ws.tolist(),'nets':nets.tolist(),'action':float(action),'grid_search_action':float((ss[values.argmin()]-soc)/(a.ETA if ss[values.argmin()]>=soc else 1/a.ETA)), 'objective_gap':gap,'grid_points':10001};break
result={'positive_weight_tests':len(rows),'max_first_action_lp_objective_gap':max(r['gap'] for r in rows),'min_gap':min(r['gap'] for r in rows),'negative_weight_counterexample':negative,'cases':rows}
path=ROOT/'research/audit_agents_optimizer.json';path.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
