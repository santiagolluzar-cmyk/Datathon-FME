"""Avaluacio local del mode dificil (no consulta ni tramet dades a la web).

python eval_dificil.py --agent-dir mode_dificil --suite setmanes
Les puntuacions sobre train no substitueixen el test del servidor.
"""
import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

def carregar_dades(repo):
    sys.path.insert(0,str(repo))
    from eval_local import construir_dataset_horari
    df=construir_dataset_horari(repo/'data/train').set_index('Fecha')
    df['pv_kw']*=12
    cols={'Atersa - IRRADIANCIA (W/m2)':'irr',
          'Atersa - TEMPERATURA AMBIENTE (SENSOR 1) (ºC)':'tamb',
          'Atersa - TEMPERATURA PANEL (SENSOR 1) (ºC)':'tpan',
          'Atersa - VELOCIDAD VIENTO (SENSOR 1) (m/s)':'wind'}
    meteo=pd.read_csv(repo/'data/train/meteo.csv',usecols=['Fecha',*cols],parse_dates=['Fecha'])
    meteo=meteo.set_index('Fecha').rename(columns=cols).apply(pd.to_numeric,errors='coerce').resample('1h').mean()
    return df.join(meteo).reset_index()

def seleccionar(df,suite):
    if suite=='continus':return df
    selected=[]
    for _,month in df.groupby(df.Fecha.dt.to_period('M')):
        weeks=[]
        for period,week in month.groupby(month.Fecha.dt.to_period('W-SUN')):
            if len(week)==168 and week.Fecha.iloc[0]==period.start_time and week.Fecha.iloc[-1]==period.end_time.floor('h'):
                weeks.append(week)
        if weeks:selected.append(weeks[-1])
    if suite=='1008h':selected=selected[:6]
    return pd.concat(selected,ignore_index=True)

def simular(df,agent_dir,repo):
    from starter_kit.environment import CampusBatteryEnv
    spec=importlib.util.spec_from_file_location('agent_avaluat',Path(agent_dir)/'agent.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    class HardEnv(CampusBatteryEnv):
        def _get_obs(self):
            obs=super()._get_obs()
            if obs is None:return None
            # Finestra de mercat restringida a tres hores del bloc continu.
            limit=min(self.t+3,self.n)
            while limit>self.t+1 and self.df.Fecha.iloc[limit-1]-self.df.Fecha.iloc[self.t]!=pd.Timedelta(hours=limit-1-self.t):limit-=1
            obs['preus_properes_24h']=self.df.precio_eur_per_kwh.iloc[self.t:limit].tolist()
            obs['co2_properes_24h']=self.df.co2_kg_per_kwh.iloc[self.t:limit].tolist()
            obs.update(cap_max_kwh=self.cap,power_max_kw=self.power)
            last=self.df.iloc[max(0,self.t-1)]
            obs['noves_dades_5min'].update(irradiancia_wm2=float(last.irr),temp_ambient_c=float(last.tamb),
                                          temp_panel_c=float(last.tpan),vent_ms=float(last.wind))
            return obs
    env=HardEnv(df,spec=dict(capacity_kwh=6000,power_kw=1500,efficiency=.9,soc_init=.5,soc_min=0,soc_max=1,dt_hours=1))
    start=time.perf_counter(); agent=mod.BatteryAgent(); obs=env.reset();exceptions=[]
    # Call the internal plan too: the public act() safely falls back on exceptions.
    original_plan=agent._plan
    def checked_plan(*args):
        try:return original_plan(*args)
        except Exception as err:
            exceptions.append(str(err));raise
    agent._plan=checked_plan
    while obs is not None:
        action=float(agent.act(obs))
        if not math.isfinite(action):raise ValueError('Consigna no finita')
        obs,_=env.step(action)
    elapsed=time.perf_counter()-start
    result={**env.evaluate_results(),'hores':len(df),'temps_s':elapsed,'soc_final_kwh':float(env.soc),
            'errors_planificacio':len(exceptions),'models_carregats':list(agent.models),
            'mercat_analog_carregat':getattr(agent,'market',None) is not None}
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--agent-dir',type=Path,default=Path('mode_dificil'))
    p.add_argument('--suite',choices=['continus','setmanes','1008h'],default='setmanes')
    p.add_argument('--json',type=Path)
    a=p.parse_args();sys.path.insert(0,str(a.repo))
    data=seleccionar(carregar_dades(a.repo),a.suite)
    path=a.agent_dir if a.agent_dir.is_absolute() else a.repo/a.agent_dir
    result=simular(data,path,a.repo)
    print(json.dumps(result,indent=2))
    if a.json:a.json.write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
