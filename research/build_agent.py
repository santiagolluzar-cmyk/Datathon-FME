"""Build self-contained, configurable hard-mode agents. No external actions."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT = dict(market_analog_weight=0.0, analog_k=4, net_residual_weight=0.0,
               current_quantiles=False, residual_scale=1.0, residual_source='clean',
               ratio_co2_cost=0.6345, horizon=24, current_scale=1.0,
               market_shrink=0.0, short_market_weight_cap=1.0)

CLASS = r'''
class BatteryAgent:
    CAP_DEFECTE = 6000.0
    PMAX_DEFECTE = 1500.0

    def __init__(self):
        self.config = json.loads((DIR / 'config.json').read_text(encoding='utf-8'))
        self.prior = json.loads((DIR / 'prior.json').read_text(encoding='utf-8'))
        self.prior['_cache'] = {}
        self.hist = History(self.prior)
        self.models = {}
        self.errors = 0
        self.last_error = None
        if lgb is None:
            raise RuntimeError('Required dependency lightgbm is unavailable')
        c = self.config
        required = []
        if c['net_residual_weight'] < 1:
            required.append('direct')
        if c['net_residual_weight'] > 0:
            required.append('residual')
        if min(c['market_analog_weight'],c['short_market_weight_cap']) < 1:
            required.extend(['price', 'co2'])
        if c['current_quantiles']:
            required.extend(['q'+str(j) for j in range(7)])
        for name in required:
            filename = 'net_'+name+'.txt' if name.startswith('q') else 'model_'+name+'.txt'
            self.models[name] = lgb.Booster(model_str=(DIR / filename).read_text(encoding='utf-8'))
        self.market = MarketAnalog(DIR / 'analog_library.npz', k=c['analog_k']) if c['market_analog_weight'] > 0 else None
        if c['residual_source'] == 'original':
            self.qd = self.qr = self.prior['qtab_original']
        else:
            self.qd, self.qr = self.prior['qtab_direct'], self.prior['qtab_residual']

    def _predict(self, name, X):
        result = np.asarray(self.models[name].predict(X, num_threads=1), dtype=float)
        if not np.all(np.isfinite(result)):
            raise ValueError('Nonfinite model prediction: '+name)
        return result

    def _forecast(self, obs):
        # Accept at most the three market hours specified by hard-mode rules.
        obs = dict(obs)
        for key in ('preus_properes_24h','co2_properes_24h'):
            if obs.get(key) is not None:
                obs[key] = list(obs[key])[:3]
        self.hist.update(obs)
        nr, mr, night, last = features(self.hist)
        xn, xm = np.asarray(nr, dtype=float), np.asarray(mr, dtype=float)
        c = self.config
        mw, nw = c['market_analog_weight'], c['net_residual_weight']
        if len(self.hist.known) < 3:
            mw = min(mw,c['short_market_weight_cap'])
        prices, co2 = np.zeros(H), np.zeros(H)
        if mw < 1:
            prices += (1-mw)*(xm[:,27] + self._predict('price', xm))
            co2 += (1-mw)*(xm[:,28] + self._predict('co2', xm))
        if mw > 0:
            pa, ca = self.market.predict(xm)
            prices += mw*pa
            co2 += mw*ca
        shrink = c.get('market_shrink',0.0)
        if shrink:
            prices = (1-shrink)*prices + shrink*xm[:,27]
            co2 = (1-shrink)*co2 + shrink*xm[:,28]
        for j,(p,q) in enumerate(self.hist.known[:3]):
            prices[j],co2[j] = p,q
        weights = np.maximum(.7*co2/c['ratio_co2_cost'] + .3*prices, 0.0)
        base = np.where(np.isfinite(xn[:,25]),xn[:,25],xn[:,6]) - np.where(np.isfinite(xn[:,23]),xn[:,23],xn[:,7])
        pred = np.zeros(H)
        if nw < 1:
            pred += (1-nw)*self._predict('direct',xn)
        if nw > 0:
            pred += nw*(base+self._predict('residual',xn))
        nets = []
        for h in range(H):
            if night[h] and math.isfinite(last):
                nets.append([last])
            else:
                key = str(h)+'_'+str(int(xn[h,8]>50))
                q = (1-nw)*np.asarray(self.qd[key]) + nw*np.asarray(self.qr[key])
                scale = c['residual_scale'] * (c['current_scale'] if h==0 else 1)
                nets.append(sorted((pred[h]+scale*q).tolist()))
        if c['current_quantiles'] and not (night[0] and math.isfinite(last)):
            q = np.array([base[0]+self._predict('q'+str(j),xn[:1])[0] for j in range(7)])
            center = float(np.median(q))
            nets[0] = sorted((center+c['current_scale']*(q-center)).tolist())
        return weights,nets,dict(net=pred,prices=prices,co2=co2)

    def _plan(self, obs, soc, cap, pmax):
        weights,nets,_ = self._forecast(obs)
        horizon = int(self.config['horizon'])
        value = backward(weights[1:horizon],nets[1:horizon],cap,pmax)
        return decide(value,float(weights[0]),nets[0],soc,cap,pmax)

    def act(self, obs):
        cap = _f(obs.get('cap_max_kwh'))
        cap = self.CAP_DEFECTE if not math.isfinite(cap) or cap<=0 else cap
        pmax = _f(obs.get('power_max_kw'))
        pmax = self.PMAX_DEFECTE if not math.isfinite(pmax) or pmax<=0 else pmax
        soc = _f(obs.get('soc_kwh'))
        soc = 0.0 if not math.isfinite(soc) else min(cap,max(0.0,soc))
        try:
            p = float(self._plan(obs,soc,cap,pmax))
            if not math.isfinite(p):
                raise ValueError('Nonfinite action')
        except Exception as error:
            self.errors += 1
            self.last_error = str(error)
            p = 0.0
        max_c = max(0.0,min(pmax,(cap-soc)/ETA-1e-4))
        max_d = max(0.0,min(pmax,soc*ETA-1e-4))
        return float(max(-max_d,min(max_c,p)))
'''


def build(bundle, destination, config=None):
    bundle, destination = Path(bundle), Path(destination)
    c = {**DEFAULT, **(config or {})}
    destination.mkdir(parents=True, exist_ok=True)
    # Stable audited functions, including the verified piecewise-linear optimizer.
    core = (ROOT/'mode_dificil/agent.py').read_text().split('class BatteryAgent:')[0]
    core = core.replace('"""Control del mode dificil: mercat per trajectories historiques semblants,\ndemanda residual i quantils condicionals, amb MPC convex exacte de 24 hores.\nNomes utilitza les dades publicades de train i les observacions rebudes.\n"""',
                        '"""Audited causal hard-mode forecasts and exact convex battery MPC."""')
    (destination/'agent.py').write_text(core+CLASS, encoding='utf-8')
    (destination/'config.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
    prior = json.loads((bundle/'prior.json').read_text())
    if c['residual_source'] == 'clean':
        prior.pop('qtab_original',None)
    (destination/'prior.json').write_text(json.dumps(prior),encoding='utf-8')
    files = []
    if c['net_residual_weight'] < 1: files.append('model_direct.txt')
    if c['net_residual_weight'] > 0: files.append('model_residual.txt')
    if min(c['market_analog_weight'],c['short_market_weight_cap']) < 1: files.extend(['model_price.txt','model_co2.txt'])
    if c['market_analog_weight'] > 0: files.append('analog_library.npz')
    if c['current_quantiles']: files.extend('net_q'+str(j)+'.txt' for j in range(7))
    for name in files:
        shutil.copy2(bundle/name,destination/name)
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.iterdir()) if p.is_file() and p.name!='manifest.json'}
    (destination/'manifest.json').write_text(json.dumps(hashes,indent=2))
    return destination


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path)
    a=p.parse_args()
    build(a.bundle,a.out,json.loads(a.config.read_text()) if a.config else None)
