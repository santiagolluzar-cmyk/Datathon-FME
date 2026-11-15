"""Agent de DIAGNOSTIC 2: revela l'estructura temporal de la simulacio (inici de cada bloc i nombre total d'hores).
Quan detecta el final (el vector de preus s'escurca) llenca un error amb el resum. No fa cap accio (retorna 0)."""
from datetime import datetime


class BatteryAgent:
    def __init__(self):
        self.n = 0
        self.prev = None
        self.blocs = []

    def act(self, obs):
        ts = datetime.fromisoformat(str(obs.get("timestamp"))[:19])
        if self.prev is None or (ts - self.prev).total_seconds() != 3600:
            self.blocs.append((self.n, ts.strftime("%m-%d %H")))
        self.prev = ts
        self.n += 1
        k = len(list(obs.get("preus_properes_24h") or []))
        if k < 3:
            raise ValueError("DIAG2 pas=%d ts=%s nprev=%d BLOCS(pas,inici)=%s" % (self.n, ts.strftime("%m-%d %H"), k, self.blocs))
        return 0.0
