"""Agent de DIAGNOSTIC: nomes per a la 'Validacio Rapida Mode Dificil' (no gasta quota). NO trametre al ranquing.
Al pas 30 llenca un error que mostra exactament que rep l'agent del servidor."""


class BatteryAgent:
    def __init__(self):
        self.v = []

    def act(self, obs):
        nd = obs.get("noves_dades_5min") or {}
        row = [str(obs.get("timestamp"))[5:16], len(list(obs.get("preus_properes_24h") or [])),
               round(float(nd.get("ultim_consum_kw", -1)), 1), round(float(nd.get("ultima_solar_kw", -1)), 1),
               round(float(nd.get("irradiancia_wm2", -1)), 1), round(float(nd.get("temp_ambient_c", -1)), 1),
               round(float(nd.get("temp_panel_c", -1)), 1), round(float(nd.get("vent_ms", -1)), 2),
               round(float(obs.get("preu_actual", -1)), 4), round(float(obs.get("co2_actual", -1)), 4)]
        self.v.append(row)
        if len(self.v) == 1:
            self.first = {"claus": sorted(obs.keys()), "noves_claus": sorted(nd.keys()), "tipus_ts": type(obs.get("timestamp")).__name__,
                          "preus": list(obs.get("preus_properes_24h") or []), "soc": obs.get("soc_kwh"),
                          "cap": obs.get("cap_max_kwh"), "pmax": obs.get("power_max_kw")}
        if len(self.v) >= 30:
            raise ValueError("DIAG " + repr(self.first) + " FILES[ts,nprev,consum,solar,irr,tamb,tpan,vent,preu,co2]=" + repr(self.v))
        return 0.0
