"""agent.py
Agent MPC (Model Predictive Control) lineal per al repte UPC Sostenible - Datathon 2026.
Optimitza el balanç ponderat 70% CO2 / 30% Cost amb 0 infraccions físiques.
"""
import numpy as np
from scipy.optimize import linprog


class BatteryAgent:
    def __init__(self):
        # Especificacions físiques oficials
        self.cap_max_kwh = 2000.0
        self.pot_max_kw = 500.0
        self.eta = 0.9 ** 0.5  # ~0.948683 (rendiment round-trip 90%)
        self.eps_seguretat = 1e-4

        # Paràmetres de normalització de la funció objectiu (ajustables amb dades de train)
        self.pes_co2 = 0.70
        self.pes_cost = 0.30
        self.ref_co2 = 0.0847    # kg CO2/kWh mitjà de referència
        self.ref_cost = 0.1380   # €/kWh mitjà de referència
        self.preu_excedent = 0.04

        # Perfil adaptatiu de demanda neta (Demanda - Solar) en kW: [2 (laborable/cap de setmana), 24 hores]
        # Valors inicials orientatius d'un campus universitari; s'actualitzen amb la telemetria real
        perfil_base = np.array([
            220.0, 200.0, 195.0, 195.0, 205.0, 240.0,  # 00-05h
            320.0, 480.0, 620.0, 650.0, 580.0, 420.0,  # 06-11h
            310.0, 280.0, 320.0, 430.0, 560.0, 640.0,  # 12-17h
            610.0, 520.0, 410.0, 320.0, 260.0, 230.0,  # 18-23h
        ])
        self.perfil_demanda_neta = np.vstack([perfil_base, perfil_base * 0.55])
        self.alpha_ema = 0.25  # Factor de suavitzat exponencial per actualitzar el perfil

        self.ultim_timestamp = None
        self.ultima_hora = None
        self.ultim_dow = None

    def _actualitzar_telemetria(self, obs: dict) -> float:
        """Actualitza l'estimació de demanda neta amb la darrera mesura de sensors."""
        dades = obs.get("noves_dades_5min") or {}
        consum = dades.get("ultim_consum_kw")
        solar = dades.get("ultima_solar_kw")

        hora_actual = int(obs["hour"])
        dow_actual = int(obs["day_of_week"])

        if consum is not None and solar is not None:
            demanda_neta_mesurada = float(consum) - float(solar)
            # La telemetria correspon a l'hora immediatament anterior
            hora_prev = (hora_actual - 1) % 24
            dow_prev = (dow_actual - 1) % 7 if hora_actual == 0 else dow_actual
            tipus_dia_prev = 1 if dow_prev >= 5 else 0

            actual = self.perfil_demanda_neta[tipus_dia_prev, hora_prev]
            self.perfil_demanda_neta[tipus_dia_prev, hora_prev] = (
                (1.0 - self.alpha_ema) * actual + self.alpha_ema * demanda_neta_mesurada
            )
            return demanda_neta_mesurada

        tipus_dia = 1 if dow_actual >= 5 else 0
        return float(self.perfil_demanda_neta[tipus_dia, hora_actual])

    def _preveure_demanda_neta(self, hora_inici: int, dow_inici: int, horitzo: int, d_recent: float) -> np.ndarray:
        """Genera el vector de demanda neta esperada per als propers `horitzo` passos."""
        previsio = np.zeros(horitzo, dtype=float)
        tipus_dia_0 = 1 if dow_inici >= 5 else 0
        # Ajust d'anomalia immediata respecte al perfil històric
        biaix_actual = d_recent - self.perfil_demanda_neta[tipus_dia_0, hora_inici]

        for k in range(horitzo):
            h = (hora_inici + k) % 24
            dies_extra = (hora_inici + k) // 24
            dow = (dow_inici + dies_extra) % 7
            tipus_dia = 1 if dow >= 5 else 0
            # L'anomalia recent decau exponencialment al llarg de les hores següents
            previsio[k] = self.perfil_demanda_neta[tipus_dia, h] + biaix_actual * (0.65 ** k)

        return previsio

    def _resoldre_mpc(
        self,
        soc_0: float,
        preus: np.ndarray,
        co2: np.ndarray,
        demanda_neta: np.ndarray,
    ) -> float:
        """Resol l'optimització lineal a H passos i retorna la consigna de l'hora actual (kW)."""
        H = len(preus)
        n_vars = 4 * H  # Per cada pas k: [P_ch, P_dis, P_imp, P_exp]

        # Pesos marginals normalitzats d'importació i exportació
        preus_imp = np.maximum(preus, self.preu_excedent + 1e-3)
        w_imp = (self.pes_co2 * (co2 / self.ref_co2)) + (self.pes_cost * (preus_imp / self.ref_cost))
        w_exp = -self.pes_cost * (self.preu_excedent / self.ref_cost)

        # Valor terminal de l'energia emmagatzemada al final de l'horitzó (evita buidat miop)
        valor_terminal = self.eta * float(np.percentile(w_imp, 70))

        # Vector de costos lineals c
        c = np.zeros(n_vars, dtype=float)
        eps_cicle = 1e-5  # Penalització mínima per evitar càrrega i descàrrega simultànies
        for k in range(H):
            idx = 4 * k
            c[idx + 0] = eps_cicle - valor_terminal * self.eta
            c[idx + 1] = eps_cicle + valor_terminal / self.eta
            c[idx + 2] = w_imp[k]
            c[idx + 3] = w_exp

        # Restriccions d'igualtat: -P_ch + P_dis + P_imp - P_exp = demanda_neta[k]
        A_eq = np.zeros((H, n_vars), dtype=float)
        b_eq = demanda_neta.copy()
        for k in range(H):
            idx = 4 * k
            A_eq[k, idx + 0] = -1.0
            A_eq[k, idx + 1] = 1.0
            A_eq[k, idx + 2] = 1.0
            A_eq[k, idx + 3] = -1.0

        # Restriccions de desigualtat: 0 <= SOC_0 + sum_{j<=k} (eta*P_ch - P_dis/eta) <= 2000
        marge_soc = 0.5
        espai_sup = max(0.0, self.cap_max_kwh - soc_0 - marge_soc)
        espai_inf = max(0.0, soc_0 - marge_soc)

        A_ub = np.zeros((2 * H, n_vars), dtype=float)
        b_ub = np.zeros(2 * H, dtype=float)
        inv_eta = 1.0 / self.eta

        for k in range(H):
            b_ub[k] = espai_sup
            b_ub[H + k] = espai_inf
            for j in range(k + 1):
                idx = 4 * j
                A_ub[k, idx + 0] = self.eta
                A_ub[k, idx + 1] = -inv_eta
                A_ub[H + k, idx + 0] = -self.eta
                A_ub[H + k, idx + 1] = inv_eta

        # Límits individuals de variables
        bounds = []
        for _ in range(H):
            bounds.extend([
                (0.0, self.pot_max_kw),  # P_ch
                (0.0, self.pot_max_kw),  # P_dis
                (0.0, None),             # P_imp
                (0.0, None),             # P_exp
            ])

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

        if res.success and res.x is not None:
            return float(res.x[0] - res.x[1])
        return 0.0

    def act(self, obs: dict) -> float:
        hora = int(obs["hour"])
        dow = int(obs["day_of_week"])
        soc_kwh = float(obs["soc_kwh"])
        preu_actual = float(obs["preu_actual"])
        co2_actual = float(obs["co2_actual"])

        d_recent = self._actualitzar_telemetria(obs)

        # Construcció del vector de 25 passos: hora actual exacta + 24h de previsió
        preus_24h = list(obs.get("preus_properes_24h") or [])
        co2_24h = list(obs.get("co2_properes_24h") or [])
        preus = np.array(preus_24h if preus_24h else [preu_actual], dtype=float)
        co2 = np.array(co2_24h if co2_24h else [co2_actual], dtype=float)
       
        demanda_neta = self._preveure_demanda_neta(hora, dow, len(preus), d_recent)
        consigna_desitjada = self._resoldre_mpc(soc_kwh, preus, co2, demanda_neta)

        # Clamping físic estricte per garantir 0 infraccions
        espai_disponible_kwh = max(0.0, self.cap_max_kwh - soc_kwh)
        energia_disponible_kwh = max(0.0, soc_kwh)
        max_carrega = max(0.0, min(self.pot_max_kw, espai_disponible_kwh / self.eta - self.eps_seguretat))
        max_descarrega = max(0.0, min(self.pot_max_kw, energia_disponible_kwh * self.eta - self.eps_seguretat))

        return float(np.clip(consigna_desitjada, -max_descarrega, max_carrega))
