"""agent.py
Repte UPC Sostenible - Datathon 2026.

Agent de control optim per horitzo lliscant (MPC exacte) resolt amb programacio dinamica,
sense cap solver extern (nomes `math`).

Idea:
1. El score es lineal en l'energia importada. Cada kWh importat a l'hora t "costa"
       w_t = 0.70 * co2_t / REF_CO2 + 0.30 * preu_t / REF_COST
   on REF_CO2 i REF_COST normalitzen les dues metriques (nomes importa la seva ratio).
   Com que la demanda neta del campus (~1.100 kW) sempre supera els 500 kW de la bateria i la
   solar mai genera excedent, carregar P kW a l'hora t suma P*w_t i descarregar-ne resta P*w_t.
2. L'observacio dona els preus i el CO2 reals de les properes 24 h (incloent-hi l'hora actual).
   A cada hora resolem EXACTAMENT el problema d'optimitzacio de 24 h i apliquem nomes la
   primera decisio (MPC).
3. El problema es resol enrere amb el "valor marginal de l'energia emmagatzemada": la funcio de
   valor V_k(SOC) es convexa i lineal a trossos, i a cada pas nomes cal fusionar dues pendents
   (carregar i descarregar). Resultat identic a un LP (validat contra linprog i forca bruta),
   unes 17 vegades mes rapid.
4. Valor terminal 0: les 24 h conegudes ja contenen un cicle diari complet. Valorar l'energia
   sobrant al final de l'horitzo (abans, percentil 70) feia que la bateria acumules energia sense
   aprofitar-la. Amb valor 0 l'agent queda a <=0,002 punts de l'optim amb informacio perfecta.
5. Seguretat: clamp fisic estricte (0 infraccions), descarrega limitada a la demanda neta de
   l'ultima hora (evita abocar energia en forats de dades amb consum ~0), comprovacio que el
   vector de 24 h comenci a l'hora actual, i consigna 0 si alguna dada no es valida.
"""
import math


def _num(valor, defecte=None):
    """Converteix a float finit o retorna `defecte`."""
    try:
        x = float(valor)
    except (TypeError, ValueError):
        return defecte
    return x if math.isfinite(x) else defecte


class BatteryAgent:
    # Especificacions fisiques oficials (starter_kit/battery_spec.json)
    CAPACITAT_KWH = 2000.0
    POTENCIA_MAX_KW = 500.0
    RENDIMENT_CICLE = 0.90

    # Normalitzacio de la funcio objectiu (mitjanes de data/train; nomes importa la ratio)
    REF_CO2 = 0.0847    # kg CO2/kWh
    REF_COST = 0.1380   # EUR/kWh
    PES_CO2 = 0.70
    PES_COST = 0.30

    VALOR_TERMINAL = 0.0     # valor (en unitats de w) d'1 kWh emmagatzemat al final de l'horitzo
    LIMIT_DEMANDA = True     # no descarregar mes que la demanda neta mesurada l'ultima hora
    MARGE_SEGURETAT = 1e-4   # kW de marge respecte als limits fisics

    def __init__(self):
        self.eta = math.sqrt(self.RENDIMENT_CICLE)                 # ~0.9487 en carregar i en descarregar
        self.max_pujada = self.POTENCIA_MAX_KW * self.eta          # kWh emmagatzemats que pot guanyar en 1 h
        self.max_baixada = self.POTENCIA_MAX_KW / self.eta         # kWh emmagatzemats que pot perdre en 1 h

    # ------------------------------------------------------------------ dades d'entrada
    def _pes(self, preu, co2):
        return self.PES_CO2 * co2 / self.REF_CO2 + self.PES_COST * preu / self.REF_COST

    def _previsio(self, obs):
        """Pesos w_k per a l'hora actual (k=0) i les seguents, a partir dels vectors de 24 h."""
        preu_ara = _num(obs.get("preu_actual"))
        co2_ara = _num(obs.get("co2_actual"))
        # Sense `or []`: si arribessin arrays de numpy, avaluar-los com a booleà donaria error.
        preus = obs.get("preus_properes_24h")
        co2s = obs.get("co2_properes_24h")
        preus = [] if preus is None else list(preus)
        co2s = [] if co2s is None else list(co2s)

        pesos = []
        for p, c in zip(preus, co2s):
            p, c = _num(p), _num(c)
            if p is None or c is None:
                break  # tallem l'horitzo a la primera dada no valida
            pesos.append((p, c))

        if preu_ara is not None and co2_ara is not None:
            # El simulador inclou l'hora actual com a primer element del vector. Si no fos aixi,
            # l'afegim davant per no desalinear tot el pla una hora.
            if not pesos or abs(pesos[0][0] - preu_ara) > 1e-9 or abs(pesos[0][1] - co2_ara) > 1e-9:
                pesos.insert(0, (preu_ara, co2_ara))
        return [self._pes(p, c) for p, c in pesos]

    def _limit_descarrega(self, obs):
        """Maxim de kWh emmagatzemats que te sentit descarregar ara (limitat per la demanda neta)."""
        if not self.LIMIT_DEMANDA:
            return self.max_baixada
        dades = obs.get("noves_dades_5min")
        if not isinstance(dades, dict):
            dades = {}
        consum = _num(dades.get("ultim_consum_kw"))
        solar = _num(dades.get("ultima_solar_kw"), 0.0)
        if consum is None:
            return self.max_baixada
        demanda_neta = max(0.0, min(self.POTENCIA_MAX_KW, consum - solar))
        return demanda_neta / self.eta

    # ------------------------------------------------------------------ optimitzacio
    def _funcio_valor(self, pesos):
        """Funcio de cost futur V(SOC) despres de l'hora actual, convexa i lineal a trossos.

        Es representa com una llista de trams (pendent, longitud) ordenats per pendent creixent
        que cobreixen SOC de 0 a CAPACITAT_KWH. -pendent = valor marginal d'1 kWh emmagatzemat.
        Recursio: V_k = restriccio a [0, CAP] de la convolucio infimal de V_{k+1} amb el cost
        d'una hora (carregar costa w/eta per kWh guardat; descarregar estalvia w*eta per kWh).
        """
        trams = [(-self.VALOR_TERMINAL, self.CAPACITAT_KWH)]
        for w in reversed(pesos):
            nous = list(trams)
            for tram in ((-w / self.eta, self.max_pujada), (-w * self.eta, self.max_baixada)):
                i = 0
                while i < len(nous) and nous[i][0] < tram[0]:
                    i += 1
                nous.insert(i, tram)
            # El domini passa a [-max_pujada, CAP + max_baixada]: retallem tots dos extrems.
            retall, resultat = self.max_pujada, []
            for pendent, longitud in nous:
                if retall > 0.0:
                    if longitud <= retall:
                        retall -= longitud
                        continue
                    longitud -= retall
                    retall = 0.0
                resultat.append([pendent, longitud])
            retall = self.max_baixada
            while retall > 1e-12 and resultat:
                if resultat[-1][1] <= retall:
                    retall -= resultat[-1][1]
                    resultat.pop()
                else:
                    resultat[-1][1] -= retall
                    retall = 0.0
            trams = [tuple(t) for t in resultat]
        return trams

    def _decidir(self, obs, soc):
        pesos = self._previsio(obs)
        if not pesos:
            return 0.0
        w0 = pesos[0]
        trams = self._funcio_valor(pesos[1:])

        # Fins on surt a compte carregar (valor marginal > w0/eta) i fins on surt a compte
        # descarregar (valor marginal < w0*eta).
        soc_carrega = soc_descarrega = 0.0
        for pendent, longitud in trams:
            if pendent < -w0 / self.eta:
                soc_carrega += longitud
            if pendent < -w0 * self.eta:
                soc_descarrega += longitud

        if soc < soc_carrega:
            delta = min(soc_carrega, soc + self.max_pujada) - soc          # kWh guardats (>0)
            return delta / self.eta                                         # kW de carrega
        if soc > soc_descarrega:
            delta = soc - max(soc_descarrega, soc - self._limit_descarrega(obs))
            return -delta * self.eta                                        # kW de descarrega
        return 0.0

    # ------------------------------------------------------------------ interficie oficial
    def act(self, obs: dict) -> float:
        soc = _num(obs.get("soc_kwh"), 0.0)
        soc = min(self.CAPACITAT_KWH, max(0.0, soc))
        try:
            consigna = float(self._decidir(obs, soc))
            if not math.isfinite(consigna):
                consigna = 0.0
        except Exception:
            consigna = 0.0

        # Clamping fisic estricte per garantir 0 infraccions
        max_carrega = max(0.0, min(self.POTENCIA_MAX_KW, (self.CAPACITAT_KWH - soc) / self.eta - self.MARGE_SEGURETAT))
        max_descarrega = max(0.0, min(self.POTENCIA_MAX_KW, soc * self.eta - self.MARGE_SEGURETAT))
        return float(max(-max_descarrega, min(max_carrega, consigna)))
