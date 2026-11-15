# Agent del mode dificil

Paquet llest per a un lliurament manual: `submission.zip` (agent, perfils, biblioteca
de trajectories historiques i vuit models de demanda). El fitxer `agent.py` necessita
els fitxers auxiliars del mateix directori.

La previsio de mercat usa quatre trajectories de train semblants a les tres hores
observades. La demanda es prediu com una correccio de la persistencia de consum i
solar. Set quantils condicionals descriuen la incertesa de l'hora actual. La decisio
usa MPC convex exacte de 24 hores, amb clamp fisic de 6.000 kWh i 1.500 kW.

Des de l'arrel del repositori:

```bash
python eval_dificil.py --agent-dir mode_dificil --suite setmanes
python eval_dificil.py --agent-dir mode_dificil --suite 1008h
python starter_kit/check_submission.py mode_dificil/submission.zip
```

L'avaluador escala la solar per 12, restringeix el mercat a tres hores, inclou la
meteorologia i conserva el SOC entre setmanes separades. Per prudencia, no ofereix
previsions de mercat que travessin un salt temporal. El servidor podria tractar
aquestes fronteres de manera diferent. `eval_local.py` continua sent l'avaluador
del mode estandard.

`validacio_local.json` recull la comparacio temporal fora d'entrenament i el temps
d'execucio. Una puntuacio calculada sobre train amb els models finals ajustats a
train no es una estimacio fora de mostra ni confirma una posicio al ranking.
No s'ha fet cap lliurament web d'aquesta versio.
