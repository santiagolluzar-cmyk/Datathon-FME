# Mode Dificil - agent v5 (Riviera Maya Boys)

`submission.zip` es el paquet llest per lliurar (agent.py + prior.json + 24 models LightGBM en text + analog.npz).

## Com funciona
1. **Mercat (+3..+24 h)**: models LightGBM especialitzats per horitzo (un per a +3, +4, +5 h; un per a +6..+8; un general per a +9..+23)
   sobre una base heuristica (0,5*ahir + 0,5*perfil), barrejats al 60% amb una previsio per **analegs** (les 16 trajectories de
   train mes semblants a les 3 hores conegudes, enganxades a la darrera hora coneguda).
2. **Demanda neta (consum - solar x12)**: models especialitzats per a l'hora actual, +1, +2 i +3..+5 h (aprofiten l'ultima hora
   mesurada, la tendencia dels nuvols i les rampes de consum) i un model general per a la resta. De 0 a 5 h el comptador queda
   congelat i la demanda es coneguda.
3. **Incertesa**: escenaris de l'error de previsio calculats fora de mostra (plecs per setmanes), per horitzo i hora del dia; per a
   l'hora actual, 9 quantils condicionals a la situacio.
4. **Decisio**: optimitzador exacte de 24 h (programacio dinamica sobre funcions convexes lineals a trossos) amb aquests escenaris.
5. **Seguretat**: clamp fisic (0 infraccions), tolerancia a hores perdudes, final de simulacio (2.016 h), degradacio si falta un model.

## Resultats (simulador oficial, 3 sequencies d'una setmana per mes fora d'entrenament; optim amb informacio perfecta = 21,9)
| Versio | Local | Web (public) |
|---|---|---|
| v1 (hard_work/original) | 19,78 | 23,45 |
| v3 (models per horitzo) | 20,33 | - |
| v4 (v3 + analegs) | 20,56 | 23,38 |
| v5 (v4 + quantils condicionals) | 20,59 | pendent |
| hibrid del company (lliuraments/) | - | 23,63 |

El ranquing public (1.008 h, la meitat de cada setmana) no ha seguit les millores locals: vegeu les notes de l'equip.
`recerca/` conte els scripts usats (rutes de l'ordinador de treball; serveixen de referencia).
