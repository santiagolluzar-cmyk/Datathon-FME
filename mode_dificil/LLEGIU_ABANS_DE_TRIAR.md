# Anàlisi del tall públic/privat (15/11, 11:10)

> **Decisió final:** es van seleccionar la de 27,36 i una variant per al privat (13,33).
> Les dues són a [`../lliurament_final/`](../lliurament_final/). Aquest document recull l'anàlisi prèvia.

## Que hem descobert sobre el ranquing public del Mode Dificil
El public es la meitat de cada setmana de test. L'energia que la bateria porta guardada en comencar cada setmana
compta al public; la que queda al final, no. Amb el MATEIX agent, canviar nomes com acaba cada setmana mou la nota uns 3-5 punts:

| Versio | Com acaba cada setmana | Public (web) | Simulacio local: 1a meitat / 2a meitat / total |
|---|---|---|---|
| v5 | guarda ~50% per a l'endema | 23,44 | 21,9 / 19,4 / 20,6 |
| v6_final_bloc | buida | 20,15 | 17,3 / 23,2 / 20,1 |
| v7_public (= Versió 1 final) | plena | 27,36 (2n al public) | 25,0 / 14,5 / 19,7 |

El lider (27,98) es exactament el que surt de sumar aquest efecte a un agent normal (23,4 + ~4,6).

## Que vol dir per al ranquing PRIVAT (el que decideix el podi)
El privat es l'altra meitat de cada setmana. Qui guanya molt al public acabant amb la bateria plena, ho perd al privat.
- **v6_final_bloc** es la que te avantatge al privat (uns +2 a +3 punts sobre un agent normal, segons la simulacio).
- **v7_public** NO s'ha de triar mai: al privat quedaria uns 4-5 punts per sota d'un agent normal.

## Recomanacio de seleccio
1. `mode_dificil/v6_final_bloc/submission.zip` (la de 20,15 al public).
2. `lliuraments/submission_hybrid16_fulltrain.zip` (23,63 al public; cobertura per si el privat no es el complementari).

Si no es tria manualment, el sistema agafa les dues millors del public: cal marcar-les a la pestanya de lliurament.
