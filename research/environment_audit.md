# Auditoria de l'entorn i de les dades

Auditoria local dels fitxers subministrats. No s'ha consultat el servidor ni el rànquing.
Reproducció: `python research/environment_audit.py`; resultats detallats a `research/environment_audit.json`.

## Comportament verificat

- `starter_kit/environment.py:94–147`: consigna positiva = càrrega; canvi de SOC `p*sqrt(.9)` carregant i `p/sqrt(.9)` descarregant; xarxa `max(load-pv+p,0)`. No hi ha degradació ni pèrdua durant els salts.
- `environment.py:174–188`: percentatges calculats sobre totals, pesos 70/30 i penalització de **0,20 punts** per infracció. La documentació escrita diu 1 punt: preval el codi i la regla facilitada a l'encàrrec.
- `eval_dificil.py:16–27,45–59`: multiplica la solar per 12 una sola vegada, usa 6.000 kWh i ±1.500 kW, afegeix meteorologia i restringeix mercat a tres hores. Redueix els vectors a dues/una hores abans d'un salt o final. El kit sense aquesta adaptació continuaria donant 24 hores.
- `environment.py:48–58,73–74`: primera observació d'una simulació exposa consum i solar de la mateixa hora. A les següents, exposa la fila anterior, fins i tot si el timestamp té un salt. `eval_dificil.py:55–57` aplica el mateix conveni a la meteorologia.
- **Aquesta particularitat no explica per si sola la regressió:** ambdós agents descarten telemetria a la primera observació i després d'un salt (`History.update`, línies 113–125). Només l'incorporen si l'observació anterior era exactament una hora abans. La memòria es reinicia, però el SOC de l'entorn es conserva.
- La prova física sintètica incorporada al script carrega 1.000 kW i descarrega 1.000 kW a través d'un salt de mes: SOC final 2.894,59074466 kWh, score 33,33, zero infraccions, confirmant convencions i persistència de SOC.

## Cobertura i qualitat de les dades

- Dataset consolidat: **5.993 hores**, de 2025-09-01 00:00 a 2026-08-16 22:00; 13 blocs continus. Cap nul a les nou columnes consolidades.
- **Novembre no té hores consolidades.** L'agost és parcial (379 hores). Els darrers set dies de cada mes falten per construcció; a més falten dues hores el 17 de juny i quatre el 4 d'agost.
- El primer/últim interval o hores pròximes a avaries poden tenir menys de dotze mostres. De les hores amb alguna mostra d'inversor vàlida: 5.987 tenen 12 mostres, 2 en tenen 11, 1 en té 7, 1 en té 4 i 2 en tenen 1. No hi ha ponderació especial per a hores incompletes.
- La columna de vent usada és zero a totes les hores. Temperatura ambient arriba a 54,27 °C. Tractar aquesta dada com a sensor real observat, no una previsió futura.
- Preu i CO₂ real són estrictament positius en les dades publicades: mínims 0,00327 €/kWh i 0,0278787 kg/kWh. Això no prova que tots els pronòstics o el test ocult ho siguin.
- Els excedents solars canvien fortament amb l'estació: 11 hores al desembre, 119 a l'abril i 93 en el fragment d'agost. Per tant, ometre agost/novembre o seleccionar només mesos freds pot canviar el rànquing de polítiques.

| Mes | Hores | Consum mitjà, kW | Solar ×12 mitjana, kW | Hores amb excedent |
|---|---:|---:|---:|---:|
| 2025-09 | 552 | 1.153,99 | 315,92 | 20 |
| 2025-10 | 576 | 1.139,75 | 258,52 | 26 |
| 2025-12 | 576 | 1.096,92 | 127,93 | 11 |
| 2026-01 | 576 | 1.081,59 | 137,53 | 13 |
| 2026-02 | 504 | 1.090,01 | 228,62 | 33 |
| 2026-03 | 576 | 1.081,72 | 289,67 | 48 |
| 2026-04 | 552 | 998,94 | 432,97 | 119 |
| 2026-05 | 576 | 1.030,17 | 414,24 | 90 |
| 2026-06 | 550 | 1.290,75 | 465,72 | 45 |
| 2026-07 | 576 | 1.449,25 | 455,34 | 43 |
| 2026-08 | 379 | 837,64 | 400,43 | 93 |

## Diferència amb el tutorial

El notebook omple cada canal d'inversor nul amb zero; `eval_local.construir_dataset_horari` exigeix tres canals vàlids abans de sumar. Això canvia:

| Hora | Solar original avaluador | Solar original tutorial | Efecte difícil |
|---|---:|---:|---|
| 2025-11-01 00:00 | exclosa | 0 kW | el tutorial admet una hora aïllada |
| 2026-06-17 14:00 | 124,183636 kW | 113,835 kW | −124,183636 kW després de ×12 |

És una diferència concreta i limitada, no una prova que expliqui −0,34 punts oficials. El significat físic del comptador pot suggerir un balanç diferent, però el repte i el notebook explícitament construeixen `load=-meter` i després resten PV; reproduir aquesta convenció, sense «corregir-la» silenciosament.

## Limitacions de la validació històrica

- Les deu setmanes són l'última setmana completa de dilluns a diumenge dins de cada mes disponible; no imiten exactament els darrers set dies de calendari reservats al servidor. No inclouen agost ni novembre.
- `eval_dificil --suite 1008h` pren simplement les primeres sis setmanes mensuals (setembre, octubre, desembre, gener, febrer, març). És una prova de temps, no una mostra estacional completa.
- `hard_work/training.py:218–240` fa early stopping dels tres models del baseline amb **totes** les setmanes reservades. La part anomenada «confirmació» no és completament independent de la selecció de la complexitat d'aquests models.
- Els perfils mensuals es refan excloent etiquetes reservades, però el `qtab` inicial es copia; l'encàrrec ja adverteix que es comparteix. No confondre `OOF` amb una independència absoluta de tots els components.
- `training.py:284` construeix historial sobre la cronologia sencera sense `split_mask`. Les files d'entrenament immediatament posteriors a una reserva poden consumir les seves dades com a retard (fins a 48 h en mercat). Per a aïllament estricte, reiniciar memòria als canvis de partició o embargar aquests orígens.
- `eval_dificil` instrumenta `_plan`; qualsevol excepció en `History.update` o en altres parts de `act` pot continuar amagada pel fallback. Cal comptar totes les vies d'excepció i validar els models esperats, no només una llista no buida.
- El score no remunera SOC final ni força retorn a 50%. Un terminal a zero en MPC pot ser racional a horitzó curt, però els resultats s'han de comparar amb SOC final i sensibilitat als reinicis.

## Disseny recomanat abans de mirar resultats nous

1. Reservar un conjunt final de diverses estacions amb dates fixades abans d'escollir variants. Entrenar perfils, models, escenaris residuals i biblioteca d'anàlegs sense cap etiqueta d'aquesta reserva.
2. Fer selecció en altres setmanes mensuals, incloent agost quan es pugui usar una finestra contínua arbitrària de set dies (no cal que comenci dilluns).
3. Purgar orígens/targets que travessen una frontera o un salt natural. Reiniciar historial als canvis train/calibratge/validació; incorporar telemetria només després de rebre una hora anterior vàlida.
4. Separar selecció del nombre d'arbres/calibratge de residuals i comprovació final. Una avaluació final única es pot informar encara que falli; no reutilitzar-la per escollir una nova variant.
5. Executar agents sencers amb SOC persistent i SOC reiniciat per bloc, més vector de mercat d'una/dos/tres hores. Reportar totals sense arrodonir, mètriques per estació, error intern, temps i SOC final. No fer mitjanes simples de scores de blocs.
6. Considerar un diagnòstic leave-month-out per estacions no representades; no presentar-lo com una estimació exacta del servidor, perquè el test ocult només exclou setmanes finals i conserva entrenament del mateix mes.

Les regles exactes del servidor en salts, preprocessament i composició de les 1.008 hores no queden provades pel paquet. Cal mantenir aquesta incertesa separada dels defectes demostrats localment.

## Particions fixades en aquesta iteració

La pipeline principal ha fixat tres comprovacions forward de desenvolupament, entrenant només abans de l'inici de cada període:

| Partició | Període d'avaluació | Hores |
|---|---|---:|
| Hivern | 2025-12-08 00:00 – 2025-12-21 23:00 | 336 |
| Primavera | 2026-03-09 00:00 – 2026-03-22 23:00 | 336 |
| Estiu | 2026-06-01 00:00 – 2026-06-14 23:00 | 336 |
| Reserva final | Les darreres 1.008 hores observades: 2026-06-21 19:00 – 2026-08-16 22:00, amb salts | 1.008 |

La reserva final conté 53 hores de juny, 576 de juliol i 379 d'agost. És una prova forward més estricta però només cobreix estiu; no és una confirmació final de totes les estacions. No s'han usat les seves puntuacions en aquesta auditoria ni al diagnòstic oracle.

## Referència amb informació perfecta, només diagnòstic

`python research/oracle.py` optimitza exactament amb LP de HiGHS sobre els tres períodes de desenvolupament. Usa **demanda, solar i mercat futurs reals**; és expressament no desplegable. Els denominadors de CO₂ i cost són comuns als tres períodes. Cada resultat s'ha reexecutat amb la dinàmica física de consignes signades, sense càrrega/descàrrega simultània, i amb zero infraccions.

| SOC entre períodes | Restricció terminal | Score oracle agregat |
|---|---|---:|
| Reinici a 3.000 kWh cada període | Lliure cada període | 17,920649 |
| Reinici a 3.000 kWh cada període | 3.000 kWh cada període | 17,266658 |
| SOC persistent | Lliure al final | 17,597491 |
| SOC persistent | 3.000 kWh al final | 17,413621 |

En la primera fila, scores individuals: hivern 10,664805; primavera 24,971894; estiu 22,824175. No s'han fet servir com a mitjana simple. Que el límit amb informació perfecta d'aquests períodes sigui inferior a 18 mentre el baseline té 23,45 oficials demostra per què no es poden extrapolar nivells absoluts ni sumar un guany local als punts oficials. No identifica per si mateix el motiu de la regressió entre agents.
