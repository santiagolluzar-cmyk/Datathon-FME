# Auditoria independent dels agents originals

Data: 2026-11-15. Totes les execucions són locals; no hi ha accés al test ocult ni noves puntuacions oficials.

## Evidència principal

1. Els dos ZIPs corresponen funcionalment als fitxers de treball. Al baseline l'única diferència de `agent.py` és LF versus CRLF: les línies normalitzades són idèntiques. Tots els models i el prior coincideixen byte per byte. Al prototip tots els components del ZIP coincideixen byte per byte. Els fitxers `model_price.txt` i `model_co2.txt` existents al directori del prototip no s'utilitzen i no es distribueixen al seu ZIP.
2. S'han reproduït exactament les sis puntuacions històriques executant els agents complets OOF: 14,42/15,83 a calibratge, 17,03/18,17 a confirmació, 16,06/17,31 en les 1.680 hores agregades. Zero infraccions i zero errors de planificació detectats. Els models esperats es carreguen. Això confirma la reproducció, però no converteix el protocol en una prova independent.
3. La comparació OOF no era simplement un refit del baseline oficial. Es va canviar tant la capacitat com la recepta d'entrenament. El baseline oficial té 600 arbres de demanda i 500 de cada mercat. L'OOF té 169, 73 i 31; per demanda canvia 63→47 fulles, mínim de dades 40→100, regularització L2 1→5, learning rate 0,03→0,035, seed 0→83 i fraccions de mostreig 0,8→0,85. El prototip, en canvi, conserva els recomptes d'arbres seleccionats entre OOF i refit final. Vegeu els paràmetres incorporats als fitxers LightGBM i `hard_work/training.py:234`.
4. L'early stopping dels models OOF antics utilitza totes les setmanes del holdout (`training.py:238–240`), també les dites de «confirmació». Per tant, aquesta confirmació va influir en la selecció del nombre d'arbres del baseline. És contaminació de selecció, encara que no sigui entrenament directe dels arbres amb aquestes etiquetes.
5. Els quatre priors comparteixen exactament la mateixa taula residual `qtab`. El prototip l'aplica, multiplicada per 2, a un predictor de demanda diferent per als horitzons futurs (`mode_dificil/agent.py:489–490`). La coincidència és demostrada; que aquesta transferència causi la regressió oficial és una hipòtesi pendent de validació amb residuals independents.
6. Les 5.694 trajectòries de la biblioteca analògica final es troben a si mateixes entre els quatre veïns en consultar els mateixos orígens del train. El pes propi mitjà és 66,7%. Per tant, les mètriques sobre el train amb la biblioteca final incorporen els seus propis futurs com a referències. Això és vàlid per entrenar amb dades publicades, però invalida usar aquestes mètriques com a estimació de generalització.

## Puntuacions diagnòstiques del paquet final — DINS DE MOSTRA

| Dades publicades | ZIP baseline oficial | ZIP prototip oficial |
|---|---:|---:|
| Totes les 5.993 hores | 19,68 | 20,01 |
| Les 10 setmanes històriques, 1.680 hores | 17,58 | 17,82 |

Aquests dos paquets es van entrenar amb tot el train publicat. Cap número d'aquesta taula és fora de mostra. En les mateixes 10 setmanes, l'avantatge entre paquets finals és de 0,24 punts, versus 1,25 entre els agents OOF històrics. La reducció del marge prova que el canvi de model importa; no prova per què el servidor va invertir el rànquing.

El RMSE dins de mostra de demanda a h12–23 és 70,11 kW per al baseline i 206,39 kW per al prototip. No és una estimació fora de mostra i la capacitat més gran del baseline està confosa amb la generalització. A h0, la demanda mitjana del prototip tampoc és la distribució de decisió real: es substitueix pels quantils condicionals. Les mètriques completes consten al JSON.

## Optimitzador i seguretat

- 100 nous casos aleatoris de programació dinàmica contrastats amb programació lineal, horitzons d'1–24 hores, 1–7 escenaris i capacitats/potències diverses. La diferència màxima del cost òptim en fixar l'acció inicial proposada és 5,7e-14. No hi ha evidència que calgui substituir el MPC amb pesos no negatius.
- Hi ha un contraexemple amb pesos negatius, perquè la representació convexa deixa de ser vàlida. El prototip els trunca a zero (`agent.py:480`); el baseline no. En les 5.993 hores publicades tots els preus i intensitats reals són no negatius; tampoc s'han trobat pesos negatius als pronòstics complets dels dos paquets finals. Aquest risc no explica les execucions locals observades. El clipping és una salvaguarda convexa, no una solució exacta per a un mercat amb costos negatius.
- S'ha comprovat l'assumpció del comptador congelat: 1.489 observacions contigües entre les 00 i les 05 tenen càrrega idèntica a l'hora anterior i PV zero. Això sustenta el tractament nocturn en les dades publicades, sense assegurar que un servidor amb un altre preprocessament ho reprodueixi.
- El canvi de normalització dels pesos és una multiplicació comuna positiva i no canvia l'òptim, tret de possibles efectes numèrics insignificants. No és una explicació plausible de la regressió.
- Tots dos agents amaguen excepcions de càrrega de models i de predicció. El prototip també amaga errors de la biblioteca analògica. L'avaluador antic captura errors de `_plan`, però no veu errors interns consumits per `_predict` o pel `try` de mercat. La verificació final ha de comptar-los explícitament, a més de comprovar `models_carregats`.

## Hipòtesis comprovables sobre la regressió oficial

- El baseline OOF debilitat i la selecció amb «confirmació» han exagerat el marge del prototip respecte del baseline real.
- Quatre trajectòries analògiques poden tenir molta variància fora de distribució: només tres hores conegudes i la història recent no identifiquen necessàriament el dia següent. El descriptor usa cap de setmana però no incorpora explícitament la categoria de festiu nacional disponible al predictor antic (`mode_dificil/agent.py:385–390`).
- La base de persistència solar/càrrega manté massa informació actual als horitzons llargs si el petit model residual no aprèn prou correcció. Cal avaluar-ho fora de mostra, no amb els RMSE del train.
- Els quantils condicionals i la taula residual compartida poden estar mal calibrats en estacions o arrencades no vistes. La seva utilitat s'ha d'aïllar amb ablacions amb el mateix mercat, demanda, períodes i SOC.
- Canvis del preprocessament del servidor, salts i composició estacional del test continuen sent desconeguts. Cap dels resultats locals permet atribuir causalment els −0,34 punts oficials a una sola peça.

## Reproducció

Des de l'arrel del projecte:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python research/audit_agents_optimizer.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python research/audit_agents_in_sample.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python research/audit_agents_quantiles.py
```

Els scripts escriuen només `research/audit_agents_*`. No modifiquen els paquets històrics. Els fitxers JSON inclouen mètriques agregades, temps, SOC final, infraccions i càrrega de models; els resultats perfectes de l'LP només són proves de l'optimitzador, mai un agent desplegable.

## Diagnòstic dels quantils condicionals conservats

Interval nominal entre q0,0714 i q0,9286: cobertura teòrica 85,71%. S'exclouen les hores amb comptador congelat conegut.

| Conjunt | Hores incertes | Cobertura | Amplada mitjana |
|---|---:|---:|---:|
| Models finals, tot train (dins de mostra) | 4.504 | 88,17% | 217,49 kW |
| Models OOF històrics, 10 setmanes exposades | 1.270 | 78,98% | 219,75 kW |
| OOF històric, només llum diürna | 813 | 77,24% | 294,53 kW |

El 32,52% de les files OOF tenien algun creuament de quantils abans d'ordenar-los. L'ordenació utilitzada a l'agent garanteix escenaris ordenats, però no restitueix la calibració nominal. La cobertura més baixa fora de les files d'entrenament reforça la necessitat de calibratge independent. Aquest conjunt OOF també ha estat utilitzat històricament per seleccionar arbres, de manera que no és una comprovació final intacta. Vegeu `audit_agents_quantiles.json`.

## Ablacions històriques 2×2×2 — diagnòstic contaminat, no selecció final

Mismes 1.680 hores, SOC persistent, escala residual 2 i qtab original compartida. Cada combinació s’ha executat amb l’agent complet i monitoratge estricte d’excepcions, incloses les capturades internament. Totes tenen zero infraccions i zero errors.

| Mercat | Demanda | Incertesa actual | Score | CO₂ estalviat | Cost estalviat |
|---|---|---|---:|---:|---:|
| LGB | Directa | qtab | 16.05570 | 14.79012% | 19.00870% |
| LGB | Directa | Quantils | 16.43157 | 14.99466% | 19.78436% |
| LGB | Residual | qtab | 16.37663 | 15.12489% | 19.29735% |
| LGB | Residual | Quantils | 16.46669 | 15.04046% | 19.79457% |
| Analògic | Directa | qtab | 16.87034 | 15.79141% | 19.38785% |
| Analògic | Directa | Quantils | 17.27348 | 16.11058% | 19.98692% |
| Analògic | Residual | qtab | 17.21725 | 16.15250% | 19.70166% |
| Analògic | Residual | Quantils | 17.31354 | 16.14279% | 20.04529% |

Paritat amb el codi històric preservat: cantonada baseline idèntica en totes les accions, SOC i score; cantonada prototip score idèntic i diferència màxima d’acció 4,27e-14 kW.

L’efecte del mercat analògic és de +0,815 a +0,847 punts segons la combinació. El canvi de demanda aporta +0,321/+0,347 punts amb qtab i només +0,035/+0,040 amb quantils condicionals; la major part del guany de demanda es concentra en la interacció amb la decisió actual. Això descriu aquest conjunt històric, no demostra què passa al test ocult.

Resultats complets amb períodes, totals, temps, SOC final, hashes, models i comptadors d’error: `research/results/historical_ablations.json`. Script portable: `research/historical_ablations.py`. No s’han executat variants addicionals després d’aquest factorial.
