# Auditoria de l'entrenament i de la validació històrica

L'auditoria s'ha executat sobre els artefactes adjunts. Els detalls numèrics i els hashes són a `training_audit.json`; el càlcul es reprodueix amb `python research/training_audit.py` des de l'arrel del projecte. No s'ha accedit al test ocult ni a cap servidor.

## Evidència confirmada

La comparació OOF antiga **no representa un simple reentrenament dels dos paquets oficials amb les mateixes receptes**. La recepta del baseline també es va canviar:

| Component | Baseline oficial 23,45 | Baseline OOF històric |
|---|---:|---:|
| Arbres de demanda | 600 | 169 |
| Fulles de demanda | 63 | 47 |
| Mínim d'exemples per fulla de demanda | 40 | 100 |
| Regularització L2 de demanda | 1 | 5 |
| Arbres de preu | 500 | 73 |
| Arbres de CO₂ | 500 | 31 |
| Learning rate | 0,03 | 0,035 |
| Fraccions de mostres i variables | 0,80 | 0,85 |
| Llavor | 0 | 83 |

Els models residuals del prototip, en canvi, mantenen els recomptes seleccionats entre OOF i paquet final: 79 arbres per a la demanda i 142/122/81/106/374/155/76 per als set quantils. Els paràmetres anteriors s'han llegit dels fitxers LightGBM, no inferit de les mides dels fitxers. Canviar la capacitat i la regularització pot alterar la comparació; això **no demostra** que més arbres siguin millors fora de mostra.

La suposada confirmació tampoc era completament intocable. `hard_work/training.py:219` defineix la validació com totes les setmanes reservades i `:238` les usa per a early stopping del baseline, incloent-hi les cinc setmanes que després s'anomenen confirmació. El prototip residual i els quantils utilitzen les setmanes de calibratge per seleccionar iteracions; les mètriques de confirmació de nombroses variants s'escriuen durant l'exploració (`hard_work/net_candidates.py:50`). Això acredita exposició dels resultats, no permet determinar quines decisions humanes es van prendre mirant-los.

Tots els `prior.json` inspeccionats comparteixen **exactament la mateixa taula residual `qtab`**: original, prototip, OOF baseline, OOF delivery i temporal. El hash canònic és `4803ca016040aeab3411fa84db11421d1cb8e37828f7b27f878c21bddead7f16`. `rebuild_prior` copia aquesta taula (`training.py:190`). La recalibració escriu `qtab_oof.json` a part (`:143`), però no actualitza `temporal/prior.json`; el validador autoritatiu copia precisament aquest últim. Per tant, no es pot considerar que els escenaris residuals del resultat autoritatiu siguin calibrats independentment dels labels reservats. L'efecte no és simètric: el prototip pot substituir els escenaris de l'hora zero amb quantils condicionals, mentre el baseline depèn de `qtab` també a l'hora zero.

Els perfils mensuals OOF sí que exclouen les setmanes indicades: recalcular-los amb els CSV reprodueix els perfils publicats amb diferència màxima 0 en demanda, solar i irradiància, i menor de 1e-16 en mercat. En passar als perfils finals canvien materialment algunes cel·les:

| Perfil | RMSE de canvi entre cel·les comunes | Canvi absolut màxim |
|---|---:|---:|
| Consum | 18,97 kW | 106,10 kW |
| Solar ja ×12 | 49,42 kW | 381,27 kW |
| Preu | 0,00649 €/kWh | 0,02836 €/kWh |
| CO₂ | 0,00774 kg/kWh | 0,02851 kg/kWh |

Aquest canvi és normal en un refit, però pot moure les bases de persistència i les distàncies dels anàlegs. No és evidència directa de la causa de la regressió oficial.

El baseline temporal s'entrena amb memòria contínua en `training.py:284`, sense passar `split_mask`. Això permet que 432 orígens d'entrenament tinguin alguna observació de les setmanes reservades en les 48 hores anteriors. Són observacions passades legalment disponibles en una execució contínua, però trenquen l'aïllament estricte dels labels reservats quan s'utilitzen per ajustar el model. També introdueixen un canvi de distribució: els models residuals/quantils es construeixen amb resets a les particions, i l'avaluació autoritativa força cold starts a cada setmana.

## Scripts històrics que no s'han de tornar a executar sense reparar

`training.py:274` i `net_candidates.py:14` carreguen actualment `mode_dificil/model_net.txt`, que ja és residual. `forecast_cache` l'interpreta com una previsió directa i no hi suma la base física. El cache resultant no representa ni el baseline original ni el prototip desplegat. Els scripts també depenen de caches i del mòdul `sweep` absents del paquet. Els resultats autoritatius executant agents complets continuen sent l'evidència històrica preferent.

`train_final_models.py` sí que suma correctament la base residual, fixa les iteracions abans del refit i entrena sobre les 5.993 hores publicades. Les 140.244 files de demanda declarades són parelles origen-horitzó, no hores independents. Els set quantils només tenen 5.993 files a h=0. Aquest desequilibri explica que una validació agregada de 24 horitzons pugui ocultar errors de l'hora que realment s'executa.

## Hipòtesis que requereixen experiments

1. Part del guany OOF respecte del baseline podria provenir del canvi de recepta i de cold starts, i no del canvi d'arquitectura desplegat.
2. Els anàlegs de mercat poden ser més sensibles al canvi estacional que el predictor residual del baseline.
3. El canvi dels perfils i de la biblioteca d'anàlegs en el refit pot reduir o invertir el guany observat.
4. La taula residual original pot perjudicar de manera diferent les dues polítiques.

Cap d'aquestes hipòtesis identifica per si sola el motiu dels −0,34 punts oficials, perquè no disposem del test ocult.

## Correcció implementada

`research/train_clean.py` ofereix una pipeline portable amb receptes fixes del baseline oficial i del prototip. Reconstrueix perfils només amb train, separa les memòries a les particions i exigeix trajectòries completes de 24 hores autoritzades. Calcula dues taules residuals noves amb models interns que no han vist les setmanes de calibratge; després refà els models externs amb tot el train autoritzat sense early stopping sobre l'avaluació. Registra dates exactes, llavors, paràmetres, versions, recomptes, temps i hashes. Conserva la taula antiga amb el nom explícit `qtab_original` només per a diagnòstics controlats.

La comprovació de compilació i d'índexs s'ha executat: al tall 2025-12-08 hi ha 1.296 hores d'entrenament i 1.227 orígens amb trajectòria completa. Les dues setmanes internes de calibratge són 2025-09-15/21 i 2025-10-13/19. No es dedueix cap puntuació d'aquesta comprovació de construcció.
