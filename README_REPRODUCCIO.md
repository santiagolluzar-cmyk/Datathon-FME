# Riviera Maya Boys: reproducció de l'auditoria i de l'agent

Tot el treball és local. Cap ordre d'aquest document fa un enviament, validació web, commit o push. Les puntuacions del servidor comunicades per l'equip són 23,45 per al baseline i 23,11 per al prototip; les puntuacions generades aquí són locals i no s'hi poden sumar.

## Fitxers lliurats

- `lliuraments/submission_hybrid16_fulltrain.zip`: candidat recomanat per a un intent exploratori; refit amb tot train. No té puntuació independent per a aquests pesos exactes.
- `lliuraments/agent_validat_prefinal.zip`: pesos exactes de la comprovació final local de 1.008 hores.
- `lliuraments/informe_Riviera_Maya_Boys.pdf`: diagnòstic i resultats, amb limitacions.
- El paquet general `projecte_RMB_reproduible.zip` és per descomprimir i treballar localment. **No és un ZIP de submission**: conté dades, scripts de recerca i múltiples agents.

## Entorn i dependències

Executeu les ordres des de l'arrel extreta del projecte. Els scripts resolen els altres camins amb `__file__`, no amb rutes del nostre ordinador. El `requirements.txt` original es conserva com a material d'entrada; l'entorn efectivament utilitzat en aquesta iteració és:

```text
Python 3.12.14
numpy 2.3.5
pandas 2.2.3
scipy 1.17.0
lightgbm 4.7.0
reportlab 4.4.9
matplotlib 3.10.8
pypdf 6.21.4
```

Els quatre primers paquets Python cobreixen entrenament i avaluació. Els tres últims són per a l'informe. `pdftoppm` serveix per revisar visualment el PDF. Una altra versió de LightGBM o un nombre de fils diferent pot canviar els bytes dels models; compareu també mètriques i paritat funcional.

Per limitar fils a les biblioteques numèriques:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

## Dades i convencions

- Fonts originals: `data/train/`, sis CSV publicats; no hi ha test ocult.
- `eval_dificil.carregar_dades` crea les 5.993 hores comunes i multiplica PV per 12 **una sola vegada**.
- Si es passa un CSV propi a `research/evaluate.py --data`, `pv_kw` ha d'estar ja multiplicat per 12. No tornar a escalar.
- `load_kw` segueix el tutorial: negatiu del comptador, límits 0–5.000 i mitjana horària. L'avaluador exigeix els tres canals d'inversor vàlids; la petita diferència amb el notebook està documentada.
- Mercat observable: hora actual i dues següents. Telemetria passada. La memòria es reinicia als salts; el SOC persistent/reiniciat és una opció separada.
- El score es calcula amb totals de cost i emissions, sense mitjanes de scores de blocs.

## Inspecció i verificacions bàsiques

```bash
python research/environment_audit.py
python research/training_audit.py
python research/audit_agents_optimizer.py
python -m unittest discover -s research -p test_evaluate.py
```

Els paquets històrics són `hard_work/original/submission.zip` i `mode_dificil/submission.zip`. Els fitxers originals i `MANIFEST.json` s'han conservat. Els resultats històrics amb agents OOF es recullen a `research/historical_reproduction.json`; no són un test final intacte.

Els següents diagnòstics són explícitament **dins de mostra / històricament exposats**, no s'han d'utilitzar com a nova evidència independent:

```bash
python research/audit_agents_in_sample.py
python research/audit_agents_quantiles.py
python research/historical_ablations.py
```

## Reentrenar les tres particions forward

Les dates exactes estan fixades a `research/split_manifest.json`. S'entrena només abans del començament de cada prova. Els models interns de residuals exclouen les seves setmanes de calibratge; les trajectòries de 24 hores i els anàlegs no travessen particions ni salts.

```bash
python research/train_clean.py --out research/folds/winter --train-before 2025-12-08 --threads 2
python research/train_clean.py --out research/folds/spring --train-before 2026-03-09 --threads 2
python research/train_clean.py --out research/folds/summer --train-before 2026-06-01 --threads 2
```

Cada directori conté `training_metadata.json`: versions, llavors, receptes, dates, calibratge, recomptes, temps i hashes. No hi ha early stopping sobre el període avaluat. Els recomptes d'arbres dels quantils són heretats del treball anterior, un límit d'independència que es manté explícit.

## Reproduir factorial, refinaments i candidat

Useu noms nous per als resultats reproduïts; no sobreescriviu els registres conservats. La represa d'experiments exigeix coincidència de configuració i fingerprints, i rebutja resultats antics amb codi/models diferents.

```bash
python research/experiments.py --out research/results/factorial_reproduced.json
python research/experiments.py --configs research/refinement_configs.json --out research/results/refinements_reproduced.json
python research/experiments.py --configs research/combination_configs.json --out research/results/combinations_reproduced.json
python research/experiments.py --configs research/guarded_config.json --out research/results/guarded_reproduced.json
```

Aquestes ordres reconstrueixen els agents complets sota `research/agents/` i avaluen els tres blocs. `guarded_config.json` és un diccionari de variants per a `experiments.py`; `build_agent.py --config` requereix **una sola configuració**, no aquest diccionari exterior.

## Prova estacional bloquejada i sensibilitats

La prova bloquejada usa nou setmanes (1.512 hores) anteriors a la reserva final. Pot entrenar amb dates posteriors a un bloc, però n'exclou totes les etiquetes, perfils, retards i trajectòries de calibratge; és desenvolupament, no la prova final.

```bash
python research/train_blocked.py
python research/evaluate_blocked.py --out research/results/blocked_reproduced.json
python research/run_development_stress.py
python research/targeted_stress.py --part all
```

Les dues últimes ordres escriuen les sortides de sensibilitat amb els seus noms habituals. Si voleu conservar també la primera execució, feu una còpia local de `research/results/` abans de reproduir-les. Proven SOC buit/ple, mercat d'1–3 hores, reinicis d'historial, SOC persistent/reiniciat i errors de previsió sintètics. Les còpies `research/stress_copies/` són diagnòstiques i **no s'han d'enviar**.

## Comprovació final congelada

La reserva final són les darreres 1.008 hores observades, del 2026-06-21 19:00 fins al 2026-08-16 22:00. L'extrem final de selecció és exclusiu: 2026-08-16 23:00. No és un test ocult i part de juliol ja havia participat en selecció històrica d'hiperparàmetres.

`research/selection.json` fixa la política abans d'obrir scores d'aquesta prova. No el canvieu a partir del resultat per continuar anomenant la mateixa reserva una comprovació final.

```bash
python research/train_clean.py --out research/folds/locked --train-before '2026-06-21 19:00:00' --threads 2
python research/confirm.py --out research/results/locked_final_reproduced.json
```

`confirm.py` compara baseline refet, prototip refet i candidat amb el mateix tall; prova SOC persistent i reiniciat. Refusa sobreescriure una sortida existent. El fitxer original `results/locked_final.json` conserva la primera obertura.

## Refit amb totes les dades i construcció portable

Entrenar amb tot train manté la política però canvia els models, perfils, residuals i biblioteca d'anàlegs. **La puntuació reservada pertany al paquet anterior al tall; no es transfereix al refit.**

```bash
python research/train_clean.py --out research/folds/full_reproduced --threads 2
python - <<'PY'
import json
from pathlib import Path
selection=json.loads(Path('research/selection.json').read_text())
Path('research/selected_policy_config.json').write_text(json.dumps(selection['config'],indent=2))
PY
python research/build_agent.py --bundle research/folds/full_reproduced --out research/agents/full_reproduced --config research/selected_policy_config.json
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
source=Path('research/agents/full_reproduced')
Path('output').mkdir(exist_ok=True)
with ZipFile('output/submission_reproduced.zip','w',ZIP_DEFLATED) as z:
    for p in sorted(source.iterdir()):
        if p.is_file():
            z.write(p,p.name)
PY
python starter_kit/check_submission.py output/submission_reproduced.zip
```

El verificador subministrat és una prova tècnica; no estima score del mode difícil. Els ZIPs lliurats ja contenen els models necessaris, sense dades futures ni còpies diagnòstiques. La verificació de paritat ZIP/directori i latència consta a `research/results/delivery_verification.json`.

Per reconstruir **els dos ZIPs i la verificació completa**, el script d'empaquetat espera el bundle `research/folds/fulltrain`, la selecció congelada i la primera prova final amb les traces corresponents:

```bash
python research/train_clean.py --out research/folds/fulltrain --threads 2
python research/package_delivery.py --out output/delivery_reproduced
```

Aquesta última ordre torna a escriure `research/results/delivery_verification.json`; conserveu el registre original si voleu comparar temps entre ordinadors. Reexecuta el paquet anterior al tall per comprovar paritat exacta amb les seves traces, i fa una prova d'execució **dins de mostra** del refit. No fa una nova selecció ni atribueix un score fora de mostra al refit complet.

Exemple d'avaluació local d'un directori d'agent (els models han de ser al mateix directori):

```bash
python research/evaluate.py --agent-dir research/agents/locked/selected --start '2026-06-21 19:00:00' --end '2026-08-16 23:00:00' --gap-soc carry --json research/results/manual_check.json --trace research/results/manual_check.csv
```

Per mesurar latència, primer executeu amb monitoratge d'excepcions i sense errors. Després podeu repetir afegint `--no-exception-audit`, amb un altre nom de sortida. El temps local és orientatiu; no equival al temps del servidor.

## Oracle i informe

```bash
python research/oracle.py
python research/make_report.py --out output/informe_Riviera_Maya_Boys.pdf
pdftoppm -scale-to 1500 -png output/informe_Riviera_Maya_Boys.pdf output/informe_page
```

L'oracle llegeix futurs reals de les tres particions de desenvolupament **només per obtenir una referència no desplegable**. No s'ha d'integrar a `BatteryAgent`. El generador d'informe exigeix els resultats finals i la verificació del lliurament completats, llegeix mètriques existents i no executa polítiques ni crea puntuacions.

## Interpretació del següent intent oficial

Consulteu l'informe i els dos ZIPs separats: paquet avaluat abans del tall versus refit amb tot train. Manteniu el baseline oficial i tots els logs. La referència de 24,25 era històrica; no s'ha verificat que segueixi vigent. Ni l'agent ni els scripts garanteixen top 1. Cap enviament està autoritzat per aquestes ordres locals.
