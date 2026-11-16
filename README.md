# Datathon FME 2026 — Riviera Maya Boys

**2nd place — UPC Sostenible challenge: "Battery Optimisation for the Campus" (Datathon FME 2026, UPC).**

*A control agent for a campus battery (6,000 kWh / ±1,500 kW, 12x solar, only 3 h of known prices) that decides every
hour whether to charge or discharge, minimising CO2 (70 %) and electricity cost (30 %). The solution combines
horizon-specialised LightGBM forecasts of market prices and campus demand, scenario-based uncertainty, and an exact
dynamic-programming optimiser (0 infractions, ~18 s for the whole test). Our key insight: the public leaderboard
only scored the first half of each test week and the private one the second half, so we submitted two mirrored
strategies — one maximising the public ranking (27.36, 2nd) and one built for the private ranking.*

Final submissions: [`lliurament_final/`](lliurament_final/). Context: this repository started from the organisers'
starter kit and was developed by a team of three during the 24-hour datathon.

---

> ## Lliurament final (Mode Difícil): [`lliurament_final/`](lliurament_final/)
>
> | | Solució | Score públic |
> |---|---|---|
> | **Versió 1** | [`lliurament_final/versio_1_public_27.36/`](lliurament_final/versio_1_public_27.36/) | **27,36** (2n) |
> | **Versió 2** | [`lliurament_final/versio_2_privat_13.33/`](lliurament_final/versio_2_privat_13.33/) | 13,33 (pensada per al rànquing privat) |
>
> Són les dues solucions seleccionades per al jurat. El README de la carpeta explica l'agent, per què hi ha
> dues versions i l'historial de la resta d'enviaments.
>
> Mode estàndard: [`agent.py`](agent.py) a l'arrel (6,43 punts).

---

# Repte UPC Sostenible: Optimització de la Bateria del Campus
## Datathon 2026 — Transició Energètica i Descarbonització

Repositori oficial del repte d'optimització energètica, ciència de dades i transició ecològica promogut per **UPC Sostenible** per a la gestió d'una bateria industrial de 2.000 kWh i 500 kW connectada al campus de la Universitat Politècnica de Catalunya.

L'objectiu dels participants és programar una política de control (`BatteryAgent`) per gestionar de forma òptima la càrrega i descàrrega de la bateria en intervals horaris, minimitzant simultàniament les emissions de $\text{CO}_2$ de la xarxa elèctrica (70% del pes) i el cost econòmic de la factura elèctrica (30% del pes) respecte a l'escenari base del campus operant sense emmagatzematge.

---

### Estructura del Projecte

```text
├── docs/                           # Documentació oficial del repte
│   ├── ENUNCIAT.md                 # Bases tècniques i formulació matemàtica del problema
│   ├── GUIA_VARIABLES.md           # Especificació detallada de telemetries i sensors
│   └── enunciat_participants_latex.md # Documentació complementària en format LaTeX
│
├── starter_kit/                    # Paquet de treball per als participants
│   ├── README.md                   # Guia ràpida de desenvolupament
│   ├── agent.py                    # Plantilla de la classe BatteryAgent (heurística base)
│   ├── check_submission.py         # Validador local de solucions per a la consola
│   ├── environment.py              # Motor de simulació seqüencial i avaluador de mètriques
│   ├── battery_spec.json           # Paràmetres físics del sistema d'emmagatzematge
│   └── tutorial.ipynb              # Quadern Jupyter de referència
│
├── data/                           # Conjunts de dades
│   ├── train/                      # Dades d'entrenament (6 fitxers CSV d'un any sencer)
│   └── test_hidden/                # Conjunt de test secret utilitzat per la plataforma web
│
├── scripts/                        # Scripts interns de curació de dades
│   ├── consolidate_data.py         # Consolidació de telemetries brutes de sensors
│   ├── download_ree.py             # Descàrrega de preus PVPC i intensitat de CO2 (API REE)
│   └── split_train_test.py         # Generador de les particions temporals train/test
│
├── app.py                          # Plataforma web oficial d'avaluació i Leaderboard (Streamlit)
├── requirements.txt                # Dependències de Python del sistema
└── README.md                       # Documentació general
```

---

### Execució de la Plataforma Web Localment

Per iniciar el servei d'avaluació i classificació en temps real:

```bash
pip install -r requirements.txt
streamlit run app.py
```
