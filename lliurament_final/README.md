# Lliurament final — Riviera Maya Boys (Mode Difícil)

Aquestes són les **dues solucions oficials** seleccionades a la plataforma per al jurat. Cada carpeta conté
exactament el `submission.zip` enviat (mateixos bytes) i el seu `agent.py` per poder-lo llegir directament.

| | Carpeta | Score públic | Per a què |
|---|---|---|---|
| **Versió 1** | [`versio_1_public_27.36/`](versio_1_public_27.36/) | **27,36** (2n al rànquing públic) | Rànquing públic |
| **Versió 2** | [`versio_2_privat_13.33/`](versio_2_privat_13.33/) | 13,33 | Rànquing privat (decideix el podi) |

Empremtes SHA-256 dels paquets enviats:
- Versió 1: `ff74a035809a2b7e…`
- Versió 2: `8ccd6dc093f85f26…`

**Resultat final oficial (Mode Difícil): 2n lloc amb 25,46 punts** — vegeu [`classificacio_final.png`](classificacio_final.png). La simulació local del privat preveia 26,9.

## Per què dues versions tan diferents

Les dues comparteixen el mateix agent. Només canvia **com acaba cada setmana de test la bateria**.

1. **Previsió de preu i CO₂ (+3 a +24 h)**: models LightGBM especialitzats per horitzó, combinats al 60 % amb una
   previsió per anàlegs (les 16 trajectòries de train més semblants a les 3 hores conegudes).
2. **Previsió de la demanda neta (consum − solar ×12)**: models especialitzats per a l'hora actual i les
   següents. De 0 a 5 h el comptador queda congelat i la demanda es coneix exactament.
3. **Incertesa**: escenaris d'error calculats fora de mostra, per horitzó i hora del dia. Per a l'hora actual,
   9 quantils condicionats a la situació (núvols, rampes de consum).
4. **Decisió**: optimitzador exacte de 24 h (programació dinàmica sobre funcions convexes) amb aquests
   escenaris, i 0 infraccions.

El test són 12 setmanes de 168 h. Amb els enviaments hem comprovat que el **rànquing públic és la primera
meitat de cada setmana (hores 0–83)** i el **privat, la segona (hores 84–167)**:

| Versió enviada | Com gestiona la bateria | Públic (web) | Privat (simulació local) |
|---|---|---|---|
| v5 (agent base) | normal | 23,44 | 19,4 |
| v6 | buida al final de cada setmana | 20,15 | 23,2 |
| **Versió 1** | plena al final de cada setmana | **27,36** | 14,5 |
| **Versió 2** | buida al final + plena a l'hora 84 | 13,33 | **26,9** |

La Versió 2 va fer al públic exactament el que preveia la simulació si el tall és a l'hora 84
(13,33 / 20,15 = 0,662, contra 0,663 previst). Això confirma on cau el tall. La Versió 1 maximitza el
rànquing públic i la Versió 2, el privat.

## Executar

Cada `submission.zip` conté `agent.py`, `prior.json`, 24 models LightGBM en text i `analog.npz`. Cal
descomprimir-lo en una carpeta i mantenir els fitxers junts. Validació local:

```bash
python starter_kit/check_submission.py lliurament_final/versio_1_public_27.36/submission.zip
python starter_kit/check_submission.py lliurament_final/versio_2_privat_13.33/submission.zip
```

Temps: ~18 s per a les 2.016 h del test (límit 60 s).

## Altres versions (historial, no seleccionades)

| Versió | On és | Públic |
|---|---|---|
| Mode estàndard (bateria 2.000 kWh) | [`../agent.py`](../agent.py) | 6,43 |
| Difícil v1 (LightGBM + MPC amb escenaris) | [`../hard_work/original/`](../hard_work/original/) | 23,45 |
| Híbrid (anàlegs + quantils) | [`../lliuraments/submission_hybrid16_fulltrain.zip`](../lliuraments/) | 23,63 |
| Prototip | [`../mode_dificil/submission.zip`](../mode_dificil/) | 23,11 |
| v4 (models per horitzó + anàlegs) | [`../mode_dificil/v4/`](../mode_dificil/v4/) | 23,38 |
| v5 (agent base de les dues finals) | [`../mode_dificil/v5/`](../mode_dificil/v5/) | 23,44 |
| v6 (buida al final de cada setmana) | [`../mode_dificil/v6_final_bloc/`](../mode_dificil/v6_final_bloc/) | 20,15 |
| v8 (híbrid + plena al final de cada setmana) | [`../mode_dificil/v8_public_hibrid/`](../mode_dificil/v8_public_hibrid/) | 26,92 |

Els scripts de recerca (simulador, entrenament, validació creuada, proves) són a
[`../mode_dificil/v5/recerca/`](../mode_dificil/v5/recerca/) i a [`../research/`](../research/).
L'anàlisi del tall públic/privat és a [`../mode_dificil/LLEGIU_ABANS_DE_TRIAR.md`](../mode_dificil/LLEGIU_ABANS_DE_TRIAR.md).
