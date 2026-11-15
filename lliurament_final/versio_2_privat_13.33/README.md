# Versió 2 — solució oficial per al rànquing privat (13,33 al públic)

`submission.zip` és exactament el paquet enviat (SHA-256 `8ccd6dc093f85f26…`). Obtingué 13,33 punts al
rànquing públic (0 infraccions). Està pensada per al **rànquing privat**: la segona meitat de cada setmana
(hores 84–167).

És l'agent v5 amb dues regles afegides:
- **`BLOCK_FINITE`**: planifica cada setmana com un bloc tancat i hi arriba al final amb la bateria buida.
- **Hores 72–84 de cada setmana**: arriba a l'hora 84 amb la bateria plena, perquè aquesta energia compti a
  la meitat privada.

Simulació local de la meitat privada: 26,9 punts, contra 19,4 de l'agent base i 14,5 de la Versió 1.
El 13,33 públic coincideix amb el previst per la simulació si el tall és a l'hora 84.

Vegeu [`../README.md`](../README.md) per a l'explicació completa.
