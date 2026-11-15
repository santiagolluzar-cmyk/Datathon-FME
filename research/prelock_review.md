# Revisió independent abans de bloquejar el candidat

No s’ha executat cap política sobre el període final durant aquesta revisió.

El camí actual de `market75_k16` implementa la configuració descrita: mercat 75% analògic amb 16 veïns i 25% LGB; demanda directa; quantils condicionals de l’hora actual; residuals nous per al futur; horitzó 24; ràtio 0,6345. Les tres còpies del candidat superen l’escàner AST subministrat.

La separació del nou entrenament és correcta per als components utilitzats: perfils només de train, reinici de History en canviar la màscara, purga de trajectòries completes de 24 hores, biblioteca analògica sense etiquetes excloses i residuals calculats amb models/perfils interns que no han vist la seva calibració. Els hashes dels models coincideixen amb les metadades i els intervals d’entrenament/calibració precedeixen els períodes avaluats.

S’han comunicat aquests punts abans de l’obertura final:

1. L’`end` del manifest final és l’última hora inclosa, però els selectors del codi interpreten `end` exclusiu. El selector genèric dona 1.007 hores. Cal marcar final exclusiu a 2026-08-16 23:00:00, o seleccionar explícitament les últimes 1.008 files i comprovar-ne el recompte.
2. `terminal_value` és l’única clau de configuració no utilitzada. A zero no canvia el candidat, però cal retirar-la o rebutjar valors no zero.
3. La represa d’experiments confia només en el nom. Cal validar configuració i fingerprints abans de reutilitzar resultats després d’un canvi de codi/models.
4. La memòria cau de dades inclou el hash d’`eval_local.py`, però li falta `eval_dificil.py`, que és el carregador executat.
5. El període final és nou per a la selecció d’aquest candidat, però no històricament intacte: els recomptes de quantils heretats i proves històriques inclouen juliol. Això s’ha d’explicar com a límit experimental, sense afirmar una independència absoluta.

El refit amb tot train segueix la mateixa recepta fixa, però canvia models, perfils, biblioteca analògica i setmanes internes triades per posició del calendari. Per això cal conservar separadament el ZIP entrenat abans del final i el ZIP refit. Només el primer és exactament l’agent de la comprovació reservada. `qtab_original` resta als priors encara que el candidat amb `residual_source=clean` no llegeix aquesta taula; per afirmar exclusió a nivell de tots els bytes de l’artefacte, cal retirar aquesta informació no utilitzada.

No s’ha detectat cap altre bloqueig de correcció algorísmica en el camí del candidat. Els hashes del codi revisat consten a `prelock_review.json`; els canvis posteriors requereixen distingir aquesta instantània.
