# Archive

Fichiers d'itération conservés pour l'historique, remplacés par des versions
plus propres et non référencés ailleurs dans le projet.

- `run_mcts.py`, `run_mcts2.py` — bancs d'essai MCTS ad hoc, remplacés par
  `bench.py mcts` / `bench.py mcts_pairwise_hybrid`.
- `run_narrated.py` — rejeu de partie MCTS avec rollout classique, remplacé
  par `run_narrated_hybrid.py` (config gagnante : rollout court + modèle
  contrastif).
- `run_stats.py` — statistiques agrégées avec rollout classique, remplacé
  par `run_stats_hybrid.py`.
- `run_trajectories.py` — dump de trajectoires de score greedy vs MCTS,
  exploration ponctuelle non reprise ailleurs.
- `REVUE.md` — audit d'une version antérieure et différente du moteur
  (`cards_data.py`/`deck_data.py`, avant le renommage en `cards.py`). Les
  défauts qu'elle documente ont été corrigés ; voir la section « Corrections
  apportées » du `README.md` racine pour l'état actuel.

`reference/archive/plots/` contient les versions v1-v3 des graphiques de
diagnostic du modèle de valeur (`feature_importance*.png`,
`true_vs_pred*.png`) ; seules les versions v4 (les plus récentes) restent
dans `reference/`.
