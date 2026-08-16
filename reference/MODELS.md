# Modèles de valeur — quel fichier est quoi

Plusieurs `.joblib`/`.npz` cohabitent dans `reference/` parce que chaque
tentative d'amélioration du modèle de valeur a été gardée comme point de
comparaison plutôt que supprimée (voir la philosophie générale du dépôt :
un résultat nul ou négatif reste un résultat, pas une erreur à effacer).
Ce fichier sert d'index pour ne pas s'y perdre.

## Fichiers vivants (chargés par défaut)

| Fichier | Rôle |
|---|---|
| `pairwise_model.joblib` | Modèle de valeur contrastif **utilisé par défaut** par `value_policy.make_pairwise_hybrid_leaf_eval()` (donc par tout MCTS instancié sans `model_path` explicite). Réentraîné le 16/08 avec `gen_pairwise_dataset.py` + `train_pairwise_model.py` sous le code actuel (heuristique `choose_draw_source` **forte**, cible carte forte en Clairière). |
| `pairwise_dataset.npz` | Dataset de paires (diff features -> diff gain réel) ayant servi à entraîner `pairwise_model.joblib` ci-dessus. 150 parties, 20193 paires, 78 features. |
| `value_model.joblib`, `value_dataset.npz` | Modèle de valeur **absolu** (MLP), approche antérieure au modèle contrastif. Toujours chargé par `value_policy.make_leaf_eval()`/`make_hybrid_leaf_eval()`, utilisé par `bench.py` comme point de comparaison. Limite connue : MAE (~12 pts) plus grande que l'écart réel entre deux coups candidats (~9 pts) -- voir la docstring de `gen_pairwise_dataset.py`, c'est justement pour corriger ce défaut que l'approche contrastive a été introduite. |

## Sauvegardes historiques (non chargées par défaut, gardées pour comparaison/rollback)

| Fichier | Contexte |
|---|---|
| `pairwise_model_pre_clairiere_forte.joblib`, `pairwise_dataset_pre_clairiere_forte.npz` | Le modèle contrastif tel qu'il était **avant** le réentraînement du 16/08 -- entraîné sous l'ancienne heuristique de Clairière (toujours la carte la moins chère). C'est ce modèle périmé qui faisait perdre MCTS+forte face à Greedy+forte (12/30) ; le réentraînement (voir `bench_heuristics.py`) fait remonter le taux de victoire à ~59%. Gardé pour pouvoir reproduire la comparaison avant/après. |
| `pairwise_model_stale_backup.joblib`, `pairwise_dataset_stale_backup.npz` | Snapshot du 15/08 19h48, avant le réentraînement "sous les règles actuelles" de cette même journée (commit `233b14e`) -- prédate même l'ajout des features Clairière (`clearing_size`/`clearing_min_cost`) au vecteur de features. Conservé comme point de repère le plus ancien. |
| `pairwise_model_true_original_aligned.joblib`, `pairwise_model_retrain1_no_clearing_aligned.joblib` | Paire de modèles produits lors de l'expérience "ajouter les features Clairière au modèle de valeur" (commit `013edbc`, 15/08 19h57) : le premier reprend l'entraînement d'origine réaligné sur le nouveau schéma de features (colonnes Clairière à zéro) pour servir de témoin, le second est réentraîné avec les features Clairière réellement peuplées. Résultat **nul** (l'ajout de ces features n'a pas amélioré le jeu de façon mesurable) -- gardés comme trace de la tentative, pas comme candidats à adopter. |

## Comment regénérer le modèle vivant

```
python reference/gen_pairwise_dataset.py [n_parties]   # -> reference/pairwise_dataset.npz
python reference/train_pairwise_model.py                # -> reference/pairwise_model.joblib
```

Les deux scripts utilisent le code actuel de `game.py`/`search.py` tel
quel (aucune heuristique n'y est figée en dur) : relancer cette paire de
commandes régénère toujours un modèle cohérent avec le comportement de jeu
du moment. C'est exactement la source du bug corrigé le 16/08 -- le modèle
vivant datait d'avant un changement de comportement par défaut (l'ancienne
heuristique de Clairière), pas d'un bug dans ces scripts.

Pour comparer deux modèles en tête-à-tête, `reference/bench_heuristics.py`
accepte n'importe quel `model_path` via `value_policy.load_pairwise_model`
sans toucher au fichier par défaut.

## Résultat de référence (16/08, `pairwise_model.joblib` réentraîné)

Tournoi rond-robin complet entre les 4 variantes {Greedy, MCTS(150it)} x
{Clairière forte, Clairière faible}, sièges alternés, `python
reference/bench_heuristics.py` :

| Match | Score | Écart moyen (SE) | Médiane |
|---|---|---|---|
| B vs A | 69/100 | +47.6 (7.9) | +43.5 |
| D vs C | 21/30 | +36.1 (14.2) | +22.5 |
| B vs C | 20/30 | +47.8 (15.5) | +10.0 |
| D vs B | 15/30 (nul) | -29.7 (14.9) | +0.5 |
| D vs A | 21/30 | -2.5 (11.7) | +11.5 |
| C vs A | 15/30 (nul) | -36.9 (18.6) | -1.5 |

Classement agrégé (victoires / parties jouées) : **B 65.0%, D 63.3%, C
37.8%, A 33.1%**.

Deux points notables par rapport au tournoi avec l'ancien modèle (D à
40% contre B, C à 73% contre A) :

- **D vs B est passé de 12/30 à 15/30** (quasi-parité) : le réentraînement
  comble l'essentiel de l'écart qui faisait perdre le bot par défaut face
  à Greedy+forte.
- **C (MCTS + Clairière faible) s'est dégradé face à A** (73% -> 50%). Le
  modèle vivant est maintenant entraîné sur des parties jouées sous la
  heuristique **forte** ; l'utiliser avec un MCTS qui joue lui-même en
  heuristique **faible** le place hors de sa distribution d'entraînement
  -- l'inverse du problème corrigé pour D. Sans conséquence en pratique
  (l'heuristique faible n'est plus utilisée nulle part par défaut), mais
  c'est une bonne illustration du principe : un modèle de valeur n'est
  fiable que sur les états qui ressemblent à ceux vus à l'entraînement.
