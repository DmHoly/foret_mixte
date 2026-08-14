# Base de combos — plan

Objectif : après chaque partie simulée, logguer chaque coup avec son gain
immédiat, pour ensuite filtrer "quels coups/combos rapportent le plus de
points pour N cartes engagées".

## Schéma d'un enregistrement (une ligne = un coup)

```
{
  "action": ("dweller", did, tree_idx, pos) | ("tree", tid) | ...,
  "card_name": "WOOD_ANT",           # nom lisible
  "cost": 3,                          # cartes défaussées pour payer
  "score_delta": 27,                  # gain immédiat au score
  "bonus_paid": true/false,           # bonus jumelles déclenché ?
  "trigger": "free_dweller" | "cave_discard" | "normal" | ...,
  "turn": 137,
  "forest_snapshot_summary": {...},   # contexte minimal (n_trees, dweller_count clés)
  "policy": "greedy" | "mcts" | "beam",
  "seed": 2026,
}
```

## Ce qu'on peut en tirer

- Trier par `score_delta / cost` -> "meilleur rapport points/cartes"
- Grouper par `card_name` -> quelle carte est le meilleur investissement
  en moyenne (sur toutes les parties)
- Isoler les moments où `bonus_paid=true` -> mesurer l'écart de valeur
  entre pose normale et pose bonus pour la même carte
- Repérer les combos temporels : deux gros deltas consécutifs sur peu de
  tours (ex. Fourmi des bois qui explose après une pose de Hêtre)

## Implémentation minimale

- Un logger simple ajouté dans une boucle de simulation externe (comme le
  script d'instrumentation qu'on vient d'utiliser), écrivant en JSONL
  (une ligne JSON par coup) dans reference/combo_log.jsonl.
- Pas besoin de toucher engine.py/game.py : tout se fait en observant
  score_before/score_after autour de chaque apply(), comme dans le script
  d'analyse de cette session.
- Une fois quelques centaines de parties loguées, un petit script d'analyse
  (pandas ou juste groupby en Python pur) pour sortir le classement.

## Pas encore fait

Rien codé, juste le plan. À lancer si tu veux qu'on construise ça.
