# Plan de refactoring — effets, bonus jumelles, champignons

Basé sur reference/cartes_effets.md. Trois briques indépendantes, à faire
dans cet ordre (chacune teste et débloque la suivante).

## Brique 1 — Paiement comme décision réelle (prérequis des deux autres) [FAIT]

Implémenté (game.py) : `card_symbol()`, `choose_payment(..., preferred_symbol=)`
qui priorise la défausse d'une carte du bon symbole sans changer le coût
payé, et `Game.apply()` qui calcule automatiquement le symbole de la carte
posée, l'utilise comme préférence de paiement, et expose le résultat via
`self.last_bonus_paid` (bool, recalculé à chaque `apply`, disponible pour
la brique 2). Testé (tests/test_rules.py) : préférence de défausse vérifiée
isolément, et bout-en-bout via `Game.apply` sur un ROE_DEER. 17/17 tests
passent, bench.py toujours cohérent avec l'oracle.

Aujourd'hui `choose_payment()` est une heuristique figée qui ignore le bonus
jumelles. Or presque tout le catalogue ("SI payé avec bonus jumelles...")
dépend de CE choix. Impossible de simuler ces cartes sans d'abord savoir
si le joueur a payé "en couleur" ou non.

Proposition minimale (pas besoin de rendre payment() un vrai noeud de
recherche tout de suite) :
- `card_symbol(card, half_index)` : symbole/couleur imprimé sur la carte
  posée (déjà dispo via `symbol` dans `_find_card`).
- **Condition du bonus, confirmée par Mehdi** : le bonus se déclenche si
  AU MOINS UNE des cartes utilisées pour payer porte le même symbole que
  celui imprimé sur la carte posée (pas besoin que tout le paiement soit
  homogène). Exemple : le Loup porte le symbole "Sapin blanc" ; payer avec
  au moins une carte de symbole Sapin blanc déclenche le rejeu de tour.
- `choose_payment()` gagne un paramètre `prefer_bonus: bool` : si le
  dweller posé a un effet bonus et qu'une carte du bon symbole est
  disponible en main pour payer, la préférer dans le choix de défausse.
- Plus tard (si la qualité de jeu le demande) : exposer les deux choix de
  paiement (bonus vs non-bonus) comme actions distinctes dans la recherche.

## Brique 2 — Effets post-pose (pioche, jouer gratuit, rejouer un tour)

### Batch 1 (sans sous-choix) [FAIT]

Implémenté (engine.py : tables DRAW_FIXED, DRAW_IF_BONUS, DRAW_PER_COUNT,
REPLAY_ALWAYS, REPLAY_IF_BONUS, TREE_DRAW_FIXED ; game.py :
`Game._resolve_dweller_effect`, branchement dans `apply()`) :
- Fouine, Tortue cistude : pioche fixe.
- Chevreuil, Daim, Hérisson commun : pioche si bonus jumelles payé.
- Renard roux (1×Lièvre), Loup (1×Cervidé) : pioche proportionnelle à un
  compteur de forêt déjà tenu incrémentalement.
- Geai des chênes : rejeu de tour inconditionnel (apply() ne fait pas
  avancer `self.current`).
- Loup : rejeu de tour SI bonus jumelles payé (en plus de sa pioche).
- Bouleau (arbre, pas dweller) : pioche fixe à la pose, table séparée
  `TREE_DRAW_FIXED` indexée par tree_id.

Laissés de côté dans ce batch, comme prévu :
- Ours brun (dépend de la clairière, non simulée).
- Sapin Douglas (dweller_id précis pas identifié avec certitude dans
  cards.py contre la description de Mehdi — à confirmer avant d'ajouter
  une entrée REPLAY_IF_BONUS, pour ne pas deviner).

Testé (tests/test_rules.py, `test_dweller_draw_and_replay_effects`) : un
scénario par mécanique (pioche fixe, rejeu inconditionnel, pioche
proportionnelle). 18/18 tests passent, bench.py toujours cohérent avec
l'oracle.

### Batch 2 (avec sous-choix) [FAIT partiellement]

Implémenté : mécanisme générique `Game.pending_effect` (game.py) — un
triplet (filtre, argument, cartes restantes) qui suspend le passage de
tour du joueur courant tant qu'il n'a pas résolu l'effet. `legal_actions()`
expose alors uniquement ("free_dweller", did, tree_idx, pos) pour les
cartes qui correspondent au filtre, et ("skip_effect",) pour décliner
(l'effet est optionnel — hypothèse pas explicitement confirmée par Mehdi
mais cohérente avec le reste du jeu, à vérifier si besoin). `remaining=1`
clôt l'effet après une pose ; `remaining=None` le garde ouvert (usages
illimités) jusqu'au skip.

Cartes câblées (`DWELLER_PLAY_FREE_IF_BONUS` / `TREE_PLAY_FREE_IF_BONUS`
dans engine.py) :
- Blaireau européen : bonus -> joue 1 animal gratuit (n'importe lequel).
- Lucane : bonus -> joue 1 oiseau gratuit.
- Cerf élaphe : bonus -> joue 1 autre Cerf élaphe gratuit (filtre par
  dweller_id exact).
- Sapin blanc (arbre, pas dweller) : bonus -> joue 1 animal gratuit.
- Moustique : bonus -> joue autant de Chauve-souris que voulu (illimité).
- Sapin Douglas (arbre) : rejoue un tour, inconditionnel (`TREE_REPLAY_ALWAYS`,
  même mécanisme que Geai des chênes, pas de sous-choix).

`search.py` mis à jour en conséquence : les trois politiques (greedy, beam,
MCTS) et leurs fallbacks gèrent maintenant `skip_effect`/`free_dweller`
correctement (`_fallback_action` retourne `skip_effect` au lieu de `draw`
quand un effet est en attente, puisque piocher n'est pas légal dans cet
état). Testé bout-en-bout (greedy et beam sur des parties complètes).

### Batch 2c (Raton laveur) [FAIT]

Implémenté (engine.py : `CAVE_CHOICE_DWELLERS` ; game.py : pending_effect
de type `("cave_choice", None, None)`, actions `("cave_discard", n)`) :
le Raton laveur envoie N cartes de la main à la Grotte (1 pt chacune via
`Forest.cave`, déjà scoré) et pioche N cartes. Le NOMBRE est un vrai choix
exposé à l'arbre de recherche ; LESQUELLES cartes partent reste une
heuristique (comme le paiement), documentée comme telle dans le code.
`search.py` (greedy_action) adapté pour évaluer cette nouvelle forme
d'action. Testé (tests/test_rules.py) : 21/21 tests passent, bench
scoring et policies cohérents.

### Taupe [FAIT]

Confirmé par Mehdi : chaîne d'actions de pose payantes normales, arrêt
automatique dès qu'on ne peut plus payer (ou choix stratégique explicite
via skip_effect). Implémenté (engine.py : `PLAY_CHAIN_DWELLERS` ; game.py :
pending_effect `("play_chain", None, None)`, réutilise les poses payantes
normales via `_payable_actions` partagé avec le tour standard). Chaque pose
dans la chaîne paie normalement (choose_payment sans préférence de bonus,
simplification assumée) et ne redéclenche pas ses propres effets de pose
en cascade — même principe que le "jouer gratuit" du batch 2b, pour rester
borné. La chaîne s'arrête d'elle-même quand `legal_actions()` ne renvoie
plus que `skip_effect` (plus aucune pose finançable). Testé
(tests/test_rules.py, `test_mole_play_chain`) : 22/22 tests passent, bench
scoring et policies cohérents.

Ajouter une table déclarative dans engine.py, indexée par dweller_id (et
éventuellement par "bonus payé ou non") :

```python
# EFFECT[dweller_id] = (kind, n) ou (kind, n_si_bonus)
# kind ∈ {"draw", "draw_if_bonus", "play_free_from_hand", "replay_turn"}
```

Dans `Game.apply()`, après le scoring/pose normale, une nouvelle étape
`_resolve_effect(player, did, bonus_paid)` :
- `draw` / `draw_if_bonus` : appelle `_draw_one` N fois — facile, pas de
  branchement de recherche supplémentaire.
- `play_free_from_hand` (Blaireau, Lucane, Sapin blanc dweller, Cerf
  élaphe, Taupe, Raton laveur, Moustique) : NÉCESSITE un sous-choix (quelle
  carte de la main jouer/défausser). C'est le vrai point dur : ça ouvre un
  noeud de décision imbriqué. Proposition : traiter comme une "action en
  attente" (`pending_effect`) que `legal_actions()` expose au prochain
  appel avant de rendre la main à l'adversaire, plutôt que de le résoudre
  par une heuristique cachée dans `apply()`. Ça garde le moteur explicite
  et laisse MCTS explorer le sous-choix.
- `replay_turn` (Geai des chênes inconditionnel, Loup/Sapin Douglas si
  bonus) : ne PAS changer `self.current` à la fin de `apply()` pour ce
  tour-ci.

Ours brun (clairière → grotte) et Fouine (pioche simple) rentrent dans le
même mécanisme que "draw", sans sous-choix.

## Brique 3 — Effets permanents de champignons

Amanite tue-mouches, Cèpe de Bordeaux, Girolle, Coulemelle : chacun réagit
à CHAQUE pose future (pas seulement la sienne), tant qu'il est en jeu.
C'est un état persistant par joueur, pas un delta instantané comme le
reste du moteur.

Proposition : `Forest` gagne un champ `active_mushroom_triggers: list`
peuplé à la pose d'un champignon (`(condition, effect)`), et
`Game.apply()`, après CHAQUE pose (pas seulement celle du champignon),
boucle sur les triggers actifs du joueur et les évalue :
- Amanite : condition = carte posée est un ANIMAL.
- Cèpe de Bordeaux : condition = position posée == Top.
- Coulemelle : condition = position posée == Bottom.
- Girolle : condition = carte posée porte le symbole arbre (ambigu — à
  reconfirmer : symbole de QUEL arbre ? le sien ou n'importe lequel ?).

Impact recherche : ceci ajoute un coût constant par pose (boucle sur peu
d'éléments, pas un nouveau facteur de branchement), donc pas de souci de
performance pour MCTS contrairement à la brique 2.

## Brique 3 — Champignons à effet permanent [FAIT]

Implémenté (engine.py : `MUSHROOM_TRIGGER` indexée par dweller_id de
champignon -> (condition, argument) ; game.py : `Game._resolve_mushroom_triggers`,
appelée après CHAQUE pose de dweller par le joueur, sur les trois chemins
existants : pose normale, `free_dweller`, et la chaîne Taupe) :
- Amanite tue-mouches : pioche à chaque habitant ANIMAL posé.
- Cèpe de Bordeaux : pioche à chaque habitant posé en position Top.
- Coulemelle : pioche à chaque habitant posé en position Bottom.
- Girolle : pioche à chaque habitant posé (portée à "n'importe quel
  habitant", puisque seuls les habitants portent un symbole imprimé dans
  ce moteur — les arbres n'en ont pas, voir card_symbol).

Portée limitée aux habitants (pas les arbres). Multiplié par le nombre
d'exemplaires du champignon en jeu (`Forest.dweller_count`), comme
Renard/Loup. Convention : la condition est évaluée APRÈS la pose de la
carte elle-même, donc poser le champignon peut déclencher son propre effet
sur cette même pose (documenté dans engine.py, à corriger si Mehdi
infirme). Testé (tests/test_rules.py, `test_mushroom_permanent_triggers`) :
vérifie le déclenchement sur la pose du champignon lui-même ET sur une
pose ultérieure d'un autre habitant. 23/23 tests passent, bench scoring et
policies cohérents.

## État global du refactoring

Toutes les briques identifiées sont faites, y compris l'Ours brun (Clairière
et Grotte implémentées, voir game.py : `Game.clearing`, `_add_to_clearing`,
`choose_draw_source`, et `CLEARING_TO_CAVE_DWELLERS` dans engine.py). Le
catalogue de reference/cartes_effets.md est donc entièrement câblé. Un oubli
du batch 1 (Fougère arborescente, pioche fixe) a été
rattrapé après coup, repéré en corrigeant une erreur de script sur le
catalogue — pas de garantie qu'il n'y en ait pas d'autres du même genre,
une relecture croisée du catalogue contre les tables DRAW_FIXED/DRAW_IF_BONUS/
DRAW_PER_COUNT/REPLAY_*/PLAY_FREE_IF_BONUS/MUSHROOM_TRIGGER dans engine.py
serait utile avant de considérer le refactoring définitivement clos.
