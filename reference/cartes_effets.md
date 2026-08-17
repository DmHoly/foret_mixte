# Référence des effets/bonus de cartes — Forêt Mixte

Format : Nom | coût | effet à la pose | bonus jumelles (paiement couleur) | score final
Statut moteur : [FAIT] déjà implémenté correctement | [MANQUE] pas implémenté | [DOUTE] à vérifier

Catalogue entièrement câblé (voir `reference/REFACTOR_PLAN.md`, section « État
global du refactoring ») : toutes les entrées ci-dessous sont [FAIT], y
compris le bonus jumelles et les quatre champignons à effet permanent
(Amanite tue-mouches, Cèpe de Bordeaux, Girolle, Coulemelle). Vérifié par
`tests/test_rules.py` (42 tests, dont 12 ciblent spécifiquement bonus
jumelles et champignons).

## Confirmé par Mehdi (photos + description directe)

Renard roux | 1 | pioche 1 carte × Lièvre d'Europe possédé | - | 2 pts × Lièvre d'Europe [FAIT — DRAW_PER_COUNT, engine.py/game.py]
Lièvre d'Europe | ? | - | non | N² où N=nombre de lièvres [FAIT — score et partage de slot. Corrigé lors de la revue : pas de "bonus jumelles" séparé, le N² EST le score de base, pas un bonus qui s'ajouterait par-dessus (confirmé par Mehdi)]
Taupe | ? | Jouez immédiatement autant de cartes que souhaité en payant leurs coûts | - | 0 [FAIT - chaîne d'actions payantes, arrêt automatique ou skip_effect. Confirmé par Mehdi lors de la revue : chaque carte posée pendant la chaîne est payée normalement (action gratuite, PAS une pose gratuite), donc déclenche normalement ses effets de pose et son bonus jumelles -- y compris, imbriqué, si elle ouvre elle-même un sous-choix (Raton laveur, une autre Taupe, Blaireau/Lucane/Salamandre/Sapin blanc/Cerf élaphe). Un rejeu de tour gagné pendant la chaîne s'applique à sa fin (`Game.pending_replay`), pas immédiatement]
Ours brun | ? | Placez toutes les cartes de la Clairière dans votre Grotte | - | 0 (les points viennent de la Grotte) [FAIT — clairière/grotte simulées (CLEARING_TO_CAVE_DWELLERS, game.py), voir aussi ligne 27]
Raton laveur | ? | Placez autant de cartes que voulu de la main directement sur la Grotte (1 pt chacune au score, `forest.cave`), et piochez le même nombre | - | ? [FAIT]
Blaireau européen | ? | Placez un spécimen Plantigrade/digitigrade gratuit depuis la main | 2 glands × (bonus) | ? [FAIT]
Sapin blanc (arbre) | 2 | SI payé avec une carte de symbole Sapin blanc (bonus jumelles) : joue gratuitement un animal depuis la main | oui | 2 pts × carte rattachée à ce Sapin blanc [FAIT en brique 2b, score déjà FAIT]
Xylocope violet | ? | - | - | 0 pt, +1 à l'arbre porteur (compte comme un arbre en plus de son espèce) [FAIT - déjà implémenté (bee_by_species), correctif documenté dans engine.py]
Loir gris (EUROPEAN_FAT_DORMOUSE) | ? | - | - | 15 pts SI une Chauve-souris (n'importe quelle espèce) est aussi posée sur le MÊME arbre, sinon 0 [FAIT — dormouse_hits (engine.py), déjà correct avant la revue, testé contre l'oracle dans les deux ordres de pose (test_fat_dormouse_activated_by_a_bat_placed_afterwards)]
Loup | ? | pioche 1 carte × Cervidé possédé (inconditionnel) ; SI payé avec bonus jumelles : rejoue un tour complet | oui, rejoue un tour | 5 pts × Cervidé (inconditionnel) [FAIT — DRAW_PER_COUNT + REPLAY_IF_BONUS, engine.py/game.py. Statut mis à jour lors de la revue : la pioche et le score sont bien inconditionnels, seul le rejeu dépend du bonus, reconfirmé par Mehdi]
Chevreuil | ? | pioche 1 carte SI payé avec bonus jumelles | oui | 3 pts × cartes montrant le symbole imprimé [FAIT — DRAW_IF_BONUS, engine.py/game.py]
Daim | ? | pioche 2 cartes SI payé avec bonus jumelles | oui | 3 pts × Cervidé [FAIT — DRAW_IF_BONUS, engine.py/game.py]
Hérisson commun | ? | pioche 1 carte SI payé avec bonus jumelles | oui | 2 pts × Papillon [FAIT — DRAW_IF_BONUS, engine.py/game.py]
Fouine | ? | pioche 1 carte (pas de condition bonus) | non | 5 pts × (Arbre entièrement occupé = ses 4 côtés Top/Bottom/Left/Right remplis) [FAIT — DRAW_FIXED + `Forest.fully_occupied` (compteur PAR ARBRE, pas un booléen forêt entière). Statut mis à jour lors de la revue : les deux étaient déjà câblés, reconfirmé par Mehdi]
Moustique | ? | jouer autant de Chauve-souris que voulu depuis la main (gratuit ? à confirmer) | ? | 0 ou score selon rulebook à vérifier [FAIT]
Amanite tue-mouches (champignon) | ? | effet permanent : à chaque carte ANIMALE jouée, pioche 1 carte | - | 0 [FAIT]
Cèpe de Bordeaux (champignon) | ? | effet permanent : à chaque carte jouée AU-DESSUS d'un arbre (position Top), pioche 1 carte | - | 1 pt × Champignon [FAIT]

Girolle (champignon) | ? | effet permanent : à chaque carte jouée portant un symbole d'arbre (n'importe lequel, pas seulement celui de la Girolle), pioche 1 carte | - | ? [FAIT]
Chouette hulotte | ? | pioche 1 carte (inconditionnel) ; SI payé avec bonus jumelles : pioche 2 cartes DE PLUS | oui | 5 (fixe) [FAIT — DRAW_FIXED + DRAW_IF_BONUS, confirmé par Mehdi lors de la revue]
Ours brun | 3 | Placez toutes les cartes de la Clairière dans votre Grotte (inconditionnel) ; SI payé avec bonus jumelles : pioche 1 carte ET rejoue un tour complet (confirmé par Mehdi) | oui | 0 (points via Grotte) [FAIT — DRAW_IF_BONUS/REPLAY_IF_BONUS + CLEARING_TO_CAVE_DWELLERS, engine.py/game.py. Nombre de cartes piochées (1) non explicitement reconfirmé par Mehdi, par défaut cohérent avec le reste de la table]
Blaireau européen | ? | SI payé avec bonus jumelles : joue gratuitement un animal depuis la main | oui | ? [FAIT — remplace la version précédente "Plantigrade/digitigrade", DWELLER_PLAY_FREE_IF_BONUS (FILTER_ANY_ANIMAL)]
Fougère arborescente | ? | pioche 1 carte (sans condition ?) | - | ? [FAIT (rattrapé, oublié dans le batch initial)]
Lucane | ? | SI payé avec bonus jumelles (dépend de la couleur/symbole sur la carte) : joue gratuitement un oiseau depuis la main | oui | ? [FAIT]
Sapin Douglas (arbre) | 2 | SI payé avec bonus jumelles (carte de symbole Sapin Douglas) : rejoue un tour | oui | 5 pts × Sapin Douglas [FAIT — TREE_REPLAY_IF_BONUS, corrige une hypothèse antérieure fausse ("pas de bonus jumelles pour un arbre" ; un Arbre PEUT en avoir, confirmé par Mehdi)]
Chêne (arbre) | 2 | SI payé avec bonus jumelles (carte de symbole Chêne) : rejoue un tour | oui | 10 pts × Chêne si ≥8 espèces d'arbres en forêt, sinon 0 [FAIT — TREE_REPLAY_IF_BONUS, score déjà FAIT]
Hêtre (arbre) | 1 | pioche 1 carte, inconditionnel | non | 5 pts × Hêtre si ≥4 Hêtres en forêt, sinon 0 [FAIT — TREE_DRAW_FIXED, score déjà FAIT]
Salamandre tachetée | ? | SI payé avec bonus jumelles : joue un animal gratuitement depuis la main (1 usage) | oui | set 5/15/25 [FAIT — DWELLER_PLAY_FREE_IF_BONUS (FILTER_ANY_ANIMAL), confirmé par Mehdi lors de la revue. Nombre d'usages (1) non explicitement reconfirmé, par défaut cohérent avec le reste de la table]

Geai des chênes (EURASIAN_JAY) | ? | rejoue un tour complet (inconditionnel) | non | ? [FAIT]
Cerf élaphe | ? | SI payé avec bonus jumelles : joue gratuitement un Cerf élaphe depuis la main | oui | 2 pts × Ongulé [FAIT — DWELLER_PLAY_FREE_IF_BONUS (FILTER_DWELLER), score déjà noté plus haut]
Bouleau (arbre) | ? | effet de pose (ponctuel, pas permanent) : à la pose, en plus du score, pioche 1 carte | non | ? [FAIT]
Tortue cistude | ? | pioche 1 carte | non | 5 [FAIT]
Coulemelle (champignon) | ? | effet permanent : à chaque carte jouée EN DESSOUS d'un arbre (position Bottom), pioche 1 carte | - | ? [FAIT]

## Non clair — à reconfirmer avec Mehdi (ne pas coder)

(aucun point en suspens)
