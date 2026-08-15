# Référence des effets/bonus de cartes — Forêt Mixte

Format : Nom | coût | effet à la pose | bonus jumelles (paiement couleur) | score final
Statut moteur : [FAIT] déjà implémenté correctement | [MANQUE] pas implémenté | [DOUTE] à vérifier

## Confirmé par Mehdi (photos + description directe)

Renard roux | 1 | pioche 1 carte × Lièvre d'Europe possédé | - | 2 pts × Lièvre d'Europe [MANQUE l'effet pioche, score déjà FAIT]
Lièvre d'Europe | ? | - | 1 par Lièvre d'Europe (bonus jumelles) | N² où N=nombre de lièvres [FAIT le score et le partage de slot, MANQUE le bonus jumelles]
Taupe | ? | Jouez immédiatement autant de cartes que souhaité en payant leurs coûts | - | 0 [FAIT - chaîne d'actions payantes, arrêt automatique ou skip_effect]
Ours brun | ? | Placez toutes les cartes de la Clairière dans votre Grotte | - | 0 (les points viennent de la Grotte) [FAIT — clairière/grotte simulées (CLEARING_TO_CAVE_DWELLERS, game.py), voir aussi ligne 27]
Raton laveur | ? | Placez autant de cartes que voulu de la main directement sur la Grotte (1 pt chacune au score, `forest.cave`), et piochez le même nombre | - | ? [FAIT]
Blaireau européen | ? | Placez un spécimen Plantigrade/digitigrade gratuit depuis la main | 2 glands × (bonus) | ? [FAIT]
Sapin blanc (arbre) | 2 | SI payé avec une carte de symbole Sapin blanc (bonus jumelles) : joue gratuitement un animal depuis la main | oui | 2 pts × carte rattachée à ce Sapin blanc [FAIT en brique 2b, score déjà FAIT]
Xylocope violet | ? | - | - | 0 pt, +1 à l'arbre porteur (compte comme un arbre en plus de son espèce) [FAIT - déjà implémenté (bee_by_species), correctif documenté dans engine.py]
Loup | ? | pioche 1 carte × Cervidé possédé ; SI payé avec bonus jumelles : rejoue un tour complet APRÈS le scoring | oui, rejoue un tour | 5 pts × Cervidé [MANQUE l'effet pioche + le bonus rejoue, score déjà FAIT (D_WOLF? à vérifier nom exact)]
Chevreuil | ? | pioche 1 carte SI payé avec bonus jumelles | oui | 3 pts × cartes montrant le symbole imprimé [MANQUE l'effet bonus pioche, score déjà FAIT et corrigé]
Daim | ? | pioche 2 cartes SI payé avec bonus jumelles | oui | 3 pts × Cervidé [MANQUE l'effet bonus pioche, score déjà FAIT]
Hérisson commun | ? | pioche 1 carte SI payé avec bonus jumelles | oui | 2 pts × Papillon [MANQUE l'effet bonus pioche, score déjà FAIT]
Fouine | ? | pioche 1 carte (pas de condition bonus) | non | 5 pts × Arbre entièrement occupé [MANQUE l'effet pioche, score déjà FAIT]
Moustique | ? | jouer autant de Chauve-souris que voulu depuis la main (gratuit ? à confirmer) | ? | 0 ou score selon rulebook à vérifier [FAIT]
Amanite tue-mouches (champignon) | ? | effet permanent : à chaque carte ANIMALE jouée, pioche 1 carte | - | 0 [FAIT]
Cèpe de Bordeaux (champignon) | ? | effet permanent : à chaque carte jouée AU-DESSUS d'un arbre (position Top), pioche 1 carte | - | 1 pt × Champignon [FAIT]

Girolle (champignon) | ? | effet permanent : à chaque carte jouée portant un symbole d'arbre (n'importe lequel, pas seulement celui de la Girolle), pioche 1 carte | - | ? [FAIT]
Chouette hulotte | ? | pioche 1 carte ; SI payé avec bonus jumelles (dépend du symbole sur la carte, comme Chevreuil) : pioche 2 cartes de plus | oui | 5 [MANQUE l'effet+bonus, score déjà dans le fichier]
Ours brun | 3 | Placez toutes les cartes de la Clairière dans votre Grotte (inconditionnel) ; SI payé avec bonus jumelles : pioche 1 carte ET rejoue un tour complet (confirmé par Mehdi) | oui | 0 (points via Grotte) [FAIT — DRAW_IF_BONUS/REPLAY_IF_BONUS + CLEARING_TO_CAVE_DWELLERS, engine.py/game.py. Nombre de cartes piochées (1) non explicitement reconfirmé par Mehdi, par défaut cohérent avec le reste de la table]
Blaireau européen | ? | SI payé avec bonus jumelles : joue gratuitement un animal depuis la main | oui | ? [MANQUE, remplace la version précédente "Plantigrade/digitigrade"]
Fougère arborescente | ? | pioche 1 carte (sans condition ?) | - | ? [FAIT (rattrapé, oublié dans le batch initial)]
Lucane | ? | SI payé avec bonus jumelles (dépend de la couleur/symbole sur la carte) : joue gratuitement un oiseau depuis la main | oui | ? [FAIT]
Sapin Douglas (arbre) | 2 | SI payé avec bonus jumelles (carte de symbole Sapin Douglas) : rejoue un tour | oui | 5 pts × Sapin Douglas [FAIT — TREE_REPLAY_IF_BONUS, corrige une hypothèse antérieure fausse ("pas de bonus jumelles pour un arbre" ; un Arbre PEUT en avoir, confirmé par Mehdi)]
Chêne (arbre) | 2 | SI payé avec bonus jumelles (carte de symbole Chêne) : rejoue un tour | oui | 10 pts × Chêne si ≥8 espèces d'arbres en forêt, sinon 0 [FAIT — TREE_REPLAY_IF_BONUS, score déjà FAIT]
Hêtre (arbre) | 1 | pioche 1 carte, inconditionnel | non | 5 pts × Hêtre si ≥4 Hêtres en forêt, sinon 0 [FAIT — TREE_DRAW_FIXED, score déjà FAIT]
Salamandre tachetée | ? | SI payé avec bonus jumelles : pose une carte (gratuitement ?) | oui | set 5/15/25 [MANQUE l'effet, score déjà dans le fichier]

Geai des chênes (EURASIAN_JAY) | ? | rejoue un tour complet (inconditionnel) | non | ? [FAIT]
Cerf élaphe | ? | SI payé avec bonus jumelles : joue gratuitement un Cerf élaphe depuis la main | oui | 2 pts × Ongulé [MANQUE l'effet, score déjà noté plus haut]
Bouleau (arbre) | ? | effet de pose (ponctuel, pas permanent) : à la pose, en plus du score, pioche 1 carte | non | ? [FAIT]
Tortue cistude | ? | pioche 1 carte | non | 5 [FAIT]
Coulemelle (champignon) | ? | effet permanent : à chaque carte jouée EN DESSOUS d'un arbre (position Bottom), pioche 1 carte | - | ? [FAIT]

## Non clair — à reconfirmer avec Mehdi (ne pas coder)

(aucun point en suspens)
