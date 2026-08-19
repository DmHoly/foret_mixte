"""Prototype : remplacer la prime de pose d'arbre decroissante par arbre
(`tree_bonus / (1 + n_trees)`, voir `search.greedy_action`) par une prime
pilotee par le vrai deficit de slots libres, comme propose par Mehdi
(session du 18/08) :

    "si tu as une capacite de pose de 5 habitants dans les 3 prochains
    tours et 5 slots de libre, poser un arbre n'est pas necessaire ; si tu
    n'as que 2 slots dispo, il va falloir en liberer."

Implementation : le besoin a venir n'est pas observable (main adverse et
pioche futures inconnues) -- on utilise le nombre d'habitants actuellement
en main comme proxy legal du besoin de placement, et le nombre de slots
vides (4 x n_trees - slots deja occupes) comme capacite actuelle. La prime
de pose ne s'active que si ce besoin depasse la capacite ; sinon elle tombe
a 0 (pas de floor artificiel : c'est le coeur de l'idee a tester -- si le
deficit est nul, planter un arbre ne doit rapporter rien de plus que son
delta_tree reel, quasi nul).

`need_scale` calibre la vitesse de montee en puissance : un deficit de
`need_scale` habitants sans slot donne la prime pleine (par defaut 3.0,
meilleur reglage trouve lors du premier passage de cette experience, voir
resultats plus bas -- teste 1.0 a 8.0, cf. commit precedent).

IMPORTANT (correction du 18/08 soir, feedback de Mehdi) : le premier
passage de cette experience comparait a tort contre B (`greedy_action`
sans tiebreak). Ce n'est plus le bot le plus fort du depot depuis
l'ajout du comparateur pairwise Gradient Boosting (`tiebreak=`, voir
`search.greedy_action` et `reference/value_policy.make_pairwise_gbm_tiebreak`)
qui departage en un seul appel batche les candidats a moins de
`tiebreak_margin` points du meilleur delta exact -- E (greedy + ce
tiebreak) bat B 74-76/100, +50 a +66 pts d'ecart (~4.7 ecarts-types, voir
reference/MODELS.md, "E est, a ce jour, le bot le plus fort du depot").
Cette version compare donc l'heuristique de pose d'arbre slot-aware,
AVEC le meme tiebreak GBM que E des deux cotes (seule la formule de prime
d'arbre change, tout le reste -- Clairiere, tiebreak -- reste identique),
contre E tel quel.

Usage :
    python reference/slot_aware_tree_bonus_experiment.py [n_games] [need_scale]
"""
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import game as G
from game import Game
import search as S
from search import greedy_action, CLEARING_URGENCY_BONUS
import value_policy as VP


def greedy_action_slots(game, rng, epsilon=0.0, candidates=None, tree_bonus=6,
                         need_scale=3.0, tree_combo_bonus=None,
                         clearing_urgency=CLEARING_URGENCY_BONUS,
                         tiebreak=None, tiebreak_margin=3.0):
    """Variante de `search.greedy_action` : prime de pose d'arbre pilotee
    par le deficit reel de slots libres plutot que par 1/(1+n_trees).
    Tout le reste (Clairiere, tiebreak GBM, paiement Grotte, repli pioche)
    est identique a E -- seul le terme de pose d'arbre change, pour isoler
    l'effet de cette heuristique precise."""
    actions = game.legal_actions()
    if len(actions) == 1:
        return actions[0]
    if epsilon and rng.random() < epsilon:
        return rng.choice(actions)

    plays = [a for a in actions if a[0] not in ("draw", "skip_effect")]
    if not plays:
        return actions[0]
    if candidates and len(plays) > candidates:
        plays = rng.sample(plays, candidates)

    player = game.players[game.current]
    forest = player.forest
    base = forest.score()

    free_slots = 4 * forest.n_trees - sum(forest.occupied_positions)
    hand_dwellers = sum(1 for c in player.hand if c[0] == G.DWELLER)
    deficit = max(0, hand_dwellers - free_slots)
    slot_bonus = tree_bonus * min(1.0, deficit / need_scale)

    best, best_gain = None, -1e9
    gains = {}
    for a in plays:
        if a[0] == "tree":
            gain = forest.delta_tree(a[1]) + slot_bonus
            if tree_combo_bonus:
                gain += tree_combo_bonus.get(a[1], 0.0)
        elif a[0] == "cave_discard":
            gain = a[1]
        else:
            _, did, tree_idx, pos = a
            forest.add_dweller(tree_idx, pos, did)
            gain = forest.score() - base
            forest.undo_dweller(tree_idx, pos, did)
        gains[a] = gain
        if gain > best_gain:
            best, best_gain = a, gain

    if tiebreak is not None:
        near = [a for a in plays if a != best and gains[a] >= best_gain - tiebreak_margin]
        if near:
            observer = game.current
            best_state = game.clone()
            best_state.apply(best)
            near_states = []
            for a in near:
                state = game.clone()
                state.apply(a)
                near_states.append(state)
            scores = tiebreak(near_states, best_state, observer)
            top = max(range(len(scores)), key=lambda i: scores[i])
            if scores[top] > 0:
                best = near[top]

    if (clearing_urgency and ("draw",) in actions
            and clearing_urgency > best_gain
            and any(G._is_strong_card(c) for c in game.clearing)):
        return ("draw",)

    if best_gain <= 0 and len(player.hand) <= 7:
        return S._fallback_action(game)
    return best


def play_one(seed, seat_of_slots, need_scale, tiebreak):
    game = Game(n_players=2, seed=seed)
    turns = 0
    while not game.over and turns < 600:
        seat = game.current
        if seat == seat_of_slots:
            action = greedy_action_slots(game, None, need_scale=need_scale, tiebreak=tiebreak)
        else:
            action = greedy_action(game, None, tiebreak=tiebreak)
        game.apply(action)
        turns += 1
    return game.scores()[seat_of_slots], game.scores()[1 - seat_of_slots]


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    need_scale = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    tiebreak = VP.make_pairwise_gbm_tiebreak()

    diffs, scores_slot, scores_e = [], [], []
    t0 = time.time()
    for i in range(n_games):
        seed = 80000 + i
        seat_of_slots = i % 2
        s_slot, s_e = play_one(seed, seat_of_slots, need_scale, tiebreak)
        diffs.append(s_slot - s_e)
        scores_slot.append(s_slot)
        scores_e.append(s_e)
        print(f"  partie {i+1}/{n_games} : slot-aware={s_slot} E={s_e} "
              f"diff={s_slot - s_e:+.0f}", flush=True)

    wins = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)
    se_ = statistics.stdev(diffs) / (len(diffs) ** 0.5) if n_games > 1 else 0.0
    print()
    print(f"slot-aware (need_scale={need_scale}) + tiebreak GBM vs E "
          f"(greedy + tiebreak GBM, defaut), {n_games} parties, sieges alternes :")
    print(f"  victoires slot-aware : {wins}/{n_games}  ({ties} nuls)")
    print(f"  score moyen slot-aware : {statistics.mean(scores_slot):.1f} | "
          f"E : {statistics.mean(scores_e):.1f}")
    print(f"  ecart moyen : {statistics.mean(diffs):+.1f} (SE {se_:.1f}), "
          f"mediane : {statistics.median(diffs):+.1f}")
    print(f"  {time.time()-t0:.0f}s total, {(time.time()-t0)/n_games:.2f}s/partie")


if __name__ == "__main__":
    main()
