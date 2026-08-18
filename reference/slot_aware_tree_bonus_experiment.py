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
`need_scale` habitants sans slot donne la prime pleine (par defaut 4 :
un arbre qui manque completement, puisqu'un arbre ouvre 4 emplacements).

Confronte a B (`search.greedy_action` par defaut = heuristique forte de
Clairiere + urgence Clairiere, deja les reglages par defaut du module) --
**pas MCTS** : B est aujourd'hui le bot le plus fort du depot (voir
reference/MODELS.md, "En pratique, B ... est aujourd'hui le bot le plus
fort du depot, plus simple et bien plus rapide que MCTS"), donc c'est la
seule reference pertinente pour juger si cette heuristique vaut le coup.

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


def greedy_action_slots(game, rng, epsilon=0.0, candidates=None, tree_bonus=6,
                         need_scale=4.0, tree_combo_bonus=None,
                         clearing_urgency=CLEARING_URGENCY_BONUS):
    """Variante de `search.greedy_action` : prime de pose d'arbre pilotee
    par le deficit reel de slots libres plutot que par 1/(1+n_trees).
    Tout le reste (Clairiere, paiement Grotte, repli pioche) est identique
    au bot B -- seul le terme de pose d'arbre change, pour isoler l'effet
    de cette heuristique precise."""
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
        if gain > best_gain:
            best, best_gain = a, gain

    if (clearing_urgency and ("draw",) in actions
            and clearing_urgency > best_gain
            and any(G._is_strong_card(c) for c in game.clearing)):
        return ("draw",)

    if best_gain <= 0 and len(player.hand) <= 7:
        return S._fallback_action(game)
    return best


def play_one(seed, slot_seat, need_scale):
    game = Game(n_players=2, seed=seed)
    rng = random.Random(seed)
    turns = 0
    while not game.over and turns < 400:
        r = random.Random(seed * 1000 + turns)
        if game.current == slot_seat:
            action = greedy_action_slots(game, r, need_scale=need_scale)
        else:
            action = greedy_action(game, r)
        game.apply(action)
        turns += 1
    return game.scores()


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    need_scale = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0

    diffs, scores_slot, scores_b, fill_rates = [], [], [], []
    t0 = time.perf_counter()
    for gi in range(n_games):
        seed = 80000 + gi
        seat = gi % 2
        sc = play_one(seed, seat, need_scale)
        s_score, b_score = sc[seat], sc[1 - seat]
        diffs.append(s_score - b_score)
        scores_slot.append(s_score)
        scores_b.append(b_score)

    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    print(f"slot-aware (need_scale={need_scale}) vs B (greedy defaut), "
          f"{n_games} parties, sieges alternes :")
    print(f"  victoires slot-aware : {wins}/{n_games}  (nul : {n_games - wins - losses})")
    print(f"  score moyen slot-aware : {statistics.mean(scores_slot):.1f} | "
          f"B : {statistics.mean(scores_b):.1f}")
    print(f"  ecart moyen : {statistics.mean(diffs):+.1f}  "
          f"(SE {statistics.stdev(diffs) / n_games ** 0.5:.1f})" if n_games > 1 else "")
    print(f"  {(time.perf_counter() - t0) / n_games * 1000:.0f} ms/partie")


if __name__ == "__main__":
    main()
