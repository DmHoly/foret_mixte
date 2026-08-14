"""Statistiques agrégées sur N parties jouées par le bot MCTS (config
gagnante : rollout court + modèle contrastif) contre greedy.

Pour chaque carte (arbre ou habitant), compte :
  - present_in : nombre de parties où le bot en a posé au moins un
    exemplaire (fréquence d'apparition, 0-N)
  - total : nombre total d'exemplaires posés, toutes parties confondues
  - avg_when_present : total / present_in (combien il en empile QUAND il
    en joue)

Usage : python run_stats_hybrid.py [n_games] [iterations]
"""
import sys

sys.path.insert(0, "reference")

import random
import statistics
import time
from collections import Counter

import value_policy as VP
from engine import DWELLERS, TREES
from game import Game
from search import MCTS, greedy_action

TREE_NAMES = [t.name for t in TREES]
DWELLER_NAMES = [d.name for d in DWELLERS]


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    tree_total = Counter()
    tree_present = Counter()
    dweller_total = Counter()
    dweller_present = Counter()
    scores, diffs = [], []

    t0 = time.perf_counter()
    for gi in range(n_games):
        seed = 9000 + gi
        me = gi % 2
        game = Game(n_players=2, seed=seed)
        rng = random.Random(seed)
        leaf_eval = VP.make_pairwise_hybrid_leaf_eval(short_rollout_depth=10, seed=seed)
        bot = MCTS(observer=me, iterations=iterations, seed=seed, leaf_eval=leaf_eval)

        turns = 0
        while not game.over and turns < 300:
            if game.current == me:
                action = bot.choose(game)
            else:
                action = greedy_action(game, rng)
            game.apply(action)
            bot.advance(action)
            turns += 1

        forest = game.players[me].forest
        for i in range(len(TREE_NAMES)):
            c = forest.tree_count[i]
            if c:
                tree_total[TREE_NAMES[i]] += c
                tree_present[TREE_NAMES[i]] += 1
        for i in range(len(DWELLER_NAMES)):
            c = forest.dweller_count[i]
            if c:
                dweller_total[DWELLER_NAMES[i]] += c
                dweller_present[DWELLER_NAMES[i]] += 1

        sc = game.scores()
        scores.append(sc[me])
        diffs.append(sc[me] - sc[1 - me])
        print(f"partie {gi + 1}/{n_games} (seed {seed}) : score {sc[me]} "
              f"(adversaire {sc[1 - me]}) | {time.perf_counter() - t0:.0f}s écoulées",
              flush=True)

    print()
    print(f"Score moyen : {statistics.mean(scores):.1f} | "
          f"écart moyen vs greedy : {statistics.mean(diffs):+.1f} | "
          f"victoires : {sum(1 for d in diffs if d > 0)}/{n_games}")

    print()
    print("=== Arbres (fréquence sur", n_games, "parties | moyenne posée quand présent) ===")
    for name, present in tree_present.most_common():
        avg = tree_total[name] / present
        print(f"  {name:20s} présent {present:2d}/{n_games} | moyenne {avg:.1f} exemplaires")

    print()
    print("=== Habitants (fréquence | moyenne posée quand présent) ===")
    for name, present in dweller_present.most_common():
        avg = dweller_total[name] / present
        print(f"  {name:28s} présent {present:2d}/{n_games} | moyenne {avg:.1f} exemplaires")


if __name__ == "__main__":
    main()
