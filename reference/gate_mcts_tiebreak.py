"""Gate le tiebreak GBM branché dans MCTS.choose() (une seule fois par
decision reelle, pas dans le rollout -- voir search.MCTS docstring et
reference/MODELS.md, "Comparateur pairwise dans greedy_action") contre
deux adversaires :

  - D : MCTS(150it) + hybrid leaf_eval, SANS tiebreak (config actuelle,
    reference/bench_heuristics.py).
  - E : greedy_action + tiebreak, SANS MCTS (le meilleur bot connu avant
    cette tentative, voir reference/gate_pairwise_tiebreak.py).

Usage : python reference/gate_mcts_tiebreak.py [n_parties] [vs]
  vs = "D" (defaut) ou "E"
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from game import Game
from search import MCTS, greedy_action
import value_policy as VP


def make_mcts_tiebreak(seat, seed, tiebreak):
    return MCTS(observer=seat, iterations=150, seed=seed, rollout_depth=40,
                leaf_eval=VP.make_pairwise_hybrid_leaf_eval(short_rollout_depth=10, seed=seed),
                tiebreak=tiebreak)


def make_mcts_plain(seat, seed):
    return MCTS(observer=seat, iterations=150, seed=seed, rollout_depth=40,
                leaf_eval=VP.make_pairwise_hybrid_leaf_eval(short_rollout_depth=10, seed=seed))


def play_vs_D(seed, seat_of_f, tiebreak):
    game = Game(n_players=2, seed=seed)
    bots = {
        seat_of_f: make_mcts_tiebreak(seat_of_f, seed * 31 + seat_of_f, tiebreak),
        1 - seat_of_f: make_mcts_plain(1 - seat_of_f, seed * 31 + (1 - seat_of_f)),
    }
    turns = 0
    while not game.over and turns < 600:
        seat = game.current
        action = bots[seat].choose(game)
        game.apply(action)
        for b in bots.values():
            b.advance(action)
        turns += 1
    return game.scores()[seat_of_f], game.scores()[1 - seat_of_f]


def play_vs_E(seed, seat_of_f, tiebreak):
    game = Game(n_players=2, seed=seed)
    bot = make_mcts_tiebreak(seat_of_f, seed * 31 + seat_of_f, tiebreak)
    turns = 0
    while not game.over and turns < 600:
        seat = game.current
        if seat == seat_of_f:
            action = bot.choose(game)
        else:
            action = greedy_action(game, None, tiebreak=tiebreak)
        game.apply(action)
        bot.advance(action)
        turns += 1
    return game.scores()[seat_of_f], game.scores()[1 - seat_of_f]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    vs = sys.argv[2] if len(sys.argv) > 2 else "D"
    tiebreak = VP.make_pairwise_gbm_tiebreak()
    play_one = play_vs_D if vs == "D" else play_vs_E

    diffs = []
    t0 = time.time()
    for i in range(n):
        seed = 90000 + i
        seat_of_f = i % 2
        sf, so = play_one(seed, seat_of_f, tiebreak)
        diffs.append(sf - so)
        print(f"  partie {i+1}/{n} : F(mcts+tiebreak)={sf} {vs}={so} diff={sf-so:+.0f}", flush=True)

    wins = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)
    se = statistics.stdev(diffs) / (len(diffs) ** 0.5) if n > 1 else 0.0
    print()
    print(f"F (MCTS + tiebreak GBM) vs {vs} : {wins}/{n} ({ties} nuls), "
          f"ecart moyen {statistics.mean(diffs):+.1f} (SE {se:.1f}), "
          f"mediane {statistics.median(diffs):+.1f} -- {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
