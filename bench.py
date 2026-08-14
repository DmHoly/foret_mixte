"""
Mesures reproductibles : scoring, simulation, confrontation des politiques.

    python bench.py scoring
    python bench.py policies
    python bench.py mcts [iterations] [profondeur] [parties]
"""

import random
import statistics
import sys
import time

import engine as E
import scoring_ref as R
from game import Game
from search import MCTS, beam_policy, greedy_action, greedy_policy, play_game


def _timeit(fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def _end_game_forest(seed=1, n_trees=25, n_dwellers=400):
    rng = random.Random(seed)
    fast, ref = E.Forest(), R.RefForest()
    for _ in range(n_trees):
        name = rng.choice(E.TREE_NAME)
        fast.add_tree(E.TREE_ID[name])
        ref.add_tree(name)
    placed = 0
    for _ in range(n_dwellers):
        did = rng.randrange(E.N_DWELLERS)
        options = list(fast.legal_positions(did))
        if not options:
            continue
        ti, pos = rng.choice(options)
        fast.add_dweller(ti, pos, did)
        ref.add_dweller(ti, E.POSITIONS[pos], E.DWELLER_NAME[did])
        placed += 1
    return fast, ref, placed


def bench_scoring():
    fast, ref, placed = _end_game_forest()
    print(f"Forêt de fin de partie : 25 arbres, {placed} habitants")
    print(f"  score moteur : {fast.score()}")
    print(f"  score oracle : {R.score_forest(ref)}")
    print()
    print(f"  score()         {_timeit(fast.score, 20000) * 1e6:8.2f} us")
    print(f"  copy()          {_timeit(fast.copy, 20000) * 1e6:8.2f} us")
    print(f"  oracle          {_timeit(lambda: R.score_forest(ref), 200) * 1e6:8.2f} us")


def bench_policies(n=40):
    for name, policy, runs in (
        ("greedy", greedy_policy, n),
        ("beam(4,2)", lambda g, r: beam_policy(g, r, 4, 2), max(10, n // 4)),
    ):
        t0 = time.perf_counter()
        scores, turns = [], []
        for s in range(runs):
            sc, t = play_game([policy] * 2, n_players=2, seed=1000 + s)
            scores += sc
            turns.append(t)
        dt = (time.perf_counter() - t0) / runs
        print(f"{name:12s} {dt * 1000:7.1f} ms/partie | score moyen "
              f"{statistics.mean(scores):6.1f} | max {max(scores)} | "
              f"{statistics.mean(turns):.0f} tours")


def bench_mcts(iterations=400, depth=40, games=8, leaf_eval=None):
    diffs, decisions = [], []
    t0 = time.perf_counter()
    for s in range(games):
        me = s % 2  # on alterne le siège pour neutraliser l'avantage du premier
        game = Game(n_players=2, seed=500 + s)
        rng = random.Random(500 + s)
        bot = MCTS(observer=me, iterations=iterations, seed=500 + s,
                   rollout_depth=depth, leaf_eval=leaf_eval)
        turns = 0
        times = []
        while not game.over and turns < 600:
            if game.current == me:
                t = time.perf_counter()
                action = bot.choose(game)
                times.append(time.perf_counter() - t)
            else:
                action = greedy_action(game, rng)
            game.apply(action)
            bot.advance(action)
            turns += 1
        sc = game.scores()
        diffs.append(sc[me] - sc[1 - me])
        decisions.append(statistics.mean(times) if times else 0)
    wins = sum(1 for d in diffs if d > 0)
    print(f"MCTS({iterations} it, profondeur {depth}) contre greedy : "
          f"{wins}/{games} victoires | écart moyen {statistics.mean(diffs):+.1f} "
          f"| {statistics.mean(decisions) * 1000:.0f} ms/décision "
          f"| {(time.perf_counter() - t0) / games:.1f} s/partie")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "scoring"
    if what == "scoring":
        bench_scoring()
    elif what == "policies":
        bench_policies()
    elif what == "mcts":
        args = [int(x) for x in sys.argv[2:]]
        bench_mcts(*args) if args else bench_mcts()
    elif what == "mcts_value":
        # MCTS avec évaluation de feuille par le modèle appris (reference/),
        # au lieu du rollout tronqué aléatoire. args : iterations games
        sys.path.insert(0, "reference")
        import value_policy as VP
        args = [int(x) for x in sys.argv[2:]]
        iterations = args[0] if len(args) > 0 else 300
        games = args[1] if len(args) > 1 else 8
        bench_mcts(iterations=iterations, games=games,
                   leaf_eval=VP.make_hybrid_leaf_eval(seed=1))
    elif what == "mcts_pairwise":
        # MCTS avec la fonction de valeur linéaire contrastive (rapide, pas
        # de coût d'inférence sklearn). args : iterations games
        sys.path.insert(0, "reference")
        import value_policy as VP
        args = [int(x) for x in sys.argv[2:]]
        iterations = args[0] if len(args) > 0 else 300
        games = args[1] if len(args) > 1 else 8
        bench_mcts(iterations=iterations, games=games,
                   leaf_eval=VP.make_pairwise_leaf_eval())
    elif what == "mcts_pairwise_hybrid":
        # rollout court (coups réels) + correction par le modèle contrastif.
        # args : iterations games short_rollout_depth
        sys.path.insert(0, "reference")
        import value_policy as VP
        args = [int(x) for x in sys.argv[2:]]
        iterations = args[0] if len(args) > 0 else 300
        games = args[1] if len(args) > 1 else 8
        depth = args[2] if len(args) > 2 else 10
        bench_mcts(iterations=iterations, games=games,
                   leaf_eval=VP.make_pairwise_hybrid_leaf_eval(
                       short_rollout_depth=depth, seed=1))
    else:
        print(__doc__)
