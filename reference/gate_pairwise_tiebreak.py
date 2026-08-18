"""Gate le comparateur pairwise Gradient Boosting branché dans
`greedy_action` (voir `search.greedy_action(..., tiebreak=...)`,
`reference/value_policy.make_pairwise_gbm_tiebreak`) contre B (Greedy +
Clairière forte + urgence, sans tiebreak -- le bot le plus fort du dépôt
avant cette tentative). Suite de reference/diagnose_nonlinear_capacity.py
(le Gradient Boosting bat le linéaire de ~10 points de précision de signe
sur les paires serrées, en validation croisée) -- ici on teste si ça se
traduit en victoires réelles, jamais promu sur la seule base du diagnostic
offline (voir reference/MODELS.md pour l'historique de cette règle).

Usage : python reference/gate_pairwise_tiebreak.py [n_parties]
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from game import Game
from search import greedy_action
import value_policy as VP


def play_one(seed, seat_of_e, tiebreak):
    game = Game(n_players=2, seed=seed)
    turns = 0
    while not game.over and turns < 600:
        seat = game.current
        if seat == seat_of_e:
            action = greedy_action(game, None, tiebreak=tiebreak)
        else:
            action = greedy_action(game, None)
        game.apply(action)
        turns += 1
    return game.scores()[seat_of_e], game.scores()[1 - seat_of_e]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    tiebreak = VP.make_pairwise_gbm_tiebreak()

    diffs = []
    t0 = time.time()
    for i in range(n):
        seed = 80000 + i
        seat_of_e = i % 2
        se, sb = play_one(seed, seat_of_e, tiebreak)
        diffs.append(se - sb)
        print(f"  partie {i+1}/{n} : E(tiebreak)={se} B={sb} diff={se-sb:+.0f}", flush=True)

    wins = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)
    se_ = statistics.stdev(diffs) / (len(diffs) ** 0.5) if n > 1 else 0.0
    print()
    print(f"E (greedy + tiebreak GBM) vs B : {wins}/{n} ({ties} nuls), "
          f"ecart moyen {statistics.mean(diffs):+.1f} (SE {se_:.1f}), "
          f"mediane {statistics.median(diffs):+.1f} -- {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
