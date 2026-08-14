"""Confronte `strength_action` (delta exact + correction de force de carte
mesurée, paiement par carte la plus faible) à `greedy_action` (delta exact
seul, paiement par coût facial), en tête-à-tête, sièges alternés.

Usage : python bench_strength.py [n_games] [w]
"""
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from game import Game
from search import greedy_action
from strength_policy import strength_action


def play_one(seed, strength_seat, w):
    game = Game(n_players=2, seed=seed)
    rng = random.Random(seed)
    turns = 0
    while not game.over and turns < 400:
        if game.current == strength_seat:
            action, payment = strength_action(game, rng, w=w)
            game.apply(action, payment=payment)
        else:
            action = greedy_action(game, rng)
            game.apply(action)
        turns += 1
    return game.scores()


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    w = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    diffs, scores_s, scores_g = [], [], []
    t0 = time.perf_counter()
    for gi in range(n_games):
        seed = 80000 + gi
        seat = gi % 2
        sc = play_one(seed, seat, w)
        s_score, g_score = sc[seat], sc[1 - seat]
        diffs.append(s_score - g_score)
        scores_s.append(s_score)
        scores_g.append(g_score)

    wins = sum(1 for d in diffs if d > 0)
    print(f"strength(w={w}) vs greedy, {n_games} parties, sièges alternés :")
    print(f"  victoires strength : {wins}/{n_games}")
    print(f"  score moyen strength : {statistics.mean(scores_s):.1f} | "
          f"greedy : {statistics.mean(scores_g):.1f}")
    print(f"  écart moyen : {statistics.mean(diffs):+.1f}")
    print(f"  {(time.perf_counter() - t0) / n_games * 1000:.0f} ms/partie")


if __name__ == "__main__":
    main()
