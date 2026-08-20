"""Fait jouer MCTS PUR (leaf_eval=None, rollout reel jusqu'a rollout_depth,
pas de modele appris) contre E (greedy_action + tiebreak GBM) sous
plusieurs reglages d'iterations/profondeur, pour visualiser directement
pourquoi MCTS pur ne s'en sort pas (voir reference/MODELS.md, section
"Deux pistes force brute testees contre B" -- ce script reproduit la
meme experience mais contre E, le bot le plus fort actuel, avec un
balayage de reglages au lieu d'un seul point).

Usage : python reference/mcts_pure_vs_e_sweep.py [n_parties_par_config]
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

CONFIGS = [
    # (iterations, rollout_depth)
    (100, 40),
    (300, 40),
    (100, 90),
    (300, 90),
    (600, 40),
]


def play_one(seed, seat_of_mcts, iterations, depth):
    game = Game(n_players=2, seed=seed)
    tiebreak = VP.make_pairwise_gbm_tiebreak()
    bot = MCTS(observer=seat_of_mcts, iterations=iterations, seed=seed * 31 + seat_of_mcts,
               rollout_depth=depth, leaf_eval=None)
    turns = 0
    while not game.over and turns < 600:
        seat = game.current
        if seat == seat_of_mcts:
            action = bot.choose(game)
        else:
            action = greedy_action(game, None, tiebreak=tiebreak)
        game.apply(action)
        if seat == seat_of_mcts:
            bot.advance(action)
        else:
            bot.advance(action)
        turns += 1
    sc = game.scores()
    return sc[seat_of_mcts], sc[1 - seat_of_mcts], turns


def run_config(iterations, depth, n):
    diffs, turns_list = [], []
    t0 = time.time()
    for i in range(n):
        seed = 90000 + i
        seat_of_mcts = i % 2
        s_mcts, s_e, turns = play_one(seed, seat_of_mcts, iterations, depth)
        diffs.append(s_mcts - s_e)
        turns_list.append(turns)
        print(f"    partie {i + 1}/{n}: MCTS {s_mcts} vs E {s_e} "
              f"(diff {s_mcts - s_e:+.0f}, {turns} tours)", flush=True)
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    draws = n - wins - losses
    mean = statistics.mean(diffs)
    se = statistics.stdev(diffs) / (n ** 0.5) if n > 1 else 0.0
    dt = (time.time() - t0) / n
    print(f"  => MCTS pur({iterations}it, prof {depth}) vs E : "
          f"{wins}/{n} victoires ({losses} defaites, {draws} nuls) | "
          f"ecart moyen {mean:+.1f} (SE {se:.1f}) | "
          f"{statistics.mean(turns_list):.0f} tours/partie | {dt:.1f} s/partie\n", flush=True)
    return wins, n, mean, se


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    results = []
    for iterations, depth in CONFIGS:
        print(f"=== MCTS pur : {iterations} iterations, rollout_depth={depth} ({n} parties) ===", flush=True)
        results.append((iterations, depth, run_config(iterations, depth, n)))

    print("\n=== Recapitulatif ===")
    print(f"{'iterations':>10} {'depth':>6} {'victoires':>10} {'ecart moyen':>12} {'SE':>6}")
    for iterations, depth, (wins, total, mean, se) in results:
        print(f"{iterations:>10} {depth:>6} {wins:>6}/{total:<4} {mean:>+12.1f} {se:>6.1f}")


if __name__ == "__main__":
    main()
