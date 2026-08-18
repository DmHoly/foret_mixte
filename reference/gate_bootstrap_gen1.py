"""Gating de reference/pairwise_model_bootstrap_gen1.joblib (généré par
gen_pairwise_dataset_bootstrap.py) contre B (Greedy + Clairière forte +
urgence, le meilleur bot du dépôt à ce jour) -- ne remplacer le modèle
vivant que si ce candidat bat B en tête-à-tête, jamais sur la seule base
d'une métrique offline (voir reference/MODELS.md, motif "mieux hors ligne
= pire en jeu" observé sur 6 tentatives précédentes).

Usage : python reference/gate_bootstrap_gen1.py [n_parties]
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

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / "pairwise_model_bootstrap_gen1.joblib"


def play_one(seed, seat_of_d):
    game = Game(n_players=2, seed=seed)
    bot = MCTS(observer=seat_of_d, iterations=150, seed=seed * 31 + seat_of_d,
               rollout_depth=40,
               leaf_eval=VP.make_pairwise_hybrid_leaf_eval(
                   model_path=CANDIDATE, seed=seed * 31 + seat_of_d))
    turns = 0
    while not game.over and turns < 600:
        seat = game.current
        action = bot.choose(game) if seat == seat_of_d else greedy_action(game, None)
        game.apply(action)
        bot.advance(action)  # garder l'arbre du candidat synchro meme sur le coup adverse
        turns += 1
    return game.scores()[seat_of_d], game.scores()[1 - seat_of_d]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    diffs = []
    t0 = time.time()
    for i in range(n):
        seed = 70000 + i
        seat_of_d = i % 2
        sd, sb = play_one(seed, seat_of_d)
        diffs.append(sd - sb)
        print(f"  partie {i+1}/{n} : D(candidat)={sd} B={sb} diff={sd-sb:+.0f}", flush=True)

    wins = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)
    se = statistics.stdev(diffs) / (len(diffs) ** 0.5) if n > 1 else 0.0
    print()
    print(f"D(candidat bootstrap gen1) vs B : {wins}/{n} ({ties} nuls), "
          f"ecart moyen {statistics.mean(diffs):+.1f} (SE {se:.1f}), "
          f"mediane {statistics.median(diffs):+.1f} -- {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
