"""Rendement marginal de chaque Fouine (BEECH_MARTEN) supplementaire, sur
des parties E vs E reellement jouees.

Question de Mehdi (18/08) : la Fouine (5 x nb_Fouines x nb_arbres_pleins)
n'a que 5 exemplaires physiques dans tout le deck -- vaut-il le coup d'en
accumuler plusieurs, ou rendement decroissant comme la plupart des cartes
a copies multiples ? Bucketise les forets par nombre de Fouines detenues
en fin de partie et mesure le nombre d'arbres pleins (fully_occupied) et
les points Fouine dans chaque bucket -- voir reference/MODELS.md,
"Combien de Fouines viser" pour le resultat et sa lecture.

Usage :
    python reference/gen_fouine_marginal.py [n_games]
"""
import random, statistics, sys, time
from pathlib import Path
REPO = Path("/home/user/foret_mixte")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reference"))

import engine as E
import scoring_ref as R
from cards import POSITIONS
from game import Game
import search as S
import value_policy as VP
from collections import Counter, defaultdict


def to_ref(forest):
    ref = R.RefForest()
    for tid in forest.species:
        ref.add_tree(E.TREE_NAME[tid])
    for tree_idx in range(len(forest.species)):
        for pos_id, pos_name in enumerate(POSITIONS):
            for did, _symbol in forest.slots[tree_idx][pos_id]:
                ref.add_dweller(tree_idx, pos_name, E.DWELLER_NAME[did])
    ref.cave = forest.cave
    return ref


def play_one(seed, tiebreak):
    game = Game(n_players=2, seed=seed)
    turns = 0
    while not game.over and turns < 600:
        action = S.greedy_action(game, None, tiebreak=tiebreak)
        game.apply(action)
        turns += 1
    return game


if __name__ == "__main__":
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    tiebreak = VP.make_pairwise_gbm_tiebreak()

    t0 = time.time()
    buckets = defaultdict(list)  # n_fouines -> list of (fully_occupied, n_trees, beech_marten_pts, total_score)
    for gi in range(n_games):
        game = play_one(80000 + gi, tiebreak)
        for seat in (0, 1):
            forest = game.players[seat].forest
            ref = to_ref(forest)
            opp = to_ref(game.players[1 - seat].forest)
            n_fouines = R.count_card_names(ref, ["BEECH_MARTEN"])
            fully_occ = R._fully_occupied_trees(ref)
            total = R.score_forest(ref, opponents=[opp])
            beech_marten_pts = 5 * n_fouines * fully_occ
            buckets[n_fouines].append((fully_occ, forest.n_trees, beech_marten_pts, total))
        if (gi + 1) % 50 == 0:
            print(f"  {gi+1}/{n_games} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{n_games} parties, {sum(len(v) for v in buckets.values())} forets -- {time.time()-t0:.0f}s\n")
    print(f"{'n_Fouines':>10s} {'n':>6s} {'arbres_pleins':>14s} {'n_arbres':>9s} "
          f"{'pts_Fouine':>11s} {'score_total':>12s}")
    for k in sorted(buckets):
        rows = buckets[k]
        n = len(rows)
        fo = statistics.mean(r[0] for r in rows)
        nt = statistics.mean(r[1] for r in rows)
        bm = statistics.mean(r[2] for r in rows)
        tot = statistics.mean(r[3] for r in rows)
        print(f"{k:10d} {n:6d} {fo:14.2f} {nt:9.2f} {bm:11.1f} {tot:12.1f}")

    print("\nvaleur marginale implicite (delta pts_Fouine entre k et k-1) :")
    means = {k: statistics.mean(r[2] for r in v) for k, v in buckets.items()}
    for k in sorted(means):
        if k - 1 in means:
            print(f"  {k-1} -> {k} Fouine(s) : +{means[k]-means[k-1]:.1f} pts en moyenne")
