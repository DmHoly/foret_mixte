"""Comme card_strength.py, mais les trajectoires du joueur observé sont
jouées par le MCTS gagnant (rollout court + modèle contrastif) au lieu de
greedy. Objectif : la table `marginal_brut` doit refléter "ce qu'une bonne
recherche choisit de jouer et pourquoi ça paie", pas seulement "ce que
greedy choisit" -- greedy et MCTS divergent justement sur les cartes à
effet différé (ROE_DEER et consorts, voir la session), donc la table
greedy pouvait sous-représenter ces cartes-là dans son propre échantillon
de parties (elle ne mesure la force que des cartes que greedy a JOUÉES).

Plus lent que la version greedy (MCTS ~150-200 ms/décision), donc n_games
par défaut est volontairement modeste.

Usage : python card_strength_mcts.py [n_games] [iterations]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import statistics
from collections import defaultdict

import engine as E
import value_policy as VP
from card_strength import analyze_game, dump_values
from game import Game
from search import MCTS, greedy_action


def play_and_log_mcts(seed, iterations=300):
    """Comme card_strength.play_and_log, mais le joueur 0 est le bot MCTS
    (rollout court + modèle contrastif), le joueur 1 reste greedy."""
    game = Game(n_players=2, seed=seed)
    leaf_eval = VP.make_pairwise_hybrid_leaf_eval(short_rollout_depth=10, seed=seed)
    bot = MCTS(observer=0, iterations=iterations, seed=seed, leaf_eval=leaf_eval)
    log = []
    turns = 0
    import random
    rng = random.Random(seed)
    while not game.over and turns < 400:
        me = game.current
        if me == 0:
            action = bot.choose(game)
        else:
            action = greedy_action(game, rng)
        game.apply(action)
        bot.advance(action)
        turns += 1
        if me != 0:
            continue
        forest = game.players[0].forest
        if action[0] == "tree":
            log.append(("tree", action[1], forest.n_trees - 1))
        elif action[0] in ("dweller", "free_dweller"):
            _, did, tree_idx, pos = action
            symbol = forest.slots[tree_idx][pos][-1][1]
            log.append(("dweller", did, tree_idx, pos, symbol))
        elif action[0] == "cave_discard":
            log.append(("cave", action[1]))
    return log, game.players[0].forest.score()


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    tree_marginals = defaultdict(list)
    dweller_marginals = defaultdict(list)
    scores = []

    t0 = time.perf_counter()
    for gi in range(n_games):
        seed = 90000 + gi
        log, score = play_and_log_mcts(seed, iterations)
        results, baseline = analyze_game(log)
        assert abs(baseline - score) < 1e-6, (baseline, score)
        scores.append(score)
        for kind, cid, marg in results:
            (tree_marginals if kind == "tree" else dweller_marginals)[cid].append(marg)
        print(f"partie {gi + 1}/{n_games} : score {score}, {len(log)} poses | "
              f"{time.perf_counter() - t0:.0f}s écoulées", flush=True)

    print()
    print(f"Score MCTS moyen : {statistics.mean(scores):.1f}")

    all_marginals = [m for lst in tree_marginals.values() for m in lst]
    all_marginals += [m for lst in dweller_marginals.values() for m in lst]
    avg_card_value = statistics.mean(all_marginals) if all_marginals else 0.0
    print(f"Valeur moyenne d'une carte posée : {avg_card_value:.2f} pts")

    print()
    print("=== ARBRES (sous MCTS) ===")
    print(f"{'carte':20s} {'n':>4s} {'coût':>5s} {'marginal moy':>13s} {'force nette':>12s}")
    rows = []
    for tid in range(E.N_TREES):
        marg = tree_marginals.get(tid, [])
        if not marg:
            continue
        mean_marg = statistics.mean(marg)
        cost = E.TREE_COST[tid]
        rows.append((mean_marg - cost * avg_card_value, E.TREE_NAME[tid], len(marg), cost, mean_marg))
    for net, name, n, cost, mean_marg in sorted(rows, reverse=True):
        print(f"{name:20s} {n:4d} {cost:5d} {mean_marg:13.2f} {net:12.2f}")

    print()
    print("=== HABITANTS (sous MCTS) ===")
    print(f"{'carte':28s} {'n':>4s} {'coût':>5s} {'empil.':>7s} "
          f"{'marginal moy':>13s} {'force nette':>12s}")
    rows = []
    for did in range(E.N_DWELLERS):
        marg = dweller_marginals.get(did, [])
        if not marg:
            continue
        mean_marg = statistics.mean(marg)
        cost = E.DWELLER_COST[did]
        share = E.SHARE_MAX[did]
        rows.append((mean_marg - cost * avg_card_value, E.DWELLER_NAME[did], len(marg), cost, share, mean_marg))
    for net, name, n, cost, share, mean_marg in sorted(rows, reverse=True):
        share_lbl = "illim." if share == -1 else ("1" if share == 0 else str(share))
        print(f"{name:28s} {n:4d} {cost:5d} {share_lbl:>7s} {mean_marg:13.2f} {net:12.2f}")

    print()
    dump_values(tree_marginals, dweller_marginals,
                out_path=Path(__file__).resolve().parent / "card_values_mcts.py")


if __name__ == "__main__":
    main()
