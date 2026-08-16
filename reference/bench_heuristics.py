"""Tournoi rond-robin entre variantes de bot, pour comparer une politique de
décision (Greedy / MCTS) et une heuristique de pioche en Clairière
(`choose_draw_source`) indépendamment l'une de l'autre.

Contexte : `game.choose_draw_source` a deux comportements possibles --
FORTE (par défaut depuis le commit "Priorise les cartes fortes de la
Clairiere...", cible en priorité Fouine/Chevreuil/Cerf/Autour/Daim/Fourmi
des bois/Lièvre/Sycomore) et FAIBLE (l'ancien comportement, toujours la
carte la moins chère). Ce script permet de recomposer les 4 combinaisons
{Greedy, MCTS} x {FORTE, FAIBLE} et de les faire s'affronter deux à deux,
sièges alternés pour neutraliser l'avantage du premier joueur, même deck
des deux côtés à seed donnée (comparaison appariée, pas juste un delta de
moyennes indépendantes).

Résultat de référence (16/08, avec reference/pairwise_model.joblib
réentraîné sous l'heuristique forte) : voir reference/MODELS.md et le
tableau affiché par `python reference/bench_heuristics.py`.

Usage :
    python reference/bench_heuristics.py            # tournoi complet (6 matchs)
    python reference/bench_heuristics.py --quick     # n réduit, pour un test rapide
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import game as G
from game import Game, card_min_cost
from search import MCTS, greedy_action
import value_policy as VP

STRONG = G.choose_draw_source  # heuristique par défaut du module (cible carte forte)


def WEAK(clearing):
    """Ancien comportement : toujours la carte la moins chère de la Clairière,
    sans notion de force. Conservé ici (pas dans game.py) uniquement pour
    servir de point de comparaison à ce banc d'essai."""
    if not clearing:
        return None
    return min(range(len(clearing)), key=lambda i: card_min_cost(clearing[i]))


VARIANTS = {
    "A": ("greedy", WEAK, "Greedy + Clairière faible"),
    "B": ("greedy", STRONG, "Greedy + Clairière forte"),
    "C": ("mcts", WEAK, "MCTS(150it) + Clairière faible"),
    "D": ("mcts", STRONG, "MCTS(150it) + Clairière forte (défaut actuel)"),
}


def _make_mcts(seat, seed, iterations=150, short_rollout_depth=10):
    return MCTS(observer=seat, iterations=iterations, seed=seed, rollout_depth=40,
                leaf_eval=VP.make_pairwise_hybrid_leaf_eval(
                    short_rollout_depth=short_rollout_depth, seed=seed))


def play_one(seed, seat_of_first, policy_first, heur_first, policy_second, heur_second):
    """Joue une partie 2 joueurs. `seat_of_first` est le siège (0 ou 1) occupé
    par la variante "first" ; l'autre siège reçoit "second". Retourne les
    scores dans l'ordre (first, second)."""
    game = Game(n_players=2, seed=seed)
    policy = {seat_of_first: policy_first, 1 - seat_of_first: policy_second}
    heur = {seat_of_first: heur_first, 1 - seat_of_first: heur_second}
    bots = {s: _make_mcts(s, seed * 31 + s) for s in (0, 1) if policy[s] == "mcts"}

    turns = 0
    while not game.over and turns < 600:
        seat = game.current
        G.choose_draw_source = heur[seat]
        action = bots[seat].choose(game) if policy[seat] == "mcts" else greedy_action(game, None)
        game.apply(action)
        for b in bots.values():
            b.advance(action)
        turns += 1

    G.choose_draw_source = STRONG  # toujours restaurer le défaut du module en sortie
    return game.scores()[seat_of_first], game.scores()[1 - seat_of_first]


def run_matchup(label_first, label_second, n=30, seedbase=0):
    policy_first, heur_first, _ = VARIANTS[label_first]
    policy_second, heur_second, _ = VARIANTS[label_second]

    diffs, scores_first, scores_second = [], [], []
    t0 = time.time()
    for i in range(n):
        seed = seedbase + i
        seat_of_first = i % 2  # alterne les sièges pour neutraliser l'avantage du premier
        sf, ss = play_one(seed, seat_of_first, policy_first, heur_first, policy_second, heur_second)
        diffs.append(sf - ss)
        scores_first.append(sf)
        scores_second.append(ss)

    wins = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)
    se = statistics.stdev(diffs) / (len(diffs) ** 0.5) if n > 1 else 0.0
    return {
        "label": f"{label_first} vs {label_second}",
        "n": n,
        "wins_first": wins,
        "wins_second": n - wins - ties,
        "ties": ties,
        "mean_diff": statistics.mean(diffs),
        "se_diff": se,
        "median_diff": statistics.median(diffs),
        "mean_score_first": statistics.mean(scores_first),
        "mean_score_second": statistics.mean(scores_second),
        "seconds": time.time() - t0,
    }


# Les 6 matchs du tournoi rond-robin complet entre les 4 variantes, avec des
# plages de seed disjointes pour rester reproductible et comparable d'un run
# a l'autre.
ROUND_ROBIN = [
    ("B", "A", 100, 1000),
    ("D", "C", 30, 2000),
    ("B", "C", 30, 3000),
    ("D", "B", 30, 4000),
    ("D", "A", 30, 5000),
    ("C", "A", 30, 6000),
]


def print_result(r):
    print(f"[{r['label']:8s}] {r['wins_first']}/{r['n']} - {r['wins_second']}/{r['n']} "
          f"({r['ties']} nul·s)  ecart moy {r['mean_diff']:+.1f} (SE {r['se_diff']:.1f})  "
          f"mediane {r['median_diff']:+.1f}  scores {r['mean_score_first']:.1f}/{r['mean_score_second']:.1f}"
          f"  -- {r['seconds']:.0f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="n reduit (10) pour un test rapide")
    args = parser.parse_args()

    print("Variantes :")
    for k, (policy, _heur, desc) in VARIANTS.items():
        print(f"  {k} = {desc}")
    print()

    results = []
    for first, second, n, seedbase in ROUND_ROBIN:
        n_run = 10 if args.quick else n
        r = run_matchup(first, second, n=n_run, seedbase=seedbase)
        print_result(r)
        results.append(r)

    print()
    print("Classement agrege (victoires / parties jouees, toutes confondues) :")
    tally = {k: [0, 0] for k in VARIANTS}
    for (first, second, _, _), r in zip(ROUND_ROBIN, results):
        tally[first][0] += r["wins_first"]; tally[first][1] += r["n"]
        tally[second][0] += r["wins_second"]; tally[second][1] += r["n"]
    for k, (w, n) in sorted(tally.items(), key=lambda kv: -kv[1][0] / kv[1][1]):
        print(f"  {k} : {w}/{n} ({100*w/n:.1f}%)")


if __name__ == "__main__":
    main()
