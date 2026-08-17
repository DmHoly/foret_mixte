"""Gating du modèle bootstrap (reference/bootstrap_model.joblib, voir
gen_bootstrap_dataset.py / train_bootstrap_model.py) : ne JAMAIS le
promouvoir comme modèle par défaut sur la seule foi de son R²/MAE offline
(la leçon centrale de reference/MODELS.md) -- il doit d'abord battre
mesurablement le modèle "officiel" actuel ET le meilleur bot glouton du
dépôt (B, Greedy + Clairière forte) en tête-à-tête réel. C'est le
mécanisme de "gating" qu'AlphaGo Zero utilise aussi (un candidat ne
remplace le meilleur réseau que s'il le bat sur un vrai match).

Trois variantes comparées, même heuristique de Clairière (forte, le
défaut du module) pour isoler l'effet du SEUL modèle de valeur :
  N = MCTS + reference/bootstrap_model.joblib (candidat, auto-jeu MCTS)
  O = MCTS + reference/pairwise_model.joblib  (officiel actuel, auto-jeu greedy)
  B = Greedy + Clairière forte                (meilleur bot glouton du dépôt)

Usage :
    python reference/bench_bootstrap.py            # tournoi complet (3 matchs)
    python reference/bench_bootstrap.py --quick     # n réduit, test rapide
"""
import argparse
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


def _make_mcts(seat, seed, model_path, iterations=150, short_rollout_depth=10):
    return MCTS(observer=seat, iterations=iterations, seed=seed, rollout_depth=40,
                leaf_eval=VP.make_pairwise_hybrid_leaf_eval(
                    model_path=model_path,
                    short_rollout_depth=short_rollout_depth, seed=seed))


VARIANTS = {
    "N": ("mcts", HERE / "bootstrap_model.joblib", "MCTS + modèle bootstrap (candidat)"),
    "O": ("mcts", HERE / "pairwise_model.joblib", "MCTS + modèle officiel"),
    "B": ("greedy", None, "Greedy + Clairière forte"),
}


def play_one(seed, seat_of_first, label_first, label_second, iterations):
    policy_first, model_first, _ = VARIANTS[label_first]
    policy_second, model_second, _ = VARIANTS[label_second]
    game = Game(n_players=2, seed=seed)
    policy = {seat_of_first: policy_first, 1 - seat_of_first: policy_second}
    model_path = {seat_of_first: model_first, 1 - seat_of_first: model_second}
    bots = {s: _make_mcts(s, seed * 31 + s, model_path[s], iterations=iterations)
            for s in (0, 1) if policy[s] == "mcts"}

    turns = 0
    while not game.over and turns < 600:
        seat = game.current
        action = bots[seat].choose(game) if policy[seat] == "mcts" else greedy_action(game, None)
        game.apply(action)
        for b in bots.values():
            b.advance(action)
        turns += 1

    return game.scores()[seat_of_first], game.scores()[1 - seat_of_first]


def run_matchup(label_first, label_second, n=20, seedbase=0, iterations=150):
    diffs, scores_first, scores_second = [], [], []
    t0 = time.time()
    for i in range(n):
        seed = seedbase + i
        seat_of_first = i % 2  # alterne les sièges pour neutraliser l'avantage du premier
        sf, ss = play_one(seed, seat_of_first, label_first, label_second, iterations)
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
        "mean_score_first": statistics.mean(scores_first),
        "mean_score_second": statistics.mean(scores_second),
        "seconds": time.time() - t0,
    }


def print_result(r):
    print(f"[{r['label']:8s}] {r['wins_first']}/{r['n']} - {r['wins_second']}/{r['n']} "
          f"({r['ties']} nul·s)  ecart moy {r['mean_diff']:+.1f} (SE {r['se_diff']:.1f})  "
          f"scores {r['mean_score_first']:.1f}/{r['mean_score_second']:.1f}"
          f"  -- {r['seconds']:.0f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="n reduit (8) pour un test rapide")
    parser.add_argument("--n", type=int, default=20, help="parties par match (defaut 20)")
    parser.add_argument("--iterations", type=int, default=150, help="iterations MCTS (defaut 150)")
    args = parser.parse_args()
    n = 8 if args.quick else args.n

    if not (HERE / "bootstrap_model.joblib").exists():
        print("reference/bootstrap_model.joblib introuvable -- lancer d'abord "
              "gen_bootstrap_dataset.py puis train_bootstrap_model.py.", file=sys.stderr)
        sys.exit(1)

    print("Variantes :")
    for k, (policy, _model, desc) in VARIANTS.items():
        print(f"  {k} = {desc}")
    print()

    matchups = [("N", "O", 3000), ("N", "B", 4000), ("O", "B", 5000)]
    results = []
    for first, second, seedbase in matchups:
        r = run_matchup(first, second, n=n, seedbase=seedbase, iterations=args.iterations)
        print_result(r)
        results.append(r)

    print()
    print("Verdict gating : le candidat N doit battre O ET B pour etre promu "
          "(remplacer reference/pairwise_model.joblib) -- jamais sur la seule "
          "base du R^2/MAE offline.")


if __name__ == "__main__":
    main()
