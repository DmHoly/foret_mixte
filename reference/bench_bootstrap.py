"""Gating des candidats bootstrap (reference/bootstrap_model.joblib et,
depuis le pilote 3, reference/policy_model.joblib) : ne JAMAIS les
promouvoir comme modèle(s) par défaut sur la seule foi de leur R²/MAE
offline (la leçon centrale de reference/MODELS.md) -- ils doivent d'abord
battre mesurablement le modèle "officiel" actuel ET le meilleur bot
glouton du dépôt (B, Greedy + Clairière forte) en tête-à-tête réel. C'est
le mécanisme de "gating" qu'AlphaGo Zero utilise aussi (un candidat ne
remplace le meilleur réseau que s'il le bat sur un vrai match).

Trois variantes comparées, même heuristique de Clairière (forte, le
défaut du module) pour isoler l'effet du SEUL changement testé :
  N = MCTS + PUCT (reference/policy_model.joblib comme prior) + évaluation
      directe par reference/bootstrap_model.joblib (SANS rollout -- voir
      value_policy.make_pairwise_leaf_eval). Candidat "pilote 3" : sans
      heuristique câblée dans l'évaluation, chaque action légale reste
      dans l'arbre (rien n'est filtré, voir search.Node.uct_select).
  O = MCTS + reference/pairwise_model.joblib, UCT nu + rollout court via
      greedy_action (configuration "officielle" actuelle, celle utilisée
      par défaut ailleurs dans le dépôt).
  B = Greedy + Clairière forte (meilleur bot glouton du dépôt).

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


def _make_official_mcts(seat, seed, iterations, short_rollout_depth=10):
    """O : configuration officielle actuelle -- UCT nu, leaf_eval hybride
    (quelques coups réels via greedy_action, heuristique, puis modèle)."""
    return MCTS(observer=seat, iterations=iterations, seed=seed, rollout_depth=40,
                leaf_eval=VP.make_pairwise_hybrid_leaf_eval(
                    short_rollout_depth=short_rollout_depth, seed=seed))


def _make_policy_puct_mcts(seat, seed, iterations):
    """N : PUCT guidé par policy_model.joblib, évaluation directe (sans
    rollout) par bootstrap_model.joblib -- aucune heuristique câblée dans
    la boucle de décision de CE bot (choose_draw_source/choose_payment
    restent heuristiques pour les deux bots, hors scope, voir
    gen_bootstrap_dataset.py)."""
    return MCTS(observer=seat, iterations=iterations, seed=seed,
                leaf_eval=VP.make_pairwise_leaf_eval(
                    model_path=HERE / "bootstrap_model.joblib"),
                policy_prior=VP.make_policy_prior(HERE / "policy_model.joblib"))


VARIANTS = {
    "N": ("mcts_policy_puct", "MCTS + PUCT (policy_model) + valeur pure (bootstrap_model), sans heuristique dans l'évaluation"),
    "O": ("mcts_official", "MCTS + modèle officiel (UCT nu + rollout court heuristique)"),
    "B": ("greedy", "Greedy + Clairière forte"),
}

_MAKERS = {
    "mcts_policy_puct": _make_policy_puct_mcts,
    "mcts_official": _make_official_mcts,
}


def play_one(seed, seat_of_first, label_first, label_second, iterations):
    kind_first, _ = VARIANTS[label_first]
    kind_second, _ = VARIANTS[label_second]
    game = Game(n_players=2, seed=seed)
    kind = {seat_of_first: kind_first, 1 - seat_of_first: kind_second}
    bots = {s: _MAKERS[kind[s]](s, seed * 31 + s, iterations)
            for s in (0, 1) if kind[s] != "greedy"}

    turns = 0
    while not game.over and turns < 600:
        seat = game.current
        action = bots[seat].choose(game) if seat in bots else greedy_action(game, None)
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

    for needed in ("bootstrap_model.joblib", "policy_model.joblib"):
        if not (HERE / needed).exists():
            print(f"reference/{needed} introuvable -- lancer d'abord "
                  "gen_bootstrap_dataset.py puis train_bootstrap_model.py/"
                  "train_policy_model.py.", file=sys.stderr)
            sys.exit(1)

    print("Variantes :")
    for k, (_kind, desc) in VARIANTS.items():
        print(f"  {k} = {desc}")
    print()

    matchups = [("N", "O", 3000), ("N", "B", 4000), ("O", "B", 5000)]
    results = []
    for first, second, seedbase in matchups:
        r = run_matchup(first, second, n=n, seedbase=seedbase, iterations=args.iterations)
        print_result(r)
        results.append(r)

    print()
    print("Verdict gating (deux paliers, voir reference/MODELS.md) : N bat O "
          "-> devient la base d'auto-jeu de la prochaine generation ; N bat O "
          "ET B -> seuil de promotion en modele officiel. Jamais sur la seule "
          "base du R^2/MAE offline.")


if __name__ == "__main__":
    main()
