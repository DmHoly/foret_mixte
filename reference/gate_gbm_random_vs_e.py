"""Gate le comparateur GBM entraine sur donnees 100% aleatoires
(reference/gen_pairwise_dataset_random.py + train_pairwise_gbm.py sur
pairwise_dataset_random.npz -> pairwise_gbm_model_random.joblib) contre E
(greedy_action + tiebreak GBM live, le meilleur bot du depot). Teste
l'hypothese de Mehdi (19/08, voir reference/MODELS.md) : un dataset
genere sans AUCUN appel a greedy_action (donc sans biais heuristique
d'aucune sorte) permet-il au comparateur de mieux discriminer, une fois
le volume de donnees monte a plusieurs milliers de parties ?

Usage : python reference/gate_gbm_random_vs_e.py [n_parties] [model_path]
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

HERE = Path(__file__).resolve().parent


def play_one(seed, seat_of_candidate, tiebreak_candidate, tiebreak_e):
    game = Game(n_players=2, seed=seed)
    turns = 0
    while not game.over and turns < 600:
        seat = game.current
        if seat == seat_of_candidate:
            action = greedy_action(game, None, tiebreak=tiebreak_candidate)
        else:
            action = greedy_action(game, None, tiebreak=tiebreak_e)
        game.apply(action)
        turns += 1
    return game.scores()[seat_of_candidate], game.scores()[1 - seat_of_candidate]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    model_path = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "pairwise_gbm_model_random.joblib")

    tiebreak_candidate = VP.make_pairwise_gbm_tiebreak(model_path=model_path)
    tiebreak_e = VP.make_pairwise_gbm_tiebreak()  # modele live (officiel)

    diffs = []
    t0 = time.time()
    for i in range(n):
        seed = 85000 + i
        seat_of_candidate = i % 2
        sc, se = play_one(seed, seat_of_candidate, tiebreak_candidate, tiebreak_e)
        diffs.append(sc - se)
        print(f"  partie {i+1}/{n} : candidat(random-data)={sc} E(live)={se} diff={sc-se:+.0f}", flush=True)

    wins = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)
    se_ = statistics.stdev(diffs) / (len(diffs) ** 0.5) if n > 1 else 0.0
    print()
    print(f"Candidat (GBM sur donnees aleatoires) vs E (live) : {wins}/{n} ({ties} nuls), "
          f"ecart moyen {statistics.mean(diffs):+.1f} (SE {se_:.1f}), "
          f"mediane {statistics.median(diffs):+.1f} -- {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
