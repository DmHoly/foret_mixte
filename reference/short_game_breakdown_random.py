"""Decompose le score par mecanisme sur des forets tronquees a ~50 coups
par joueur (pas des parties completes), en jeu 100% aleatoire -- question
de Mehdi (19/08) : les gros leviers identifies sur des parties completes
E vs E (~190 coups/joueur, voir reference/gen_score_breakdown.py et
docs/strategic_guide.html : cervides 25%, Sycomore/Fouine en tete des
combos) restent-ils les memes sur une partie courte comme celle qu'il
joue en vrai (~50 coups/joueur, sans jugement de qualite de jeu) ?

Reutilise `score_forest_breakdown`/`to_ref`/`LEVERS` de
reference/gen_score_breakdown.py (meme decomposition carte-a-carte,
verifiee egale au score reel du moteur) mais sur des trajectoires
100% aleatoires (rng.choice(legal_actions()), comme
reference/gen_pairwise_dataset_random.py) tronquees a CAP actions par
joueur au lieu de jouees jusqu'a la fin naturelle -- 30 000 configurations
de foret en quelques dizaines de secondes puisque le jeu aleatoire est
quasi gratuit.

Usage : python reference/short_game_breakdown_random.py [n_games] [cap_par_joueur]
"""
import random
import statistics
import sys
import time
from pathlib import Path

REPO = Path("/home/user/foret_mixte")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reference"))

from collections import Counter
from game import Game
from gen_score_breakdown import to_ref, score_forest_breakdown, LEVERS

# Combos vedettes du guide complet (docs/combo_guide.html), pour comparaison
# directe des frequences de realisation partie courte vs partie complete.
HEADLINE_COMBOS = [
    "SYCAMORE", "BEECH_MARTEN", "ROE_DEER", "RED_DEER", "GOSHAWK", "FALLOW_DEER",
]


def play_random_capped(seed, cap):
    g = Game(n_players=2, seed=seed)
    rng = random.Random(seed)
    counts = [0, 0]
    while not g.over and min(counts) < cap:
        actor = g.current
        if counts[actor] >= cap:
            # ce joueur a atteint son quota, on laisse l'autre continuer
            # jusqu'a ce qu'il l'atteigne aussi (comparaison equitable)
            pass
        acts = g.legal_actions()
        g.apply(acts[0] if len(acts) == 1 else rng.choice(acts))
        counts[actor] += 1
    return g


if __name__ == "__main__":
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    t0 = time.time()
    total_bd = Counter()
    combo_hits = Counter()
    total_score = 0.0
    n_forests = 0
    scores_all = []
    cards_placed_all = []

    for gi in range(n_games):
        g = play_random_capped(50000 + gi, cap)
        refs = [to_ref(g.players[s].forest) for s in (0, 1)]
        for seat in (0, 1):
            bd = score_forest_breakdown(refs[seat], opponents=[refs[1 - seat]])
            total_bd.update(bd)
            for name in HEADLINE_COMBOS:
                if bd.get(name, 0) > 0:
                    combo_hits[name] += 1
            sc = sum(bd.values())
            total_score += sc
            scores_all.append(sc)
            cards_placed_all.append(g.players[seat].forest.n_trees + g.players[seat].forest.n_dwellers)
            n_forests += 1
        if (gi + 1) % 2000 == 0:
            print(f"  {gi+1}/{n_games} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{n_games} parties tronquees a {cap} coups/joueur, {n_forests} forets")
    print(f"score moyen/foret : {total_score/n_forests:.1f} (mediane {statistics.median(scores_all):.0f}) "
          f"| cartes posees/foret : {statistics.mean(cards_placed_all):.1f} -- {time.time()-t0:.0f}s\n")

    print(f"{'source':28s} {'moy/foret':>10s} {'% du score':>10s}")
    for name, pts in total_bd.most_common(15):
        print(f"{name:28s} {pts/n_forests:10.2f} {pts/total_score*100:9.1f}%")

    print(f"\n-- regroupe par levier --")
    lever_totals = []
    for lever, cards in LEVERS.items():
        pts = sum(total_bd.get(c, 0) for c in cards)
        lever_totals.append((lever, pts))
    lever_totals.sort(key=lambda x: -x[1])
    for lever, pts in lever_totals:
        print(f"{lever:32s} {pts/n_forests:8.2f} pts/foret  {pts/total_score*100:5.1f}%")

    print(f"\n-- frequence de realisation, combos vedettes (courte vs voir docs/combo_guide.html) --")
    for name in HEADLINE_COMBOS:
        print(f"{name:28s} {combo_hits[name]/n_forests*100:5.1f}% des forets (courte)")
