"""Meme question que short_game_breakdown_random.py mais avec E (greedy +
tiebreak GBM) au lieu du jeu aleatoire -- Mehdi (19/08) : la version
aleatoire montre ce qui arrive PAR HASARD en 50 coups, pas la strategie
OPTIMALE pour un nombre de coups reduit. Ici E joue "normalement" (ses
heuristiques sont reglees pour une partie complete, pas explicitement pour
un budget court) mais on tronque a `cap` coups/joueur et on regarde :

1. La decomposition par mecanisme (comme d'habitude), pour voir ce qu'un
   joueur COMPETENT accumule en 50 coups.
2. Top vs bottom : les forets a fort score en 50 coups sont-elles
   distinguees par des leviers differents de la moyenne ? (pas juste "quel
   mecanisme rapporte le plus en moyenne", mais "qu'est-ce qui differencie
   une bonne partie courte d'une mauvaise").

Usage : python reference/short_game_breakdown_e.py [n_games] [cap_par_joueur]
"""
import statistics
import sys
import time
from pathlib import Path

REPO = Path("/home/user/foret_mixte")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reference"))

from collections import Counter
from game import Game
import search as S
import value_policy as VP
from gen_score_breakdown import to_ref, score_forest_breakdown, LEVERS

HEADLINE_COMBOS = ["SYCAMORE", "BEECH_MARTEN", "ROE_DEER", "RED_DEER", "GOSHAWK", "FALLOW_DEER"]


def play_e_capped(seed, cap, tiebreak):
    g = Game(n_players=2, seed=seed)
    counts = [0, 0]
    while not g.over and min(counts) < cap:
        actor = g.current
        action = S.greedy_action(g, None, tiebreak=tiebreak)
        g.apply(action)
        counts[actor] += 1
    return g


if __name__ == "__main__":
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    tiebreak = VP.make_pairwise_gbm_tiebreak()

    t0 = time.time()
    total_bd = Counter()
    combo_hits = Counter()
    total_score = 0.0
    n_forests = 0
    rows = []  # (score, breakdown dict, n_cards)

    for gi in range(n_games):
        g = play_e_capped(60000 + gi, cap, tiebreak)
        refs = [to_ref(g.players[s].forest) for s in (0, 1)]
        for seat in (0, 1):
            bd = score_forest_breakdown(refs[seat], opponents=[refs[1 - seat]])
            total_bd.update(bd)
            for name in HEADLINE_COMBOS:
                if bd.get(name, 0) > 0:
                    combo_hits[name] += 1
            sc = sum(bd.values())
            total_score += sc
            n_cards = g.players[seat].forest.n_trees + g.players[seat].forest.n_dwellers
            rows.append((sc, dict(bd), n_cards))
            n_forests += 1
        if (gi + 1) % 200 == 0:
            print(f"  {gi+1}/{n_games} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{n_games} parties (E, tronquees a {cap} coups/joueur), {n_forests} forets")
    scores_all = [r[0] for r in rows]
    cards_all = [r[2] for r in rows]
    print(f"score moyen/foret : {total_score/n_forests:.1f} (mediane {statistics.median(scores_all):.0f}) "
          f"| cartes posees/foret : {statistics.mean(cards_all):.1f} -- {time.time()-t0:.0f}s\n")

    print(f"{'source':28s} {'moy/foret':>10s} {'% du score':>10s}")
    for name, pts in total_bd.most_common(15):
        print(f"{name:28s} {pts/n_forests:10.2f} {pts/total_score*100:9.1f}%")

    print(f"\n-- regroupe par levier (moyenne toutes forets) --")
    lever_totals = []
    for lever, cards in LEVERS.items():
        pts = sum(total_bd.get(c, 0) for c in cards)
        lever_totals.append((lever, pts))
    lever_totals.sort(key=lambda x: -x[1])
    for lever, pts in lever_totals:
        print(f"{lever:32s} {pts/n_forests:8.2f} pts/foret  {pts/total_score*100:5.1f}%")

    print(f"\n-- frequence de realisation, combos vedettes --")
    for name in HEADLINE_COMBOS:
        print(f"{name:28s} {combo_hits[name]/n_forests*100:5.1f}% des forets")

    # --- top vs bottom : qu'est-ce qui distingue les meilleures forets courtes ? ---
    rows.sort(key=lambda r: -r[0])
    k = max(1, n_forests // 5)
    top = rows[:k]
    bottom = rows[-k:]
    print(f"\n-- top 20% (score moyen {statistics.mean(r[0] for r in top):.1f}, "
          f"{statistics.mean(r[2] for r in top):.1f} cartes) "
          f"vs bottom 20% (score moyen {statistics.mean(r[0] for r in bottom):.1f}, "
          f"{statistics.mean(r[2] for r in bottom):.1f} cartes) --")
    all_sources = set()
    for r in rows:
        all_sources.update(r[1].keys())
    diffs = []
    for src in all_sources:
        top_avg = statistics.mean(r[1].get(src, 0) for r in top)
        bot_avg = statistics.mean(r[1].get(src, 0) for r in bottom)
        diffs.append((src, top_avg, bot_avg, top_avg - bot_avg))
    diffs.sort(key=lambda x: -x[3])
    print(f"{'source':28s} {'top20%':>8s} {'bottom20%':>10s} {'ecart':>8s}")
    for src, t, b, d in diffs[:15]:
        print(f"{src:28s} {t:8.2f} {b:10.2f} {d:+8.2f}")
