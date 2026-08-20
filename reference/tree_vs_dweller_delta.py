"""Distribution du delta de score immediat (gain exact, pas une moyenne)
pour les poses d'ARBRE vs les poses d'HABITANT, aux memes noeuds de
decision -- question de Mehdi (19/08) : "un arbre vaut ~7 pts donc ca ne
sert a rien de poser un habitant qui score moins" -- vrai en moyenne
(voir docs/strategic_guide.html, delta_tree = +7.16 pts moyen), mais
cette moyenne melange debut et fin de partie (un arbre tardif revalorise
beaucoup plus de choses qu'un arbre precoce). Ici on regarde la
DISTRIBUTION complete des deux, au meme noeud de decision, dans le
regime "partie courte" (50 coups/joueur) deja etudie.

Usage : python reference/tree_vs_dweller_delta.py [n_games]
"""
import statistics
import sys
from pathlib import Path

REPO = Path("/home/user/foret_mixte")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reference"))

from game import Game
import search as S
import value_policy as VP

CAP = 50
SAMPLE_EVERY = 3


def percentile(vals, p):
    s = sorted(vals)
    idx = int(len(s) * p)
    return s[min(idx, len(s) - 1)]


if __name__ == "__main__":
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    tiebreak = VP.make_pairwise_gbm_tiebreak()

    tree_deltas = []
    dweller_deltas = []
    tree_deltas_early = []   # 1ers 15 coups/joueur
    dweller_deltas_early = []

    for gi in range(n_games):
        seed = 70000 + gi
        g = Game(n_players=2, seed=seed)
        counts = [0, 0]
        turn = 0
        while not g.over and min(counts) < CAP:
            actor = g.current
            if turn % SAMPLE_EVERY == 0:
                observer = g.current
                before = g.scores()[observer]
                for a in g.legal_actions():
                    if a[0] not in ("tree", "dweller", "free_dweller"):
                        continue
                    branch = g.clone()
                    branch.apply(a)
                    delta = branch.scores()[observer] - before
                    bucket = tree_deltas if a[0] == "tree" else dweller_deltas
                    bucket.append(delta)
                    if counts[actor] < 15:
                        bucket_e = tree_deltas_early if a[0] == "tree" else dweller_deltas_early
                        bucket_e.append(delta)
            action = S.greedy_action(g, None, tiebreak=tiebreak)
            g.apply(action)
            counts[actor] += 1
            turn += 1
        if (gi + 1) % 50 == 0:
            print(f"  {gi+1}/{n_games}", flush=True)

    def report(name, vals):
        if not vals:
            print(f"{name} : aucune donnee")
            return
        print(f"{name} (n={len(vals)}) : moyenne {statistics.mean(vals):.1f}  "
              f"mediane {statistics.median(vals):.1f}  "
              f"p25 {percentile(vals,0.25):.1f}  p75 {percentile(vals,0.75):.1f}  "
              f"%<=0 {sum(1 for v in vals if v<=0)/len(vals)*100:.1f}%  "
              f"%<7 {sum(1 for v in vals if v<7)/len(vals)*100:.1f}%")

    print(f"\n=== Sur toute la partie courte (0-50 coups/joueur) ===")
    report("ARBRE   ", tree_deltas)
    report("HABITANT", dweller_deltas)

    print(f"\n=== Seulement les 15 premiers coups/joueur (vraiment tot) ===")
    report("ARBRE   ", tree_deltas_early)
    report("HABITANT", dweller_deltas_early)
