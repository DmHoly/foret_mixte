"""Mesure la "force" intrinsèque de chaque carte par retrait contrefactuel
(leave-one-out) sur des forêts de FIN de partie réelle, sous E (greedy +
tiebreak Gradient Boosting, le bot le plus fort du dépôt) -- pas sous
greedy nu comme `card_strength.py`.

Pourquoi une version dédiée à E : le classement du dépôt a changé après
que `card_strength.py`/`card_values.py` aient été générés (voir
reference/MODELS.md, "Comparateur pairwise dans greedy_action"). Plus
important, question de Mehdi (18/08 soir) sur la décomposition du score
par mécanisme (`gen_score_breakdown.py`) : cette décomposition attribue
les points EXACTEMENT comme la formule de scoring les distribue carte par
carte, mais **n'est PAS une mesure contrefactuelle** -- pour un combo qui
compte ses propres pairs (ROE_DEER compte "symboles de l'arbre porteur",
et un ROE_DEER est lui-même un symbole pour tout ROE_DEER voisin sur le
même hôte), retirer une seule copie fait baisser le score de PLUS que sa
part attribuée par la formule (elle emporte aussi la contribution qu'elle
apportait aux autres copies). Inversement, pour une carte qui appartient
à un TYPE compté par une autre carte (LYNX est un PawedAnimal, ce qui
alimente STAG_BEETLE ; WILD_BOAR est un ClovenhoofedAnimal, ce qui
alimente FALLOW_DEER), la part attribuée par la formule SOUS-estime sa
vraie valeur -- elle ignore ces effets de bord positifs sur d'autres
cartes. Le retrait contrefactuel capture les deux effets correctement,
parce qu'il mesure "combien perd le score TOTAL", pas "combien cette
ligne de formule vaut isolément".

Reprend telle quelle la méthode de `card_strength.py` (rejoue la partie,
puis retire CHAQUE carte posée une par une de la forêt FINALE, mesure la
perte de points) -- seul changement : la politique de génération des
trajectoires est E des deux côtés (greedy + tiebreak GBM), pas greedy nu.

Usage : python reference/card_strength_E.py [n_games]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import statistics
import time
from collections import defaultdict

import engine as E
from game import DWELLER, Game, TREE
from search import greedy_action
import value_policy as VP


def play_and_log(seed, tiebreak):
    """Identique à `card_strength.play_and_log`, mais avec E (greedy +
    tiebreak) des deux côtés au lieu de greedy nu.

    La Grotte (`forest.cave`) est suivie par delta avant/après CHAQUE
    action de player 0, pas seulement sur `cave_discard` -- corrige un bug
    présent dans `card_strength.py` original : BROWN_BEAR (Ours brun)
    envoie toute la Clairière courante à la Grotte comme effet de pose
    inconditionnel (`CLEARING_TO_CAVE_DWELLERS`, voir `game.py`
    `_resolve_dweller_effect`), un mécanisme totalement différent de
    `cave_discard` (Raton laveur) que l'ancien journal ne capturait pas du
    tout -- silencieusement faux (`baseline != score`) dès qu'un Ours brun
    est joué. Suivre le delta réel de `forest.cave`, quelle qu'en soit la
    cause, est robuste à ce cas ET à tout futur mécanisme similaire.
    """
    game = Game(n_players=2, seed=seed)
    log = []
    turns = 0
    forest0 = game.players[0].forest
    while not game.over and turns < 400:
        me = game.current
        cave_before = forest0.cave
        action = greedy_action(game, None, tiebreak=tiebreak)
        game.apply(action)
        turns += 1
        if me != 0:
            continue
        cave_delta = forest0.cave - cave_before
        if cave_delta:
            log.append(("cave", cave_delta))
        if action[0] == "tree":
            log.append(("tree", action[1], forest0.n_trees - 1))
        elif action[0] in ("dweller", "free_dweller"):
            _, did, tree_idx, pos = action
            symbol = forest0.slots[tree_idx][pos][-1][1]
            log.append(("dweller", did, tree_idx, pos, symbol))
    return log, forest0.score()


def rebuild(log, skip_tree_ord=None, skip_dweller_pos=None):
    f = E.Forest()
    tree_map = {}
    new_idx = 0
    dweller_seen = -1
    for ev in log:
        if ev[0] == "tree":
            _, tid, ord_ = ev
            if ord_ == skip_tree_ord:
                continue
            f.add_tree(tid)
            tree_map[ord_] = new_idx
            new_idx += 1
        elif ev[0] == "cave":
            f.cave += ev[1]
        else:
            _, did, tree_ord, pos, symbol = ev
            if tree_ord == skip_tree_ord:
                continue
            dweller_seen += 1
            if dweller_seen == skip_dweller_pos:
                continue
            f.add_dweller(tree_map[tree_ord], pos, did, symbol)
    return f


def analyze_game(log):
    baseline = rebuild(log).score()
    results = []

    tree_events = [(i, ev) for i, ev in enumerate(log) if ev[0] == "tree"]
    for _, (_, tid, ord_) in tree_events:
        alt = rebuild(log, skip_tree_ord=ord_).score()
        results.append(("tree", tid, baseline - alt))

    dweller_events = [ev for ev in log if ev[0] == "dweller"]
    for pos, (_, did, tree_ord, _p, _sym) in enumerate(dweller_events):
        alt = rebuild(log, skip_dweller_pos=pos).score()
        results.append(("dweller", did, baseline - alt))

    return results, baseline


def dump_values(tree_marginals, dweller_marginals, out_path=None):
    out_path = out_path or (Path(__file__).resolve().parent / "card_values_E.py")
    tree_vals = {E.TREE_NAME[tid]: statistics.mean(m)
                 for tid, m in tree_marginals.items() if m}
    dweller_vals = {E.DWELLER_NAME[did]: statistics.mean(m)
                     for did, m in dweller_marginals.items() if m}
    lines = [
        '"""Généré par card_strength_E.py -- marginal_brut moyen par carte',
        "(retrait contrefactuel sur forêt de fin de partie réelle, E des deux côtés).",
        'Ne pas éditer à la main, régénérer via `python reference/card_strength_E.py N`."""',
        "",
        "TREE_MARGINAL_BRUT = " + repr(tree_vals),
        "DWELLER_MARGINAL_BRUT = " + repr(dweller_vals),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Table sauvegardée dans {out_path}")


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    tiebreak = VP.make_pairwise_gbm_tiebreak()

    tree_marginals = defaultdict(list)
    dweller_marginals = defaultdict(list)
    scores = []

    t0 = time.perf_counter()
    for gi in range(n_games):
        seed = 70000 + gi
        log, score = play_and_log(seed, tiebreak)
        results, baseline = analyze_game(log)
        assert abs(baseline - score) < 1e-6, (baseline, score)
        scores.append(score)
        for kind, cid, marg in results:
            (tree_marginals if kind == "tree" else dweller_marginals)[cid].append(marg)
        print(f"partie {gi + 1}/{n_games} : score {score}, {len(log)} poses | "
              f"{time.perf_counter() - t0:.0f}s écoulées", flush=True)

    print()
    print(f"Score E moyen : {statistics.mean(scores):.1f}")

    all_marginals = [m for lst in tree_marginals.values() for m in lst]
    all_marginals += [m for lst in dweller_marginals.values() for m in lst]
    avg_card_value = statistics.mean(all_marginals) if all_marginals else 0.0
    print(f"Valeur moyenne d'une carte posée (toutes confondues) : "
          f"{avg_card_value:.2f} pts -- sert de taux de change pour le coût")

    print()
    print("=== ARBRES (retrait en cascade : l'arbre + tout ce qui est dessus) ===")
    print(f"{'carte':20s} {'n':>4s} {'coût':>5s} {'marginal moy':>13s} "
          f"{'force nette':>12s}")
    rows = []
    for tid in range(E.N_TREES):
        name = E.TREE_NAME[tid]
        marg = tree_marginals.get(tid, [])
        if not marg:
            continue
        mean_marg = statistics.mean(marg)
        cost = E.TREE_COST[tid]
        net = mean_marg - cost * avg_card_value
        rows.append((net, name, len(marg), cost, mean_marg))
    for net, name, n, cost, mean_marg in sorted(rows, reverse=True):
        print(f"{name:20s} {n:4d} {cost:5d} {mean_marg:13.2f} {net:12.2f}")

    print()
    print("=== HABITANTS (retrait individuel, reste de la forêt inchangé) ===")
    print(f"{'carte':28s} {'n':>4s} {'coût':>5s} {'empil.':>7s} "
          f"{'marginal moy':>13s} {'force nette':>12s}")
    rows = []
    for did in range(E.N_DWELLERS):
        name = E.DWELLER_NAME[did]
        marg = dweller_marginals.get(did, [])
        if not marg:
            continue
        mean_marg = statistics.mean(marg)
        cost = E.DWELLER_COST[did]
        share = E.SHARE_MAX[did]
        net = mean_marg - cost * avg_card_value
        rows.append((net, name, len(marg), cost, share, mean_marg))
    for net, name, n, cost, share, mean_marg in sorted(rows, reverse=True):
        share_lbl = "illim." if share == -1 else ("1" if share == 0 else str(share))
        print(f"{name:28s} {n:4d} {cost:5d} {share_lbl:>7s} {mean_marg:13.2f} {net:12.2f}")

    print()
    dump_values(tree_marginals, dweller_marginals)


if __name__ == "__main__":
    main()
