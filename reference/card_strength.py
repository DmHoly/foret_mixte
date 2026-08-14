"""Mesure la "force" intrinsèque de chaque carte par retrait contrefactuel
(leave-one-out) sur des forêts de FIN de partie réelle, plutôt qu'un delta
local au moment de la pose.

Pourquoi le retrait en fin de partie et pas le delta immédiat (celui que
`greedy_action` utilise) : plusieurs cartes ont une valeur qui se compose
avec toute la composition finale de la forêt (ROE_DEER compte tous les
symboles d'arbre posés, y compris les habitants posés APRÈS lui ; les
seuils MOSS/pic épeiche dépendent du nombre total d'arbres en fin de
partie). Un delta pris au moment de la pose sous-estime structurellement
ces cartes -- c'est le biais d'horizon diagnostiqué en session (le MCTS à
rollout court + modèle contrastif néglige un peu les cervidés pour cette
raison). Ici on rejoue la partie jusqu'au bout, puis on retire CHAQUE
carte posée une par une de la forêt FINALE et on mesure combien de points
ça coûte -- la vraie contribution de cette carte à CE résultat final.

Deux mesures différentes selon le type de carte :
  - HABITANT : retrait de cette seule instance, le reste de la forêt
    inchangé (marginal pur, "à composition égale par ailleurs").
  - ARBRE : retrait en cascade -- l'arbre ET tout ce qui a été posé
    dessus (on ne peut pas garder des habitants sans leur support). C'est
    donc "la valeur totale que cette pousse d'arbre a rendue possible",
    pas seulement ses propres points de seuil d'espèce.

Le retrait se fait par reconstruction : on rejoue le journal des poses du
joueur observé, en omettant l'instance ciblée (et, pour un arbre, tout ce
qui référence son index), sur une Forest fraîche. Le moteur incrémental
étant rapide (add_tree/add_dweller ~qq µs), reconstruire toute la forêt
une fois par instance retirée reste rapide (dizaines de ms par partie).

Politique de génération des trajectoires : greedy (rapide, volume élevé).
Ce n'est pas la politique optimale, donc la métrique reflète "la force
d'une carte sous du jeu raisonnable", pas sous jeu parfait -- suffisant
pour classer les cartes entre elles.

Usage : python card_strength.py [n_games]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import statistics
import time
from collections import defaultdict

import engine as E
from game import DWELLER, Game, TREE
from search import greedy_policy


def play_and_log(seed):
    """Joue une partie greedy vs greedy, retourne le journal des poses du
    joueur 0 : liste de ('tree', tree_id, ord) ou
    ('dweller', dweller_id, tree_ord, pos, symbol), dans l'ordre réel.
    `tree_ord` est l'index d'arbre tel qu'assigné par Forest (0, 1, 2...
    dans l'ordre de plantation), ce qui correspond exactement au
    `tree_idx` déjà porté par les actions du jeu.
    """
    game = Game(n_players=2, seed=seed)
    log = []
    turns = 0
    while not game.over and turns < 400:
        me = game.current
        action = greedy_policy(game, None)
        game.apply(action)
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
            # Raton laveur : n cartes envoyées à la Grotte (1 pt chacune,
            # voir Game._apply_pending). Compté à part : ce n'est pas une
            # "carte posée en forêt" au sens de la métrique par carte, mais
            # ça doit être reproduit pour que baseline == score réel.
            log.append(("cave", action[1]))
    # Forest.score() (pas game.scores()) : la métrique par carte doit
    # rester indépendante de la résolution de majorité 2 joueurs (LINDEN,
    # pic épeiche), qui dépend de la forêt ADVERSE et n'a rien à voir avec
    # la force intrinsèque d'une carte. Forest.score() assume la majorité
    # acquise, comme en solo -- c'est exactement ce qu'on veut mesurer ici.
    return log, game.players[0].forest.score()


def rebuild(log, skip_tree_ord=None, skip_dweller_pos=None):
    """Reconstruit une Forest à partir du journal, en omettant
    éventuellement un arbre (et tout ce qui le référence, en cascade) ou
    une seule instance d'habitant (par sa position dans la sous-liste des
    événements 'dweller').
    """
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
                continue  # posé sur l'arbre retiré : tombe avec lui
            dweller_seen += 1
            if dweller_seen == skip_dweller_pos:
                continue
            f.add_dweller(tree_map[tree_ord], pos, did, symbol)
    return f


def analyze_game(log):
    """Retourne les listes de marginaux (score, type, id) pour cette partie."""
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
    """Sauvegarde marginal_brut moyen par carte dans un module Python
    rechargeable (reference/card_values.py), pour ne pas rejouer les
    parties à chaque usage par une politique."""
    out_path = out_path or (Path(__file__).resolve().parent / "card_values.py")
    tree_vals = {E.TREE_NAME[tid]: statistics.mean(m)
                 for tid, m in tree_marginals.items() if m}
    dweller_vals = {E.DWELLER_NAME[did]: statistics.mean(m)
                     for did, m in dweller_marginals.items() if m}
    lines = [
        '"""Généré par card_strength.py -- marginal_brut moyen par carte',
        "(retrait contrefactuel sur forêt de fin de partie réelle, jeu greedy).",
        'Ne pas éditer à la main, régénérer via `python card_strength.py N`."""',
        "",
        "TREE_MARGINAL_BRUT = " + repr(tree_vals),
        "DWELLER_MARGINAL_BRUT = " + repr(dweller_vals),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Table sauvegardée dans {out_path}")


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    tree_marginals = defaultdict(list)
    dweller_marginals = defaultdict(list)
    scores = []

    t0 = time.perf_counter()
    for gi in range(n_games):
        seed = 70000 + gi
        log, score = play_and_log(seed)
        results, baseline = analyze_game(log)
        assert abs(baseline - score) < 1e-6, (baseline, score)
        scores.append(score)
        for kind, cid, marg in results:
            (tree_marginals if kind == "tree" else dweller_marginals)[cid].append(marg)
        print(f"partie {gi + 1}/{n_games} : score {score}, {len(log)} poses | "
              f"{time.perf_counter() - t0:.0f}s écoulées", flush=True)

    print()
    print(f"Score greedy moyen : {statistics.mean(scores):.1f}")

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
