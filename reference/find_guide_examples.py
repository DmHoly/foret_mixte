"""Variante de find_tiebreak_example.py : cherche a la fois des cas
EVIDENTS (ecart net entre le meilleur candidat et les autres, >= 8 pts)
et des cas COMPLEXES (tiebreak decisif, ecart <= 3 pts) pour illustrer le
guide rapide -- demande de Mehdi (19/08).

Usage : python reference/find_guide_examples.py [n_games]
"""
import sys
from pathlib import Path

REPO = Path("/home/user/foret_mixte")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reference"))

from game import Game, DWELLER
from engine import TREE_NAME, DWELLER_NAME
from gen_technique_guide import FR_NAMES
import search as S
import value_policy as VP

POS_FR = {"Top": "Haut", "Bottom": "Bas", "Left": "Gauche", "Right": "Droite"}
POSITIONS = ["Top", "Bottom", "Left", "Right"]


def fr(name):
    return FR_NAMES.get(name, name)


def action_label(a):
    if a[0] == "tree":
        return f"Arbre : {fr(TREE_NAME[a[1]])}"
    if a[0] in ("dweller", "free_dweller"):
        _, did, tree_idx, pos = a
        return f"{fr(DWELLER_NAME[did])} (arbre #{tree_idx}, {POS_FR.get(POSITIONS[pos], pos)})"
    return str(a)


def analyze(game, tiebreak, tiebreak_margin=3.0):
    actions = game.legal_actions()
    plays = [a for a in actions if a[0] not in ("draw", "skip_effect")]
    if len(plays) < 2:
        return None
    player = game.players[game.current]
    forest = player.forest
    base = forest.score()
    free_slots = 4 * forest.n_trees - sum(forest.occupied_positions)
    hand_dwellers = sum(1 for c in player.hand if c[0] == DWELLER)
    deficit = max(0, hand_dwellers - free_slots)
    slot_bonus = 6 * min(1.0, deficit / 3.0)

    gains = {}
    best, best_gain = None, -1e9
    for a in plays:
        if a[0] == "tree":
            gain = forest.delta_tree(a[1]) + slot_bonus
        else:
            _, did, tree_idx, pos = a
            forest.add_dweller(tree_idx, pos, did)
            gain = forest.score() - base
            forest.undo_dweller(tree_idx, pos, did)
        gains[a] = gain
        if gain > best_gain:
            best, best_gain = a, gain

    second_gain = max((g for a, g in gains.items() if a != best), default=-1e9)
    gap = best_gain - second_gain

    near = [a for a in plays if a != best and gains[a] >= best_gain - tiebreak_margin]
    chosen_by_tiebreak = None
    if near:
        observer = game.current
        best_state = game.clone()
        best_state.apply(best)
        near_states = [game.clone() for _ in near]
        for st, a in zip(near_states, near):
            st.apply(a)
        scores = tiebreak(near_states, best_state, observer)
        top = max(range(len(scores)), key=lambda i: scores[i])
        if scores[top] > 0:
            chosen_by_tiebreak = near[top]

    return {
        "plays": plays, "gains": gains, "best": best, "best_gain": best_gain,
        "gap": gap, "chosen_by_tiebreak": chosen_by_tiebreak,
    }


if __name__ == "__main__":
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    tiebreak = VP.make_pairwise_gbm_tiebreak()

    obvious_found = []
    complex_found = []

    for gi in range(n_games):
        seed = 95000 + gi
        g = Game(n_players=2, seed=seed)
        turn = 0
        while not g.over and turn < 100:
            info = analyze(g, tiebreak)
            if info is not None and len(info["plays"]) >= 3:
                if info["gap"] >= 8 and len(obvious_found) < 6:
                    obvious_found.append((gi, turn, info))
                if info["chosen_by_tiebreak"] is not None and info["chosen_by_tiebreak"] != info["best"] \
                        and len(complex_found) < 6:
                    complex_found.append((gi, turn, info))
            action = S.greedy_action(g, None, tiebreak=tiebreak)
            g.apply(action)
            turn += 1
        if len(obvious_found) >= 6 and len(complex_found) >= 6:
            break

    def dump(title, items):
        print(f"\n\n########## {title} ##########")
        for gi, turn, info in items:
            print(f"\n=== partie {gi}, tour {turn} ===")
            ranked = sorted(info["plays"], key=lambda a: -info["gains"][a])[:6]
            for a in ranked:
                tags = []
                if a == info["best"]:
                    tags.append("meilleur gain EXACT")
                if a == info["chosen_by_tiebreak"]:
                    tags.append("CHOISI par le tiebreak")
                tag = "  <- " + " / ".join(tags) if tags else ""
                print(f"  {action_label(a):45s} {info['gains'][a]:+5.1f} pts{tag}")

    dump("CHOIX EVIDENTS (ecart net)", obvious_found)
    dump("CHOIX COMPLEXES (tiebreak decisif)", complex_found)
