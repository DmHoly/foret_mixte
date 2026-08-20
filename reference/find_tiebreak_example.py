"""Trouve un exemple REEL ou le tiebreak GBM change la decision (le gain
exact ne suffit pas a departager, l'ecart entre candidats est < 3 pts) --
pour illustrer concretement a Mehdi comment la regle "ecart <= 3 pts ->
pense investissement" se joue en vrai, avec des vraies cartes.

Usage : python reference/find_tiebreak_example.py [n_games]
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


def greedy_with_trace(game, tiebreak, tiebreak_margin=3.0):
    """Reimplementation minimale de greedy_action qui expose gains + candidats
    proches + score tiebreak, pour pouvoir tracer un exemple concret."""
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

    near = [a for a in plays if a != best and gains[a] >= best_gain - tiebreak_margin]
    if not near:
        return None

    observer = game.current
    best_state = game.clone()
    best_state.apply(best)
    near_states = [game.clone() for _ in near]
    for st, a in zip(near_states, near):
        st.apply(a)
    scores = tiebreak(near_states, best_state, observer)
    top = max(range(len(scores)), key=lambda i: scores[i])
    if scores[top] <= 0:
        return None  # tiebreak confirme le choix du gain exact, pas d'exemple interessant

    chosen = near[top]
    return {
        "candidates": [(a, gains[a]) for a in plays if a == best or a in near],
        "best_by_delta": (best, best_gain),
        "chosen_by_tiebreak": (chosen, gains[chosen]),
        "turn": None,
    }


if __name__ == "__main__":
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    tiebreak = VP.make_pairwise_gbm_tiebreak()

    found = 0
    for gi in range(n_games):
        seed = 90000 + gi
        g = Game(n_players=2, seed=seed)
        turn = 0
        while not g.over and turn < 100:  # partie courte, 50 coups/joueur
            trace = greedy_with_trace(g, tiebreak)
            if trace is not None and found < 5:
                found += 1
                print(f"=== Exemple {found} (partie {gi}, tour {turn}) ===")
                print("Candidats en lice (gain exact immediat) :")
                for a, gain in sorted(trace["candidates"], key=lambda x: -x[1]):
                    tag = ""
                    if a == trace["best_by_delta"][0]:
                        tag = "  <- meilleur gain EXACT"
                    if a == trace["chosen_by_tiebreak"][0]:
                        tag += "  <- CHOISI par le tiebreak"
                    print(f"  {action_label(a):45s} {gain:+5.1f} pts{tag}")
                print()
            action = S.greedy_action(g, None, tiebreak=tiebreak)
            g.apply(action)
            turn += 1
        if found >= 5:
            break
    print(f"{found} exemples trouves sur {gi+1} parties explorees")
