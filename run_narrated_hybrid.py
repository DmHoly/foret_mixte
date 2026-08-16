"""Rejoue une partie MCTS (config gagnante : rollout court + modèle
contrastif) coup par coup, en affichant à chaque tour :
  - la main du bot AVANT de jouer (ce qu'il garde/pourrait jouer),
  - le coup choisi (pioche, arbre planté, ou habitant posé et où),
  - les alternatives les plus visitées par la recherche (ce à quoi il a
    pensé, pondéré par le nombre de simulations),
  - le score avant/après.

Usage : python run_narrated_hybrid.py [seed] [iterations]
"""
import sys

sys.path.insert(0, "reference")

import random

import value_policy as VP
from engine import DWELLERS, POSITIONS, TREES
from game import DWELLER, Game, TREE
from search import MCTS, greedy_action

TREE_NAMES = [t.name for t in TREES]
DWELLER_NAMES = [d.name for d in DWELLERS]


def card_label(card):
    if card[0] == TREE:
        return TREE_NAMES[card[1]]
    # carte habitant : deux moitiés (did, tree_id, pos)
    h1, h2 = card[1], card[2]
    return f"{DWELLER_NAMES[h1[0]]}/{DWELLER_NAMES[h2[0]]}"


def hand_label(hand):
    return ", ".join(card_label(c) for c in hand) if hand else "(vide)"


def action_label(action):
    if action[0] == "draw":
        source = action[1]
        return "Pioche (deck)" if source is None else f"Pioche Clairière (carte #{source})"
    if action[0] == "skip_effect":
        return "Passe (effet en attente décliné)"
    if action[0] == "cave_discard":
        return f"Grotte : envoie {action[1]} carte(s)"
    if action[0] == "free_dweller":
        _, did, tree_idx, pos = action
        return f"Pose GRATUITE {DWELLER_NAMES[did]} en {POSITIONS[pos]} (arbre #{tree_idx})"
    if action[0] == "tree":
        return f"Plante {TREE_NAMES[action[1]]}"
    _, did, tree_idx, pos = action
    return f"Pose {DWELLER_NAMES[did]} en {POSITIONS[pos]} (arbre #{tree_idx})"


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 5555
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    game = Game(n_players=2, seed=seed)
    rng = random.Random(seed)
    leaf_eval = VP.make_pairwise_hybrid_leaf_eval(short_rollout_depth=10, seed=seed)
    bot = MCTS(observer=0, iterations=iterations, seed=seed, leaf_eval=leaf_eval)

    turn = 0
    while not game.over and turn < 200:
        me = game.current
        if me == 0:
            hand_before = list(game.players[0].hand)
            score_before = game.scores()[0]
            root = bot.root
            action = bot.choose(game)
            alt = sorted(root.children.items(), key=lambda kv: -kv[1].visits)[:4]
            alt_str = ", ".join(f"{action_label(a)} ({c.visits}v)" for a, c in alt)

            game.apply(action)
            bot.advance(action)
            score_after = game.scores()[0]

            turn += 1
            print(f"--- Tour {turn} (score {score_before} -> {score_after}, "
                  f"{score_after - score_before:+d}) ---")
            print(f"  Main avant   : {hand_label(hand_before)}")
            print(f"  Coup joué    : {action_label(action)}")
            print(f"  Alternatives : {alt_str}")
        else:
            action = greedy_action(game, rng)
            game.apply(action)
            bot.advance(action)

    print()
    print("Score final :", game.scores())
    forest = game.players[0].forest
    print(f"Arbres ({forest.n_trees}) :",
          ", ".join(f"{TREE_NAMES[i]}={forest.tree_count[i]}"
                     for i in range(len(TREE_NAMES)) if forest.tree_count[i]))
    print("Habitants :",
          ", ".join(f"{DWELLER_NAMES[i]}={forest.dweller_count[i]}"
                     for i in range(len(DWELLER_NAMES)) if forest.dweller_count[i]))


if __name__ == "__main__":
    main()
