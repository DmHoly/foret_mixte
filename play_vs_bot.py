"""Joue une partie interactive contre un bot du dépôt, dans le terminal.

À chaque tour, ton coup te est demandé (numéro dans une liste de coups
légaux) ; le bot joue automatiquement à son tour, avec un court résumé de
ce qu'il a fait. À la fin, le résultat est ajouté à
`human_vs_bot_log.jsonl` (à la racine du dépôt) pour suivre ta progression
sur plusieurs parties/stratégies.

Usage :
    python play_vs_bot.py [bot] [siege] [seed]

    bot   : B (greedy + heuristiques), E (greedy + heuristiques + tiebreak
            Gradient Boosting -- le bot le plus fort du dépôt, défaut),
            D (MCTS 150it sans tiebreak), F (MCTS 150it + tiebreak).
            Voir reference/MODELS.md pour ce que chacun signifie.
    siege : 0 ou 1, ton siège (défaut 0, tu commences).
    seed  : graine de partie (défaut aléatoire, pour rejouer EXACTEMENT
            la même partie avec une autre stratégie, fixe la graine).

Exemples :
    python play_vs_bot.py E              # contre le bot le plus fort
    python play_vs_bot.py B 1            # contre le greedy simple, tu joues en second
    python play_vs_bot.py E 0 12345      # graine fixe, pour rejouer la meme partie
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))

import value_policy as VP  # noqa: E402
from engine import DWELLERS, POSITIONS, TREES  # noqa: E402
from game import DWELLER, Game, TREE  # noqa: E402
from search import MCTS, greedy_action  # noqa: E402
from gen_technique_guide import FR_NAMES  # noqa: E402 -- reutilise la table de traduction

POS_FR = {"Top": "Haut", "Bottom": "Bas", "Left": "Gauche", "Right": "Droite"}


def fr(name):
    return FR_NAMES.get(name, POS_FR.get(name, name))


TREE_NAMES = [fr(t.name) for t in TREES]
DWELLER_NAMES = [fr(d.name) for d in DWELLERS]
POSITIONS_FR = [fr(p) for p in POSITIONS]

LOG_PATH = Path(__file__).resolve().parent / "human_vs_bot_log.jsonl"

BOT_LABELS = {
    "B": "B (greedy + heuristiques)",
    "E": "E (greedy + heuristiques + tiebreak GB)",
    "D": "D (MCTS 150it, sans tiebreak)",
    "F": "F (MCTS 150it + tiebreak GB)",
}


# ---------------------------------------------------------------------------
# Affichage (mêmes conventions que run_narrated_hybrid.py)
# ---------------------------------------------------------------------------

def card_label(card):
    if card[0] == TREE:
        return TREE_NAMES[card[1]]
    h1, h2 = card[1], card[2]
    return f"{DWELLER_NAMES[h1[0]]}/{DWELLER_NAMES[h2[0]]}"


def hand_label(hand):
    return ", ".join(card_label(c) for c in hand) if hand else "(vide)"


def action_label(action):
    if action[0] == "draw":
        return "Piocher"
    if action[0] == "skip_effect":
        return "Passer (décliner l'effet en attente)"
    if action[0] == "cave_discard":
        return f"Grotte : envoyer {action[1]} carte(s)"
    if action[0] == "free_dweller":
        _, did, tree_idx, pos = action
        return f"Pose GRATUITE {DWELLER_NAMES[did]} en {POSITIONS_FR[pos]} (arbre #{tree_idx})"
    if action[0] == "tree":
        return f"Planter {TREE_NAMES[action[1]]}"
    _, did, tree_idx, pos = action
    return f"Poser {DWELLER_NAMES[did]} en {POSITIONS_FR[pos]} (arbre #{tree_idx})"


def forest_summary(forest):
    trees = ", ".join(f"{TREE_NAMES[i]}×{forest.tree_count[i]}"
                       for i in range(len(TREE_NAMES)) if forest.tree_count[i])
    dwellers = ", ".join(f"{DWELLER_NAMES[i]}×{forest.dweller_count[i]}"
                          for i in range(len(DWELLER_NAMES)) if forest.dweller_count[i])
    return trees or "(aucun arbre)", dwellers or "(aucun habitant)"


def print_state(game, human_seat):
    bot_seat = 1 - human_seat
    scores = game.scores()
    me = game.players[human_seat]
    opp = game.players[bot_seat]
    print()
    print(f"===== Tour (score toi={scores[human_seat]} / bot={scores[bot_seat]}) =====")
    print(f"Deck restant : {len(game.deck)}   Clairière : {hand_label(game.clearing)}   "
          f"Hivers vus : {game.winters_seen}   Grotte : {me.forest.cave} pts")
    print(f"Ta main ({len(me.hand)}) : {hand_label(me.hand)}")
    t, d = forest_summary(me.forest)
    print(f"Ta forêt : arbres [{t}]  habitants [{d}]")
    t, d = forest_summary(opp.forest)
    print(f"Forêt adverse : arbres [{t}]  habitants [{d}]  (main cachée, {len(opp.hand)} cartes)")
    if game.pending_effect is not None:
        print(f"Effet en attente : {game.pending_effect}")


# ---------------------------------------------------------------------------
# Bots
# ---------------------------------------------------------------------------

class GreedyBot:
    def __init__(self, tiebreak=None):
        self.tiebreak = tiebreak

    def choose(self, game):
        return greedy_action(game, None, tiebreak=self.tiebreak)

    def on_move_applied(self, action):
        pass


class MCTSBotWrapper:
    def __init__(self, mcts):
        self.mcts = mcts

    def choose(self, game):
        return self.mcts.choose(game)

    def on_move_applied(self, action):
        self.mcts.advance(action)


def make_bot(name, seat, seed):
    if name == "B":
        return GreedyBot()
    if name == "E":
        return GreedyBot(tiebreak=VP.make_pairwise_gbm_tiebreak())
    if name == "D":
        mcts = MCTS(observer=seat, iterations=150, seed=seed, rollout_depth=40,
                    leaf_eval=VP.make_pairwise_hybrid_leaf_eval(short_rollout_depth=10, seed=seed))
        return MCTSBotWrapper(mcts)
    if name == "F":
        mcts = MCTS(observer=seat, iterations=150, seed=seed, rollout_depth=40,
                    leaf_eval=VP.make_pairwise_hybrid_leaf_eval(short_rollout_depth=10, seed=seed),
                    tiebreak=VP.make_pairwise_gbm_tiebreak())
        return MCTSBotWrapper(mcts)
    raise ValueError(f"bot inconnu : {name} (attendu B, D, E ou F)")


# ---------------------------------------------------------------------------
# Journal des parties
# ---------------------------------------------------------------------------

def log_result(bot_name, human_seat, human_score, bot_score, result, seed, turns):
    record = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bot": bot_name, "human_seat": human_seat,
        "human_score": human_score, "bot_score": bot_score,
        "result": result, "seed": seed, "turns": turns,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_stats():
    if not LOG_PATH.exists():
        return
    records = [json.loads(line) for line in LOG_PATH.read_text().splitlines() if line.strip()]
    by_bot = {}
    for r in records:
        by_bot.setdefault(r["bot"], []).append(r)
    print()
    print("--- Historique (toutes tes parties) ---")
    for bot, rs in sorted(by_bot.items()):
        wins = sum(1 for r in rs if r["result"] == "victoire")
        losses = sum(1 for r in rs if r["result"] == "defaite")
        ties = sum(1 for r in rs if r["result"] == "nul")
        avg_diff = sum(r["human_score"] - r["bot_score"] for r in rs) / len(rs)
        print(f"  vs {BOT_LABELS.get(bot, bot):45s} : {wins}V {losses}D {ties}N sur {len(rs)}  "
              f"(écart moyen {avg_diff:+.1f})")


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------

def main():
    bot_name = sys.argv[1].upper() if len(sys.argv) > 1 else "E"
    human_seat = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else random.randint(0, 10**6)

    if bot_name not in BOT_LABELS:
        print(f"Bot inconnu : {bot_name}. Choix possibles : {', '.join(BOT_LABELS)}")
        return
    if human_seat not in (0, 1):
        print("Le siège doit être 0 ou 1.")
        return

    bot_seat = 1 - human_seat
    game = Game(n_players=2, seed=seed)
    bot = make_bot(bot_name, bot_seat, seed)

    print(f"Partie contre {BOT_LABELS[bot_name]} -- tu es le joueur {human_seat}, seed={seed}.")
    print("Entre le numéro du coup à jouer à chaque tour. 'q' pour abandonner.")

    turns = 0
    aborted = False
    while not game.over and turns < 600:
        cur = game.current
        if cur == human_seat:
            print_state(game, human_seat)
            actions = game.legal_actions()
            for i, a in enumerate(actions):
                print(f"  [{i}] {action_label(a)}")
            choice = input("Ton coup > ").strip()
            if choice.lower() in ("q", "quit"):
                print("Partie abandonnée.")
                aborted = True
                break
            try:
                idx = int(choice)
                action = actions[idx]
            except (ValueError, IndexError):
                print("Choix invalide, réessaie.")
                continue
        else:
            action = bot.choose(game)
            print(f"[{bot_name} joue] {action_label(action)}")

        game.apply(action)
        bot.on_move_applied(action)
        turns += 1

    if aborted:
        return

    print()
    print("=== Partie terminée ===")
    scores = game.scores()
    print(f"Toi (joueur {human_seat})        : {scores[human_seat]} pts")
    print(f"{bot_name} (joueur {bot_seat})            : {scores[bot_seat]} pts")
    if scores[human_seat] > scores[bot_seat]:
        result = "victoire"
        print("Tu as gagné !")
    elif scores[human_seat] < scores[bot_seat]:
        result = "defaite"
        print("Le bot gagne.")
    else:
        result = "nul"
        print("Égalité.")

    log_result(bot_name, human_seat, scores[human_seat], scores[bot_seat], result, seed, turns)
    print_stats()


if __name__ == "__main__":
    main()
