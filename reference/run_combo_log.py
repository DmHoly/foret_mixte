"""Batch de parties instrumentées -> reference/combo_log.jsonl

Un coup = une ligne JSON. Ne modifie pas engine.py/game.py, tout est fait
par instrumentation externe (monkeypatch temporaire), comme dans les
scripts d'analyse de la session du 14/08.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine as E
import game as G
import search as S

OUT = Path(__file__).resolve().parent / "combo_log.jsonl"


def card_label(action):
    if action[0] == "tree":
        return E.TREE_NAME[action[1]]
    if action[0] in ("dweller", "free_dweller"):
        return E.DWELLER_NAME[action[1]]
    return action[0]


def run_one(seed, policy_name, policy_fn, writer):
    g = G.Game(n_players=2, seed=seed)
    turns = 0
    while not g.over and turns < 400:
        cur = g.current
        action = policy_fn(g, random.Random(seed * 1000 + turns))
        bonus_before = g.pending_effect is not None
        score_before = G.score_players([p.forest for p in g.players])[cur]
        cost = 0
        if action[0] in ("tree", "dweller"):
            # coût réel : recalculé après coup via la variation de main
            hand_before = len(g.players[cur].hand)
        g.apply(action)
        score_after = G.score_players([p.forest for p in g.players])[cur]

        if action[0] in ("tree", "dweller", "free_dweller", "cave_discard"):
            record = {
                "seed": seed,
                "turn": turns,
                "policy": policy_name,
                "player": cur,
                "action_kind": action[0],
                "card": card_label(action),
                "score_delta": score_after - score_before,
                "bonus_context": bonus_before,  # ce coup résolvait un pending_effect
            }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
        turns += 1


def main(n_games=20):
    with open(OUT, "w", encoding="utf-8") as writer:
        for i in range(n_games):
            seed = 4000 + i
            run_one(seed, "greedy", S.greedy_action, writer)
    print(f"{n_games} parties loguées dans {OUT}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(n)
