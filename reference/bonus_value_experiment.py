"""Valeur en points du déclenchement du bonus jumelles.

Pour chaque pose (dans une vraie partie greedy) d'un dweller qui a un
effet conditionné au bonus (DRAW_IF_BONUS / DWELLER_PLAY_FREE_IF_BONUS /
REPLAY_IF_BONUS), et où une carte du bon symbole est disponible en main
pour payer :

  - Branche A ("bonus") : on force le paiement à inclure une carte du bon
    symbole -> déclenche l'effet.
  - Branche B ("sans bonus") : on force le paiement à l'éviter -> le même
    coût est payé, mais l'effet ne se déclenche pas.

Puis on complète chaque branche par K rollouts (même politique, mêmes
graines de rng appariées pour réduire la variance non liée au choix), et
on compare le score final moyen. La différence = valeur en points du
bonus pour CE coup précis, dans CE contexte de partie précis.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine as E
import game as G
import search as S

BONUS_TABLES = (E.DRAW_IF_BONUS.keys() | {} .keys())  # placeholder, complété ci-dessous


def is_bonus_eligible(did):
    return (did in E.DRAW_IF_BONUS
            or did in E.DWELLER_PLAY_FREE_IF_BONUS
            or did in E.REPLAY_IF_BONUS)


def choose_payment_avoid(hand, cost, avoided_symbol):
    """Comme choose_payment, mais évite délibérément de défausser une
    carte du symbole donné quand c'est possible (contrefactuel "sans
    bonus"), en gardant le même coût payé.
    """
    if cost <= 0:
        return []

    def best_cost(i):
        c = hand[i]
        if c[0] == G.DWELLER:
            return min(G.card_cost(c, 0), G.card_cost(c, 1))
        return E.TREE_COST[c[1]]

    def matches(i):
        c = hand[i]
        if c[0] != G.DWELLER:
            return False
        return (G.card_symbol(c, 0) == avoided_symbol
                or G.card_symbol(c, 1) == avoided_symbol)

    ranked = sorted(
        range(len(hand)),
        key=lambda i: (1 if matches(i) else 0, -best_cost(i)),  # non-matches d'abord
    )
    return ranked[:cost]


def rollout_avg_score(state, rng, player, k=5, depth=20):
    total = 0.0
    for _ in range(k):
        s = state.clone()
        r = random.Random(rng.random())
        scores = S.rollout(s, r, max_moves=depth)
        total += scores[player]
    return total / k


def run_experiment(n_games=8, k_rollouts=5, depth=20):
    results = []
    for gi in range(n_games):
        seed = 5000 + gi
        g = G.Game(n_players=2, seed=seed)
        turns = 0
        while not g.over and turns < 300:
            cur = g.current
            action = S.greedy_action(g, random.Random(seed * 1000 + turns))
            if action[0] == "dweller" and g.pending_effect is None:
                _, did, tree_idx, pos = action
                if is_bonus_eligible(did):
                    found = g._find_card(g.players[cur].hand, action)
                    if found is not None:
                        _, half_index, symbol = found
                        card = g.players[cur].hand[found[0]]
                        cost = G.card_cost(card, half_index)
                        remaining_hand = g.players[cur].hand[:found[0]] + g.players[cur].hand[found[0]+1:]
                        has_match = any(
                            G.card_symbol(c, 0) == symbol or G.card_symbol(c, 1) == symbol
                            for c in remaining_hand if c[0] == G.DWELLER
                        )
                        if cost >= 1 and has_match and symbol is not None:
                            base_state = g.clone()
                            rng_pair = random.Random(seed * 7919 + turns)

                            state_bonus = base_state.clone()
                            payment_bonus = G.choose_payment(remaining_hand, cost, preferred_symbol=symbol)
                            state_bonus.apply(action, payment=payment_bonus)
                            # si un pending_effect s'est ouvert (batch 2b), on le skip
                            # pour isoler la valeur immédiate + les pioches, sans
                            # ouvrir un deuxième axe de décision dans l'expérience.
                            if state_bonus.pending_effect is not None:
                                state_bonus.apply(("skip_effect",))

                            state_no_bonus = base_state.clone()
                            payment_no_bonus = choose_payment_avoid(remaining_hand, cost, symbol)
                            state_no_bonus.apply(action, payment=payment_no_bonus)

                            avg_bonus = rollout_avg_score(state_bonus, random.Random(rng_pair.random()), cur, k_rollouts, depth)
                            avg_no_bonus = rollout_avg_score(state_no_bonus, random.Random(rng_pair.random()), cur, k_rollouts, depth)

                            results.append({
                                "card": E.DWELLER_NAME[did],
                                "seed": seed,
                                "turn": turns,
                                "value": avg_bonus - avg_no_bonus,
                            })
            g.apply(action)
            turns += 1
    return results


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    results = run_experiment(n_games=n)
    print(f"{len(results)} instances de choix bonus/pas-bonus mesurées sur {n} parties")
    if results:
        avg = sum(r["value"] for r in results) / len(results)
        print(f"Valeur moyenne du bonus (points de score final, rollout tronqué) : {avg:+.2f}")
        from collections import defaultdict
        by_card = defaultdict(list)
        for r in results:
            by_card[r["card"]].append(r["value"])
        print()
        print("Par carte :")
        for card, vals in sorted(by_card.items(), key=lambda kv: -sum(kv[1])/len(kv[1])):
            print(f"  {card:20s} moy={sum(vals)/len(vals):+6.2f}  (n={len(vals)})")
