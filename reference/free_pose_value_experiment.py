"""Valeur en points de l'utilisation effective d'une pose gratuite
(pending_effect de type filtre, batch 2b), une fois le bonus déjà
déclenché : comparer "jouer une carte gratuite" vs "skip_effect", à
partir du même état de jeu (même main, mêmes options), complété par K
rollouts appariés.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine as E
import game as G
import search as S


def rollout_avg_score(state, rng, player, k=5, depth=20):
    total = 0.0
    for _ in range(k):
        s = state.clone()
        r = random.Random(rng.random())
        scores = S.rollout(s, r, max_moves=depth)
        total += scores[player]
    return total / k


def run_experiment(n_games=20, k_rollouts=8, depth=20):
    results = []
    for gi in range(n_games):
        seed = 6000 + gi
        g = G.Game(n_players=2, seed=seed)
        turns = 0
        while not g.over and turns < 300:
            cur = g.current
            action = S.greedy_action(g, random.Random(seed * 1000 + turns))
            g.apply(action)
            # juste après l'apply, si un pending_effect de type filtre s'est
            # ouvert (batch 2b), on a une vraie décision "l'utiliser ou pas"
            if (g.pending_effect is not None
                    and g.pending_effect[0] not in ("cave_choice", "play_chain")
                    and g.current == cur):
                legal = g.legal_actions()
                free_actions = [a for a in legal if a[0] == "free_dweller"]
                if free_actions:
                    base_state = g.clone()
                    rng_pair = random.Random(seed * 7919 + turns)

                    state_use = base_state.clone()
                    # politique simple : choisit la pose gratuite qui maximise
                    # le score immédiat (proche d'un greedy sur l'action).
                    best_a, best_gain = None, -1e9
                    for a in free_actions:
                        s2 = base_state.clone()
                        before = s2.scores()[cur]
                        s2.apply(a)
                        gain = s2.scores()[cur] - before
                        if gain > best_gain:
                            best_gain, best_a = gain, a
                    state_use.apply(best_a)

                    state_skip = base_state.clone()
                    state_skip.apply(("skip_effect",))

                    avg_use = rollout_avg_score(state_use, random.Random(rng_pair.random()), cur, k_rollouts, depth)
                    avg_skip = rollout_avg_score(state_skip, random.Random(rng_pair.random()), cur, k_rollouts, depth)

                    results.append({
                        "card": E.DWELLER_NAME[best_a[1]],
                        "seed": seed,
                        "turn": turns,
                        "value": avg_use - avg_skip,
                    })
            turns += 1
    return results


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    results = run_experiment(n_games=n)
    print(f"{len(results)} instances 'pose gratuite vs skip' mesurées sur {n} parties")
    if results:
        avg = sum(r["value"] for r in results) / len(results)
        print(f"Valeur moyenne d'utiliser la pose gratuite (points, rollout tronqué) : {avg:+.2f}")
        from collections import defaultdict
        by_card = defaultdict(list)
        for r in results:
            by_card[r["card"]].append(r["value"])
        print()
        print("Par carte posée gratuitement :")
        for card, vals in sorted(by_card.items(), key=lambda kv: -sum(kv[1])/len(kv[1])):
            print(f"  {card:20s} moy={sum(vals)/len(vals):+6.2f}  (n={len(vals)})")
