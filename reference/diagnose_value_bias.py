"""Diagnostic : le modèle de valeur est-il biaisé (pas juste bruité) sur
les états hors distribution d'entraînement (branches spéculatives que
MCTS explore, jamais visitées par du greedy) ?

Pour deux groupes d'états :
  - "on_policy"  : atteints en suivant greedy depuis le début de partie
  - "off_policy" : mêmes parties, mais on dévie à un moment donné avec
    quelques coups légaux aléatoires (simule ce que MCTS explore et que
    greedy n'atteindrait jamais)

... on compare la prédiction du modèle à une estimation de référence
(moyenne de K rollouts, la même méthode qui marchait en pratique), et on
regarde si l'erreur (prédiction - référence) a un biais directionnel
différent entre les deux groupes.
"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import game as G
import search as S
import features as F
import value_policy as VP


def reference_value(state, player_index, rng, k=8, depth=250):
    """Moyenne de K rollouts jusqu'à (quasi) la fin réelle de partie, pour
    matcher l'horizon sur lequel le modèle a été entraîné (gain jusqu'à la
    fin de partie, pas un rollout tronqué à 25-35 coups comme dans MCTS).
    k réduit par rapport à la première version car chaque rollout est
    maintenant beaucoup plus long.
    """
    total = 0.0
    for _ in range(k):
        s = state.clone()
        scores = S.rollout(s, random.Random(rng.random()), max_moves=depth)
        total += scores[player_index]
    return total / k


def sample_states(n_games=15, seed0=40000, off_policy_deviation=4):
    on_policy, off_policy = [], []
    for gi in range(n_games):
        seed = seed0 + gi
        g = G.Game(n_players=2, seed=seed)
        turns = 0
        # avance un nombre aléatoire de coups greedy pour varier le point
        # d'échantillonnage à chaque partie
        n_pre = random.Random(seed).randint(10, 60)
        while not g.over and turns < n_pre:
            g.apply(S.greedy_action(g, random.Random(seed * 1000 + turns)))
            turns += 1
        if g.over:
            continue

        # groupe on-policy : état actuel, atteint uniquement par greedy
        cur = g.current
        on_policy.append((g.clone(), cur))

        # groupe off-policy : quelques coups légaux aléatoires depuis le
        # même point de départ, pour simuler une branche spéculative
        g_off = g.clone()
        rng = random.Random(seed + 9999)
        for _ in range(off_policy_deviation):
            if g_off.over:
                break
            actions = g_off.legal_actions()
            action = rng.choice(actions)
            g_off.apply(action)
        if not g_off.over:
            off_policy.append((g_off, g_off.current))

    return on_policy, off_policy


def evaluate_group(states, model, rng_seed=0):
    errors = []
    rng = random.Random(rng_seed)
    for state, player_index in states:
        pred = state.scores()[player_index] + VP.predict_remaining_gain(model, state, player_index)
        ref = reference_value(state, player_index, rng)
        errors.append(pred - ref)
    return np.array(errors)


if __name__ == "__main__":
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    VP._MODEL_CACHE.clear()
    model = VP.load_model()

    on_policy, off_policy = sample_states(n_games=n_games)
    print(f"{len(on_policy)} états on-policy, {len(off_policy)} états off-policy")

    err_on = evaluate_group(on_policy, model, rng_seed=1)
    err_off = evaluate_group(off_policy, model, rng_seed=2)

    print()
    print("Erreur (prédiction - référence rollout) :")
    print(f"  on-policy  : moyenne={err_on.mean():+7.2f}  écart-type={err_on.std():6.2f}  (n={len(err_on)})")
    print(f"  off-policy : moyenne={err_off.mean():+7.2f}  écart-type={err_off.std():6.2f}  (n={len(err_off)})")
