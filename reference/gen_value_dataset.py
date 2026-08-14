"""Génère un dataset (features d'état -> gain restant réel) par self-play.

Deux sources de trajectoires, mélangées dans un même dataset (session du
14/08, pour corriger le décalage de distribution constaté en branchant le
premier modèle dans MCTS : un modèle entraîné uniquement sur du greedy
propre se comporte mal face aux états que MCTS explore réellement) :
  - greedy bruité (epsilon élevé) : plus de variance que le greedy pur,
    moins cher que du MCTS complet.
  - MCTS à rollout classique (PAS le modèle de valeur qu'on entraîne, pour
    éviter la boucle circulaire) : donne des états réellement visités
    par une recherche arborescente, la distribution qui compte pour la
    tâche finale.

Sortie : reference/value_dataset.npz (X, y), rechargeable directement par
train_value_model.py.
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


def _collect(g, seed, policy_fn, sample_every=1):
    """Joue une partie avec `policy_fn(game, rng) -> action`, retourne les
    (feats, score_au_snapshot) et le score final par joueur."""
    snapshots = []
    turns = 0
    while not g.over and turns < 400:
        cur = g.current
        if turns % sample_every == 0:
            feats = F.extract_features(g, cur)
            score_now = g.scores()[cur]
            snapshots.append((cur, score_now, feats))
        action = policy_fn(g, random.Random(seed * 1000 + turns))
        g.apply(action)
        turns += 1
    return snapshots, g.scores()


def generate_noisy_greedy(n_games, seed0, epsilon=0.4, sample_every=1):
    X, y = [], []
    for gi in range(n_games):
        seed = seed0 + gi
        g = G.Game(n_players=2, seed=seed)
        policy = lambda game, rng: S.greedy_action(game, rng, epsilon=epsilon)
        snapshots, final_scores = _collect(g, seed, policy, sample_every)
        for player_index, score_now, feats in snapshots:
            X.append(feats)
            y.append(final_scores[player_index] - score_now)
    return X, y


def generate_mcts_rollout(n_games, seed0, iterations=150, rollout_depth=25, sample_every=2):
    """MCTS à rollout classique (pas leaf_eval) : les deux joueurs sont
    des bots MCTS, pour visiter les états réellement rencontrés en
    recherche arborescente. Plus cher, donc `sample_every` sous-échantillonne
    et `n_games` reste modeste par rapport au greedy bruité.
    """
    X, y = [], []
    for gi in range(n_games):
        seed = seed0 + gi
        g = G.Game(n_players=2, seed=seed)
        bots = [S.MCTS(observer=i, iterations=iterations, seed=seed + i,
                        rollout_depth=rollout_depth) for i in range(2)]
        snapshots = []
        turns = 0
        while not g.over and turns < 400:
            cur = g.current
            if turns % sample_every == 0:
                feats = F.extract_features(g, cur)
                score_now = g.scores()[cur]
                snapshots.append((cur, score_now, feats))
            action = bots[cur].choose(g)
            g.apply(action)
            for b in bots:
                b.advance(action)
            turns += 1
        final_scores = g.scores()
        for player_index, score_now, feats in snapshots:
            X.append(feats)
            y.append(final_scores[player_index] - score_now)
    return X, y


if __name__ == "__main__":
    n_noisy = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_mcts = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    X_all, y_all = [], []

    Xg, yg = generate_noisy_greedy(n_noisy, seed0=10000)
    X_all += Xg
    y_all += yg
    print(f"greedy bruité : {n_noisy} parties -> {len(Xg)} lignes")

    Xm, ym = generate_mcts_rollout(n_mcts, seed0=20000)
    X_all += Xm
    y_all += ym
    print(f"MCTS rollout  : {n_mcts} parties -> {len(Xm)} lignes")

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)
    out = Path(__file__).resolve().parent / "value_dataset.npz"
    np.savez_compressed(out, X=X, y=y, feature_names=np.array(F.FEATURE_NAMES))
    print(f"total : {X.shape[0]} lignes, {X.shape[1]} features -> {out}")
