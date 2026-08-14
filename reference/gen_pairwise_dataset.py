"""Génère un dataset PAIRÉ (diff de features -> diff de gain réel), pour
corriger le défaut diagnostiqué du modèle absolu (`gen_value_dataset.py` +
`train_value_model.py`) : son MAE (~12 pts) est plus grand que l'écart de
valeur entre deux coups candidats à un même nœud (~9 pts mesurés), donc
MCTS ne peut pas s'en servir pour départager des branches soeurs.

Idée : au lieu de regresser le gain final absolu (bruit dominé par tout ce
qui reste aléatoire dans la partie -- pioche, adversaire), on regresse la
DIFFÉRENCE de gain entre deux coups candidats joués depuis LE MÊME état,
avec la MÊME suite de pioche (common random numbers). `Game.clone()` copie
la liste `deck` telle quelle (pas un rng partagé qui la régénère) : deux
clones d'un même état tirent donc la même séquence de cartes tant qu'ils
n'ont pas consommé un nombre différent de pioches. Cela annule une bonne
partie du bruit partagé entre les deux branches et isole l'effet du coup.

Le modèle appris sur ces diffs est un Ridge SANS intercept sur
`feat(après A) - feat(après B)` -> `gain(A) - gain(B)`. Comme le modèle est
linéaire, ses poids `w` définissent directement une fonction de valeur par
état à constante additive près : `value(état) = w . feat(état)`. Cette
constante n'a pas besoin d'être calibrée pour servir de `leaf_eval` : seul
l'ORDRE relatif entre joueurs compte pour `reward_vector` (rang normalisé),
et un biais constant appliqué symétriquement aux deux joueurs ne change pas
ce rang.

Sortie : reference/pairwise_dataset.npz (Xd, yd).
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


def _candidate_actions(state, k, rng):
    actions = [a for a in state.legal_actions() if a[0] not in ("draw", "skip_effect")]
    if len(actions) <= 1:
        return []
    if len(actions) > k:
        actions = rng.sample(actions, k)
    return actions


def _rollout_score(state, seed, depth, observer):
    rng = random.Random(seed)
    moves = 0
    while not state.over and moves < depth:
        acts = state.legal_actions()
        if len(acts) == 1:
            state.apply(acts[0])
        else:
            state.apply(S.greedy_action(state, rng, S.ROLLOUT_EPSILON, S.ROLLOUT_CANDIDATES))
        moves += 1
    return state.scores()[observer]


def generate_pairs(n_games, seed0, k_candidates=4, depth=20, sample_every=4,
                    trajectory_epsilon=0.4):
    Xd, yd = [], []
    for gi in range(n_games):
        seed = seed0 + gi
        g = G.Game(n_players=2, seed=seed)
        traj_rng = random.Random(seed)
        turns = 0
        while not g.over and turns < 400:
            if turns % sample_every == 0:
                observer = g.current
                cand_rng = random.Random(seed * 7919 + turns)
                candidates = _candidate_actions(g, k_candidates, cand_rng)
                if len(candidates) >= 2:
                    common_seed = seed * 104729 + turns
                    scores_by_action = {}
                    feats_by_action = {}
                    for a in candidates:
                        branch = g.clone()
                        branch.apply(a)
                        # features sur l'état juste après le coup, avant le
                        # rollout : cohérent avec l'usage en leaf_eval (état
                        # déjà avancé par sélection/expansion). Le score brut
                        # est ajouté à part (pas dans extract_features, qui
                        # l'exclut volontairement pour le modèle absolu -- ici
                        # il est sans risque de fuite d'horloge : les deux
                        # candidats partent du MÊME état/tour, donc le DIFF
                        # de ce terme encode exactement le delta de score
                        # immédiat entre les deux coups, déjà calculé par le
                        # moteur, jamais bruité).
                        raw_score = branch.scores()[observer]
                        feats_by_action[a] = F.extract_features(branch, observer) + [raw_score]
                        scores_by_action[a] = _rollout_score(
                            branch, common_seed, depth, observer)
                    for i in range(len(candidates)):
                        for j in range(i + 1, len(candidates)):
                            ai, aj = candidates[i], candidates[j]
                            fi = np.asarray(feats_by_action[ai], dtype=np.float32)
                            fj = np.asarray(feats_by_action[aj], dtype=np.float32)
                            Xd.append(fi - fj)
                            yd.append(scores_by_action[ai] - scores_by_action[aj])
            action = S.greedy_action(g, traj_rng, epsilon=trajectory_epsilon)
            g.apply(action)
            turns += 1
    return Xd, yd


if __name__ == "__main__":
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 150

    Xd, yd = generate_pairs(n_games, seed0=30000)
    Xd = np.asarray(Xd, dtype=np.float32)
    yd = np.asarray(yd, dtype=np.float32)
    feature_names = list(F.FEATURE_NAMES) + ["raw_score"]
    out = Path(__file__).resolve().parent / "pairwise_dataset.npz"
    np.savez_compressed(out, Xd=Xd, yd=yd, feature_names=np.array(feature_names))
    print(f"{n_games} parties -> {Xd.shape[0]} paires, {Xd.shape[1]} features -> {out}")
