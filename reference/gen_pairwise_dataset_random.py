"""Variante de gen_pairwise_dataset.py : trajectoire ET rollout de label
100% aléatoires (pas d'appel a S.greedy_action nulle part), pour tester
l'hypothese de Mehdi (19/08) -- les tentatives precedentes (voir
reference/MODELS.md) generent toutes leurs donnees via de l'auto-jeu
greedy (ou MCTS lui-meme en bootstrap gen1), donc heritent des heuristiques
(`choose_draw_source`, urgence Clairiere, ROLLOUT_EPSILON=0.25 fixe) meme
quand elles cherchent a s'en affranchir. Un jeu aleatoire pur est quasi
gratuit (~2ms/partie contre les 20-40ms d'une partie greedy) : on peut
generer un dataset bien plus gros (milliers de parties) sans biais
heuristique d'aucune sorte.

Garde EXACTEMENT le meme dispositif anti-bruit que gen_pairwise_dataset.py
(candidats au meme noeud, k_rollout rollouts a graines communes, split par
partie) -- seule la source des donnees change, pas la methode de mesure.
Objectif du test : si le plafond observe sur 7 tentatives precedentes
vient bien d'un biais de distribution (comme l'hypothese de Mehdi le
suppose) plutot que d'un plafond de features (diagnostique le 18/08 par
le reponderation des paires serrees, sans effet), ce dataset devrait
produire un gating meilleur. S'il ne fait pas mieux que k=5 (9/30), ca
confirme le plafond de features plutot que le biais de distribution.

Usage : python reference/gen_pairwise_dataset_random.py [n_games] [k_rollout]
"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import game as G
import features as F

HERE = Path(__file__).resolve().parent


def _candidate_actions(state, k, rng):
    actions = [a for a in state.legal_actions() if a[0] not in ("draw", "skip_effect")]
    if len(actions) <= 1:
        return []
    if len(actions) > k:
        actions = rng.sample(actions, k)
    return actions


def _random_rollout_score(state, seed, depth, observer):
    rng = random.Random(seed)
    moves = 0
    while not state.over and moves < depth:
        acts = state.legal_actions()
        state.apply(acts[0] if len(acts) == 1 else rng.choice(acts))
        moves += 1
    return state.scores()[observer]


def generate_pairs(n_games, seed0, k_candidates=4, depth=20, sample_every=4,
                    k_rollout=5, progress_features=False):
    Xd, yd, Xa, Xb, game_idx = [], [], [], [], []
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
                    progress = [turns, len(g.deck), g.winters_seen] if progress_features else []
                    scores_by_action = {}
                    feats_by_action = {}
                    for a in candidates:
                        branch = g.clone()
                        branch.apply(a)
                        raw_score = branch.scores()[observer]
                        feats_by_action[a] = F.extract_features(branch, observer) + [raw_score] + progress
                        vals = []
                        for k in range(k_rollout):
                            common_seed = seed * 104729 + turns * 1000 + k
                            vals.append(_random_rollout_score(
                                branch.clone(), common_seed, depth, observer))
                        scores_by_action[a] = sum(vals) / k_rollout
                    for i in range(len(candidates)):
                        for j in range(i + 1, len(candidates)):
                            ai, aj = candidates[i], candidates[j]
                            fi = np.asarray(feats_by_action[ai], dtype=np.float32)
                            fj = np.asarray(feats_by_action[aj], dtype=np.float32)
                            Xd.append(fi - fj)
                            yd.append(scores_by_action[ai] - scores_by_action[aj])
                            Xa.append(fi)
                            Xb.append(fj)
                            game_idx.append(gi)
            acts = g.legal_actions()
            g.apply(acts[0] if len(acts) == 1 else traj_rng.choice(acts))
            turns += 1
    return Xd, yd, Xa, Xb, game_idx


if __name__ == "__main__":
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    k_rollout = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    Xd, yd, Xa, Xb, game_idx = generate_pairs(n_games, seed0=70000, k_rollout=k_rollout)
    Xd = np.asarray(Xd, dtype=np.float32)
    yd = np.asarray(yd, dtype=np.float32)
    Xa = np.asarray(Xa, dtype=np.float32)
    Xb = np.asarray(Xb, dtype=np.float32)
    game_idx = np.asarray(game_idx, dtype=np.int32)
    feature_names = list(F.FEATURE_NAMES) + ["raw_score"]
    out = HERE / "pairwise_dataset_random.npz"
    np.savez_compressed(out, Xd=Xd, yd=yd, Xa=Xa, Xb=Xb, game_idx=game_idx,
                         feature_names=np.array(feature_names))
    print(f"{n_games} parties (100% aleatoires) -> {Xd.shape[0]} paires, "
          f"{Xd.shape[1]} features -> {out}")
