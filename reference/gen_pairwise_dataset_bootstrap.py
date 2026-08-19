"""Génération 1 de la piste bootstrap MCTS façon AlphaZero (voir la section
"Piste future (non implémentée)" de reference/MODELS.md avant ce commit --
ce fichier l'implémente).

Toutes les tentatives précédentes de `gen_pairwise_dataset.py` échantillonnent
les paires d'entraînement le long d'une trajectoire d'AUTO-JEU GREEDY
(`S.greedy_action` bruité). Mais MCTS, une fois en jeu, ne suit pas du
greedy : son terme d'exploration UCT le pousse à visiter des branches qu'un
joueur greedy n'atteint jamais. Le modèle n'a jamais vu ces états à
l'entraînement. Idée du bootstrap : échantillonner le long d'une trajectoire
jouée par MCTS lui-même (avec le modèle courant en `leaf_eval`), pas par
greedy -- pour faire coïncider distribution d'entraînement et distribution
d'usage. Correction minimale (option 1 de MODELS.md) : le mécanisme
d'étiquetage des paires (candidats à un même nœud, k rollouts à seed
commune) est repris tel quel de `gen_pairwise_dataset.py` ; seule la façon
dont la partie AVANCE entre deux points d'échantillonnage change.

IMPORTANT -- ce script ne touche PAS aux heuristiques de `greedy_action`
(`choose_draw_source` cible carte forte, `CLEARING_URGENCY_BONUS`) : elles
restent actives par défaut partout (rollouts de labellisation, rollout
court du leaf_eval hybride utilisé par les bots MCTS de self-play). Une
session précédente a débranché ces heuristiques pour générer des données
d'entraînement et observé un effondrement du classement du bot entraîné
dessus (voir reference/MODELS.md, "*_pre_clairiere_urgence*",
"*_pre_clairiere_forte*") -- ne pas répéter cette erreur ici.

Coût : MCTS(50 it.) est nettement plus lent que greedy par décision. Ce
script est pensé pour une PREMIÈRE génération volontairement petite (30
parties, 50 itérations), comme recommandé dans MODELS.md, pas pour
remplacer `gen_pairwise_dataset.py` à l'échelle production.

Sortie : reference/pairwise_dataset_bootstrap_gen1.npz (même schéma que
pairwise_dataset.npz). Le dataset greedy existant n'est pas touché ; le
modèle réentraîné sur cette sortie doit battre B en tête-à-tête
(`bench_heuristics.py`) avant d'être promu comme modèle par défaut --
gating, jamais sur la seule base d'une métrique offline (leçon de la
section précédente de MODELS.md).
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import game as G
import search as S
import features as F
import value_policy as VP
from gen_pairwise_dataset import _candidate_actions, _rollout_score


def generate_pairs_bootstrap(n_games, seed0, mcts_iterations=50, k_candidates=4,
                              depth=20, sample_every=4, k_rollout=5,
                              trajectory_model_path=None, progress_features=False,
                              verbose=False):
    """Comme `generate_pairs` (gen_pairwise_dataset.py), mais la trajectoire
    est jouée par deux `search.MCTS` (un par siège, `leaf_eval` = modèle
    pairwise courant via `trajectory_model_path`) plutôt qu'un auto-jeu
    greedy bruité. Voir la docstring du module pour le pourquoi.
    """
    Xd, yd, Xa, Xb, game_idx = [], [], [], [], []
    for gi in range(n_games):
        t0 = time.time()
        seed = seed0 + gi
        g = G.Game(n_players=2, seed=seed)
        bots = {
            s: S.MCTS(observer=s, iterations=mcts_iterations, seed=seed * 31 + s,
                      rollout_depth=40,
                      leaf_eval=VP.make_pairwise_hybrid_leaf_eval(
                          model_path=trajectory_model_path, seed=seed * 31 + s))
            for s in range(2)
        }
        turns = 0
        while not g.over and turns < 400:
            if turns % sample_every == 0:
                observer = g.current
                cand_rng = __import__("random").Random(seed * 7919 + turns)
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
                            vals.append(_rollout_score(
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
            seat = g.current
            action = bots[seat].choose(g)
            g.apply(action)
            for b in bots.values():
                b.advance(action)
            turns += 1
        if verbose:
            print(f"  partie {gi+1}/{n_games} : {turns} tours, {time.time()-t0:.1f}s, "
                  f"{len(Xd)} paires cumulées", flush=True)
    return Xd, yd, Xa, Xb, game_idx


if __name__ == "__main__":
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    mcts_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    k_rollout = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    Xd, yd, Xa, Xb, game_idx = generate_pairs_bootstrap(
        n_games, seed0=90000, mcts_iterations=mcts_iterations, k_rollout=k_rollout,
        verbose=True)
    Xd = np.asarray(Xd, dtype=np.float32)
    yd = np.asarray(yd, dtype=np.float32)
    Xa = np.asarray(Xa, dtype=np.float32)
    Xb = np.asarray(Xb, dtype=np.float32)
    game_idx = np.asarray(game_idx, dtype=np.int32)
    feature_names = list(F.FEATURE_NAMES) + ["raw_score"]
    out = Path(__file__).resolve().parent / "pairwise_dataset_bootstrap_gen1.npz"
    np.savez_compressed(out, Xd=Xd, yd=yd, Xa=Xa, Xb=Xb, game_idx=game_idx,
                         feature_names=np.array(feature_names))
    print(f"{n_games} parties MCTS({mcts_iterations}it) -> {Xd.shape[0]} paires, "
          f"{Xd.shape[1]} features -> {out}")
