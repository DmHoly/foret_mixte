"""Génère un dataset pairé à partir d'AUTO-JEU MCTS, pas d'auto-jeu greedy
(voir reference/MODELS.md, section "Piste future : bootstrap MCTS façon
AlphaZero").

Différence avec gen_pairwise_dataset.py :

  - Trajectoire : chaque joueur est un `search.MCTS` persistant (arbre
    réutilisé d'un tour à l'autre, comme dans bench.py/bench_heuristics.py),
    pas `S.greedy_action`. Ça corrige le problème diagnostiqué sur 6
    tentatives précédentes : le modèle de valeur était entraîné sur des états
    visités par un joueur GREEDY, alors qu'en usage réel MCTS visite
    délibérément (terme d'exploration UCT) des branches qu'un greedy
    n'atteindrait jamais -- distribution d'entraînement et distribution
    d'usage ne coïncidaient pas.
  - Cible : au lieu de relancer un rollout séparé par candidat (bruité, need
    for k_rollout), on réutilise directement les statistiques de l'arbre de
    recherche que MCTS.choose() vient de calculer -- `child.value /
    child.visits` pour chaque action candidate au nœud racine, jugée par
    des centaines de simulations, pas un seul rollout tronqué. C'est
    l'option 2 documentée dans reference/MODELS.md, la plus proche
    d'AlphaZero (le modèle apprend à imiter le jugement de la recherche
    complète).

Une conséquence importante : `child.value/child.visits` est un score
RANG NORMALISÉ dans [0, 1] (voir `search.reward_vector`), pas un delta de
points comme dans gen_pairwise_dataset.py. Le dataset produit ici n'est
donc PAS directement comparable en échelle à pairwise_dataset.npz -- il a
son propre fichier de sortie (reference/bootstrap_dataset.npz) et son
propre modèle entraîné (reference/bootstrap_model.joblib), jamais
mélangés avec les fichiers "officiels" (voir reference/MODELS.md pour la
procédure de gating avant toute promotion).

Deux ajouts (session du 17/08, sur demande de Mehdi) :

  - **Cible de politique en plus de la valeur** : pour la même paire de
    candidats (ai, aj) à la même racine, on ajoute
    `yp = log(visites[ai]) - log(visites[aj])` -- la distribution de
    visites de la racine EST la politique améliorée façon AlphaZero
    (`π ∝ N(s,a)`), déjà calculée gratuitement par `MCTS.choose()`, jamais
    exploitée jusqu'ici. Entraînée séparément par `train_policy_model.py`,
    utilisée comme prior PUCT (`search.Node.uct_select`, `value_policy.
    make_policy_prior`) -- PAS mélangée à `yd` dans le modèle de valeur,
    échelles et rôles différents (voir `train_policy_model.py`).
  - **Auto-jeu sans heuristique câblée dans l'évaluation** :
    `use_pure_leaf_eval=True` (nouveau défaut) fait jouer les bots MCTS
    avec `value_policy.make_pairwise_leaf_eval` (évaluation directe par le
    modèle linéaire, sans rollout) au lieu de
    `make_pairwise_hybrid_leaf_eval` (qui fait d'abord quelques coups
    réels via `search.greedy_action`, lui-même truffé de bonus ajustés à
    la main -- `tree_bonus`, `CLEARING_URGENCY_BONUS`, voir search.py).
    Le facteur de branchement reste entièrement dans l'arbre : `legal
    _actions()` n'est jamais filtré, `MCTS.choose()` explore chaque action
    légale (voir `Node.uct_select`) -- la seule heuristique qui reste,
    volontairement, est `choose_draw_source`/`choose_payment` (pioche
    Clairière et défausse de paiement), hors scope ici : les exposer à
    l'arbre a déjà été tenté et abandonné (`git log 3e1a79b`/`5506f89`),
    voir la discussion dans la conversation du 17/08.

Sortie : reference/bootstrap_dataset.npz (Xd, yd, yp, Xa, Xb, game_idx,
feature_names) -- `yd`/`yp` partagent les mêmes `Xd`/`Xa`/`Xb`/`game_idx`
(mêmes paires), juste deux cibles différentes.
"""
import math
import random
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


def _root_candidates(bot, legal, min_visits):
    """Actions "pose" (hors draw/skip_effect) visitées au moins `min_visits`
    fois au nœud racine de `bot`, juste après `bot.choose(game)`.

    `min_visits` écarte les branches à peine explorées (bruit d'UCT en
    exploration pure, pas encore une estimation de valeur convergée) --
    sans ce filtre, une action visitée 1 fois pourrait entrer dans une
    paire avec une valeur quasi arbitraire.
    """
    legal_set = set(legal)
    out = []
    for a, child in bot.root.children.items():
        if a not in legal_set or a[0] in ("draw", "skip_effect"):
            continue
        if child.visits < min_visits:
            continue
        out.append(a)
    return out


def generate_pairs(n_games, seed0, mcts_iterations=50, model_path=None,
                    short_rollout_depth=10, sample_every=4, min_visits=3,
                    progress_features=False, use_pure_leaf_eval=True,
                    policy_prior_path=None, verbose=False):
    """Retourne (Xd, yd, yp, Xa, Xb, game_idx). Voir le docstring du module
    pour la différence de méthode (trajectoire MCTS, cibles = stats de
    l'arbre) et les deux ajouts du 17/08 (cible de politique, auto-jeu sans
    heuristique dans l'évaluation).

    `model_path` : modèle contrastif linéaire chargé pour le `leaf_eval`
    des bots MCTS qui jouent l'auto-jeu (par défaut, le modèle "officiel"
    reference/pairwise_model.joblib -- c'est bien le point : on VEUT que le
    MCTS générateur de données se comporte comme le MCTS réellement utilisé
    aujourd'hui, pour corriger sa distribution d'états visités, pas pour
    en inventer une autre).

    `use_pure_leaf_eval` : voir le docstring du module -- True (défaut)
    retire le rollout court basé sur `greedy_action` (heuristique) de
    l'évaluation, False garde `make_pairwise_hybrid_leaf_eval` comme dans
    les pilotes 1/2.

    `policy_prior_path` : chemin d'un `policy_model.joblib` déjà entraîné,
    pour brancher `MCTS(policy_prior=...)` (PUCT au lieu d'UCT nu) sur
    CETTE génération d'auto-jeu -- None (défaut) au premier passage,
    puisqu'aucun modèle de politique n'existe encore avant d'avoir tourné
    ce générateur une première fois.
    """
    # make_pairwise_leaf_eval (pur, sans rollout) ne prend pas de `seed` --
    # rien d'aléatoire à ensemencer sans rollout. make_pairwise_hybrid_leaf_eval
    # en a besoin pour son epsilon-greedy interne.
    def _make_leaf_eval(seed):
        if use_pure_leaf_eval:
            return VP.make_pairwise_leaf_eval(model_path=model_path)
        return VP.make_pairwise_hybrid_leaf_eval(
            model_path=model_path, short_rollout_depth=short_rollout_depth, seed=seed)

    policy_prior = (VP.make_policy_prior(policy_prior_path)
                    if policy_prior_path is not None else None)

    Xd, yd, yp, Xa, Xb, game_idx = [], [], [], [], [], []
    t0 = time.time()
    for gi in range(n_games):
        seed = seed0 + gi
        g = G.Game(n_players=2, seed=seed)
        bots = [
            S.MCTS(observer=i, iterations=mcts_iterations, seed=seed * 31 + i,
                   leaf_eval=_make_leaf_eval(seed * 31 + i),
                   policy_prior=policy_prior)
            for i in range(2)
        ]
        turns = 0
        while not g.over and turns < 400:
            mover = g.current
            legal = g.legal_actions()
            if len(legal) == 1:
                action = legal[0]
            else:
                action = bots[mover].choose(g)
                if turns % sample_every == 0:
                    candidates = _root_candidates(bots[mover], legal, min_visits)
                    if len(candidates) >= 2:
                        progress = [turns, len(g.deck), g.winters_seen] if progress_features else []
                        feats_by_action, val_by_action, visits_by_action = {}, {}, {}
                        for a in candidates:
                            branch = g.clone()
                            branch.apply(a)
                            raw_score = branch.scores()[mover]
                            feats_by_action[a] = F.extract_features(branch, mover) + [raw_score] + progress
                            child = bots[mover].root.children[a]
                            val_by_action[a] = child.value / child.visits
                            visits_by_action[a] = child.visits
                        for i in range(len(candidates)):
                            for j in range(i + 1, len(candidates)):
                                ai, aj = candidates[i], candidates[j]
                                fi = np.asarray(feats_by_action[ai], dtype=np.float32)
                                fj = np.asarray(feats_by_action[aj], dtype=np.float32)
                                Xd.append(fi - fj)
                                yd.append(val_by_action[ai] - val_by_action[aj])
                                yp.append(math.log(visits_by_action[ai])
                                          - math.log(visits_by_action[aj]))
                                Xa.append(fi)
                                Xb.append(fj)
                                game_idx.append(gi)
            g.apply(action)
            for b in bots:
                b.advance(action)
            turns += 1
        if verbose:
            elapsed = time.time() - t0
            print(f"  partie {gi + 1}/{n_games} : {turns} tours, "
                  f"{len(yd)} paires cumulées, {elapsed:.0f}s écoulées",
                  file=sys.stderr)
    return Xd, yd, yp, Xa, Xb, game_idx


if __name__ == "__main__":
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    mcts_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    min_visits = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    policy_prior_path = sys.argv[4] if len(sys.argv) > 4 else None

    Xd, yd, yp, Xa, Xb, game_idx = generate_pairs(
        n_games, seed0=90000, mcts_iterations=mcts_iterations,
        min_visits=min_visits, policy_prior_path=policy_prior_path, verbose=True)
    Xd = np.asarray(Xd, dtype=np.float32)
    yd = np.asarray(yd, dtype=np.float32)
    yp = np.asarray(yp, dtype=np.float32)
    Xa = np.asarray(Xa, dtype=np.float32)
    Xb = np.asarray(Xb, dtype=np.float32)
    game_idx = np.asarray(game_idx, dtype=np.int32)
    feature_names = list(F.FEATURE_NAMES) + ["raw_score"]
    out = Path(__file__).resolve().parent / "bootstrap_dataset.npz"
    np.savez_compressed(out, Xd=Xd, yd=yd, yp=yp, Xa=Xa, Xb=Xb, game_idx=game_idx,
                         feature_names=np.array(feature_names))
    print(f"{n_games} parties MCTS ({mcts_iterations} it., min_visits={min_visits}, "
          f"policy_prior={policy_prior_path or 'aucun (UCT nu)'}) -> "
          f"{Xd.shape[0]} paires, {Xd.shape[1]} features -> {out}")
