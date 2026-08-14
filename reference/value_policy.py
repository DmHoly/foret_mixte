"""Évaluation de feuille MCTS via le modèle de valeur appris (sklearn),
en remplacement du rollout tronqué aléatoire.

Usage :
    from value_policy import make_leaf_eval, make_hybrid_leaf_eval
    leaf_eval = make_leaf_eval()          # modèle pur (déconseillé, voir plus bas)
    leaf_eval = make_hybrid_leaf_eval()   # quelques coups réels + modèle (recommandé)
    bot = search.MCTS(observer=0, iterations=300, leaf_eval=leaf_eval)

`leaf_eval(state)` retourne une liste de scores ESTIMÉS (un par joueur) :
score actuel + gain restant prédit par le modèle, dans le même format que
`search.rollout()` pour être un remplacement direct.

Diagnostic de session (14/08) : `make_leaf_eval` (modèle pur, sans coup
réel) perd nettement contre le rollout classique, pas par manque de
données d'entraînement mais parce que le modèle est structurellement peu
sensible à UN SEUL coup marginal (features en proportions sur une forêt de
20-30 cartes déjà posées -> une carte de plus bouge à peine la composition
relative). L'écart de prédiction entre deux coups candidats à un même
nœud (~9 pts mesurés) est plus petit que le bruit du modèle (MAE ~12-14
pts) : le signal utile pour différencier les coups est noyé dans le bruit
d'estimation, donc MCTS perd son signal d'exploitation et dérive vers une
recherche quasi non guidée. `make_hybrid_leaf_eval` corrige ça en jouant
d'abord quelques coups réels (comme un rollout court) avant d'appeler le
modèle : les coups réels font émerger la différence marginale entre
branches, le modèle ne sert plus qu'à extrapoler au-delà de cet horizon
court, là où son bruit compte relativement moins face à ce qui a déjà été
observé concrètement.
"""
import random
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import features as F
import search as S

HERE = Path(__file__).resolve().parent
_MODEL_CACHE = {}


def load_model(path=None):
    path = path or (HERE / "value_model.joblib")
    key = str(path)
    if key not in _MODEL_CACHE:
        bundle = joblib.load(path)
        _MODEL_CACHE[key] = bundle["model"]
    return _MODEL_CACHE[key]


def predict_remaining_gain(model, game, player_index):
    feats = np.asarray([F.extract_features(game, player_index)], dtype=np.float32)
    return float(model.predict(feats)[0])


def make_leaf_eval(model_path=None):
    """Modèle pur, sans aucun coup réel avant évaluation. Déconseillé en
    pratique (voir diagnostic ci-dessus) : gardé pour comparaison/tests,
    préférer `make_hybrid_leaf_eval`.
    """
    model = load_model(model_path)

    def leaf_eval(state):
        scores_now = state.scores()
        if state.over:
            return [float(s) for s in scores_now]
        return [
            scores_now[i] + predict_remaining_gain(model, state, i)
            for i in range(len(state.players))
        ]

    return leaf_eval


def load_pairwise_model(path=None):
    path = path or (HERE / "pairwise_model.joblib")
    key = str(path)
    if key not in _MODEL_CACHE:
        bundle = joblib.load(path)
        _MODEL_CACHE[key] = (np.asarray(bundle["weights"], dtype=np.float32),
                              list(bundle["feature_names"]))
    return _MODEL_CACHE[key]


def make_pairwise_leaf_eval(model_path=None):
    """Fonction de valeur linéaire entraînée par contraste (voir
    gen_pairwise_dataset.py / train_pairwise_model.py) : diff de features
    -> diff de score après rollout court, avec le score brut du candidat
    en feature (sans risque de fuite d'horloge : les candidats comparés
    partagent le même tour). `value(état) = w . feat(état)`, pas
    d'inférence sklearn par appel (juste un produit scalaire), donc pas de
    coût de latence contrairement à `make_leaf_eval`/`make_hybrid_leaf_eval`.
    """
    weights, _names = load_pairwise_model(model_path)

    def leaf_eval(state):
        scores_now = state.scores()
        if state.over:
            return [float(s) for s in scores_now]
        out = []
        for i in range(len(state.players)):
            feats = F.extract_features(state, i) + [scores_now[i]]
            out.append(float(np.dot(weights, np.asarray(feats, dtype=np.float32))))
        return out

    return leaf_eval


def make_pairwise_hybrid_leaf_eval(model_path=None, short_rollout_depth=10, seed=None):
    """Quelques coups réels (rollout court, comme `make_hybrid_leaf_eval`)
    suivis d'une évaluation par le modèle linéaire contrastif, au lieu
    d'appeler le modèle directement sur l'état de sélection/expansion brut
    (`make_pairwise_leaf_eval`).

    Raison d'être : le modèle contrastif est entraîné sur des états déjà
    avancés d'un rollout de ~20 coups (voir gen_pairwise_dataset.py), donc
    son domaine de validité correspond mieux à un état qui a déjà un peu
    évolué qu'à un état frais de sélection MCTS. `short_rollout_depth`
    coups réels rapprochent l'état soumis au modèle de la distribution sur
    laquelle il a été entraîné, en plus de laisser les coups réels
    différencier les branches (le rôle qu'ils jouent déjà dans
    `make_hybrid_leaf_eval`).
    """
    weights, _names = load_pairwise_model(model_path)
    rng = random.Random(seed)

    def leaf_eval(state):
        moves = 0
        while not state.over and moves < short_rollout_depth:
            actions = state.legal_actions()
            if len(actions) == 1:
                state.apply(actions[0])
            else:
                state.apply(S.greedy_action(
                    state, rng, S.ROLLOUT_EPSILON, S.ROLLOUT_CANDIDATES))
            moves += 1
        scores_now = state.scores()
        if state.over:
            return [float(s) for s in scores_now]
        out = []
        for i in range(len(state.players)):
            feats = F.extract_features(state, i) + [scores_now[i]]
            out.append(float(np.dot(weights, np.asarray(feats, dtype=np.float32))))
        return out

    return leaf_eval


def make_hybrid_leaf_eval(model_path=None, short_rollout_depth=18, seed=None):
    """Quelques coups réels (politique gloutonne bruitée, comme le
    rollout classique) suivis d'une évaluation par le modèle -- au lieu
    d'un rollout complet (25-35 coups) ou d'un modèle pur (0 coup).
    `short_rollout_depth` coups réels avant l'appel au modèle.
    """
    model = load_model(model_path)
    rng = random.Random(seed)

    def leaf_eval(state):
        moves = 0
        while not state.over and moves < short_rollout_depth:
            actions = state.legal_actions()
            if len(actions) == 1:
                state.apply(actions[0])
            else:
                state.apply(S.greedy_action(
                    state, rng, S.ROLLOUT_EPSILON, S.ROLLOUT_CANDIDATES))
            moves += 1
        scores_now = state.scores()
        if state.over:
            return [float(s) for s in scores_now]
        return [
            scores_now[i] + predict_remaining_gain(model, state, i)
            for i in range(len(state.players))
        ]

    return leaf_eval
