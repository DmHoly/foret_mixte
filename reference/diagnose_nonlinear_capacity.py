"""Teste si un modèle NON LINÉAIRE discrimine mieux les paires serrées que
le modèle linéaire actuel (voir reference/MODELS.md, "Un modèle non
linéaire (Gradient Boosting) discrimine mieux les paires serrées").

Contexte : le modèle linéaire pairwise tourne au niveau du hasard (47.9%)
sur les paires à |diff réel| < 3 pts -- vérifié irréductible par
repondération (voir reference/train_pairwise_model.py, argument tau).
Reste la question ouverte : plafond de MODÈLE (linéaire trop simple) ou
de FEATURES (aucun modèle ne peut faire mieux) ? Ce script teste la
première hypothèse en pur diagnostic (PAS encore branché dans MCTS --
voir la docstring de MODELS.md pour le piège de décomposabilité à éviter
avant intégration) : `HistGradientBoostingRegressor` entraîné sur la
MÊME cible `Xd -> yd` que le modèle linéaire, évalué par validation
croisée à 4 blocs groupés par partie (pas un split unique) sur plusieurs
configurations d'hyperparamètres, pour écarter un résultat de chance.

Usage : python reference/diagnose_nonlinear_capacity.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve().parent

HGB_CONFIGS = [
    dict(max_depth=3, max_iter=150, learning_rate=0.05, l2_regularization=1.0),
    dict(max_depth=4, max_iter=200, learning_rate=0.05, l2_regularization=1.0),
    dict(max_depth=6, max_iter=150, learning_rate=0.05, l2_regularization=1.0),
    dict(max_depth=4, max_iter=300, learning_rate=0.03, l2_regularization=3.0),
    dict(max_depth=3, max_iter=300, learning_rate=0.03, l2_regularization=3.0),
    dict(max_depth=2, max_iter=150, learning_rate=0.05, l2_regularization=1.0),
]


def close_sign_acc(pred, y, lo=0.0, hi=3.0):
    """Precision de signe sur |y| in [lo, hi), egalites exactes (y==0)
    exclues -- voir train_pairwise_model.py pour le pourquoi."""
    nz = y != 0
    mask = nz & (np.abs(y) >= lo) & (np.abs(y) < hi)
    return float(np.mean(np.sign(pred[mask]) == np.sign(y[mask]))), int(mask.sum())


def cv_close_sign_acc(model_factory, X, y, groups, n_splits=4):
    gkf = GroupKFold(n_splits=n_splits)
    accs = []
    for tr, va in gkf.split(X, y, groups):
        m = model_factory()
        m.fit(X[tr], y[tr])
        acc, _ = close_sign_acc(m.predict(X[va]), y[va])
        accs.append(acc)
    return float(np.mean(accs)), float(np.std(accs))


def main():
    data = np.load(HERE / "pairwise_dataset.npz", allow_pickle=True)
    Xd, yd, groups = data["Xd"], data["yd"], data["game_idx"]
    train_idx, _ = next(GroupShuffleSplit(
        n_splits=1, test_size=0.15, random_state=0).split(Xd, yd, groups))
    Xtr, ytr, groups_tr = Xd[train_idx], yd[train_idx], groups[train_idx]

    mean, std = cv_close_sign_acc(
        lambda: RidgeCV(alphas=np.logspace(-2, 3, 20), fit_intercept=False),
        Xtr, ytr, groups_tr)
    print(f"Lineaire (Ridge)          : CV precision signe |diff|<3 = {mean:.1%} +/- {std:.1%}")

    for cfg in HGB_CONFIGS:
        mean, std = cv_close_sign_acc(
            lambda cfg=cfg: HistGradientBoostingRegressor(random_state=0, early_stopping=False, **cfg),
            Xtr, ytr, groups_tr)
        print(f"HGB {cfg}  : CV precision signe |diff|<3 = {mean:.1%} +/- {std:.1%}")


if __name__ == "__main__":
    main()
