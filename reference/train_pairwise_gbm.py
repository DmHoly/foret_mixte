"""Entraîne le modèle Gradient Boosting pairwise -- comparateur pour
`search.greedy_action(..., tiebreak=...)` (voir `reference/value_policy.py`,
`make_pairwise_gbm_tiebreak`, et `reference/MODELS.md`, section "Un modèle
non linéaire (Gradient Boosting) discrimine mieux les paires serrées").

Contrairement au modèle linéaire (`train_pairwise_model.py`), ce modèle
NE SE DÉCOMPOSE PAS en fonction de valeur par état (`arbre(a) - arbre(b)
!= arbre(a-b)` pour un ensemble d'arbres) -- utilisé uniquement en
comparaison directe de deux états déjà avancés, jamais comme `leaf_eval`
MCTS. Voir la docstring de `search.greedy_action` pour son usage exact
(départage des candidats presque à égalité de gain exact).

Hyperparamètres retenus par validation croisée
(`reference/diagnose_nonlinear_capacity.py`, 6 configs testées, toutes
dans 57.3-57.7% de précision de signe sur les paires serrées contre 47.7%
pour le linéaire -- peu sensible au choix fin dans cette plage) : un choix
simple au milieu de la plage testée, avec arrêt anticipé sur un split de
validation interne pour éviter de sur-régler `max_iter` à la main.

Usage : python train_pairwise_gbm.py [dataset.npz] [model_out.joblib]
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_pairwise_model import sign_accuracy_report

HERE = Path(__file__).resolve().parent

GBM_PARAMS = dict(max_depth=4, max_iter=200, learning_rate=0.05,
                   l2_regularization=1.0, random_state=0,
                   early_stopping=True, validation_fraction=0.1,
                   n_iter_no_change=15)


def main():
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "pairwise_dataset.npz"
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "pairwise_gbm_model.joblib"

    data = np.load(dataset_path, allow_pickle=True)
    Xd, yd = data["Xd"], data["yd"]
    groups = data["game_idx"]

    # Meme split que train_pairwise_model.py (groupe par partie, pas par
    # paire) pour rester comparable.
    train_idx, test_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=0.15, random_state=0).split(Xd, yd, groups))
    Xtr, Xte, ytr, yte = Xd[train_idx], Xd[test_idx], yd[train_idx], yd[test_idx]

    model = HistGradientBoostingRegressor(**GBM_PARAMS)
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    mae = float(np.mean(np.abs(pred - yte)))
    ss_res = float(np.sum((pred - yte) ** 2))
    ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot

    print(f"n_iter effectif (early stopping) : {model.n_iter_}")
    print(f"Test R^2  : {r2:.4f}")
    print(f"Test MAE  : {mae:.2f} pts")
    print(sign_accuracy_report(pred, yte))

    joblib.dump(model, out_path)
    print(f"Modele sauvegarde dans {out_path}")


if __name__ == "__main__":
    main()
