"""Entraîne la fonction de valeur linéaire sur reference/pairwise_dataset.npz
(diff de features -> diff de gain réel, common random numbers). Voir
gen_pairwise_dataset.py pour la justification. Sauvegarde le vecteur de
poids dans reference/pairwise_model.joblib ; `value(état) = w . feat(état)`.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent


def main():
    data = np.load(HERE / "pairwise_dataset.npz", allow_pickle=True)
    Xd, yd = data["Xd"], data["yd"]
    feature_names = list(data["feature_names"])

    Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.15, random_state=0)

    # fit_intercept=False : un décalage constant sur un état n'a de sens que
    # différencé (voir docstring de gen_pairwise_dataset.py), donc pas
    # d'intercept ici -- seuls les poids par feature sont identifiables.
    model = RidgeCV(alphas=np.logspace(-2, 3, 20), fit_intercept=False)
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    mae = np.mean(np.abs(pred - yte))
    r2 = model.score(Xte, yte)
    baseline_mae = np.mean(np.abs(yte))  # prédire "0 différence" tout le temps

    print(f"alpha choisi : {model.alpha_:.3g}")
    print(f"Test R^2  : {r2:.4f}")
    print(f"Test MAE  : {mae:.2f} pts (baseline 'aucune différence' : {baseline_mae:.2f} pts)")

    out = HERE / "pairwise_model.joblib"
    joblib.dump({"weights": model.coef_, "feature_names": feature_names}, out)
    print(f"Modèle sauvegardé dans {out}")


if __name__ == "__main__":
    main()
