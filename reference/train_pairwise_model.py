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
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent


def fit_and_save(data_path=None, out_path=None, unit="pts", target_key="yd"):
    """Ajuste le modèle linéaire contrastif sur `data_path` (schéma
    Xd/<target_key>/game_idx/feature_names) et sauvegarde le résultat dans
    `out_path`. Factorisé hors de `main()` pour être réutilisable par
    d'autres générateurs de dataset au même schéma (voir
    train_bootstrap_model.py, train_policy_model.py).

    `unit` : uniquement pour l'affichage MAE (ex. "pts" pour un dataset
    pairwise classique, dont `yd` est un delta de score réel ; le dataset
    bootstrap a un `yd` en rang normalisé [0, 1], pas des points -- voir
    gen_bootstrap_dataset.py).

    `target_key` : nom de la colonne cible dans le `.npz` -- "yd" (défaut,
    diff de valeur) ou "yp" (diff de politique, `log(visites)`, voir
    train_policy_model.py). `Xd` (les features) est toujours partagé entre
    les deux cibles -- même paires, juste une régression différente."""
    data_path = Path(data_path) if data_path else HERE / "pairwise_dataset.npz"
    out_path = Path(out_path) if out_path else HERE / "pairwise_model.joblib"

    data = np.load(data_path, allow_pickle=True)
    Xd, yd = data["Xd"], data[target_key]
    feature_names = list(data["feature_names"])

    # Split AU NIVEAU DES PARTIES (game_idx), pas des paires : plusieurs
    # paires viennent de la même partie (échantillonnées tous les
    # sample_every tours), un split par paire laisse fuiter des paires
    # corrélées entre train et test (voir gen_pairwise_dataset.py et
    # reference/MODELS.md, diagnostiqué le 16/08 sur le modèle MLP -- moins
    # critique ici vu la faible capacité du modèle linéaire, mais gardé
    # cohérent avec train_pairwise_mlp.py).
    if "game_idx" in data:
        groups = data["game_idx"]
        train_idx, test_idx = next(GroupShuffleSplit(
            n_splits=1, test_size=0.15, random_state=0).split(Xd, yd, groups))
        Xtr, Xte, ytr, yte = Xd[train_idx], Xd[test_idx], yd[train_idx], yd[test_idx]
    else:
        from sklearn.model_selection import train_test_split
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
    print(f"Test MAE  : {mae:.2f} {unit} (baseline 'aucune différence' : {baseline_mae:.2f} {unit})")

    joblib.dump({"weights": model.coef_, "feature_names": feature_names}, out_path)
    print(f"Modèle sauvegardé dans {out_path}")


def main():
    fit_and_save()


if __name__ == "__main__":
    main()
