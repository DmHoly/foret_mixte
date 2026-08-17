"""Entraîne le modèle de POLITIQUE linéaire sur reference/bootstrap_dataset.npz
(colonne `yp`, générée par gen_bootstrap_dataset.py). Réutilise l'ajustement
Ridge de train_pairwise_model.py (même schéma de dataset, `target_key="yp"`
au lieu de "yd").

Cible = `log(visites[ai]) - log(visites[aj])` pour deux candidats à la même
racine -- la distribution de visites de la racine EST la politique
améliorée façon AlphaZero (π ∝ N(s,a)), pas un delta de score comme le
modèle de valeur. Le poids appris sert de PRIOR PUCT (`search.py`,
`value_policy.make_policy_prior`) : `score_brut(état) = w_policy . feat(état)`,
softmax sur les candidats légaux au moment de la sélection dans l'arbre
(voir Node.uct_select) -- ce fichier ne sauvegarde qu'un score brut, jamais
une probabilité déjà normalisée (la normalisation dépend des actions
légales de l'itération courante, pas connue à l'entraînement).

Sauvegarde le vecteur de poids dans reference/policy_model.joblib, séparé
de bootstrap_model.joblib (valeur) -- même Xd, cibles et rôles différents,
jamais mélangés.
"""
from pathlib import Path

from train_pairwise_model import fit_and_save

HERE = Path(__file__).resolve().parent


def main():
    fit_and_save(
        data_path=HERE / "bootstrap_dataset.npz",
        out_path=HERE / "policy_model.joblib",
        unit="(log visites)",
        target_key="yp",
    )


if __name__ == "__main__":
    main()
