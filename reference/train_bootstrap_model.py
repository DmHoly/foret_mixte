"""Entraîne la fonction de valeur linéaire sur reference/bootstrap_dataset.npz
(généré par gen_bootstrap_dataset.py : auto-jeu MCTS, cible = stats de
l'arbre de recherche, pas un rollout séparé). Réutilise l'ajustement Ridge
de train_pairwise_model.py (même schéma de dataset), juste avec des
chemins d'entrée/sortie différents.

`yd` ici est un rang normalisé [0, 1] (voir search.reward_vector), pas un
delta de points comme pairwise_dataset.npz -- le MAE affiché est donc dans
cette échelle, pas en points (voir gen_bootstrap_dataset.py).

Sauvegarde le vecteur de poids dans reference/bootstrap_model.joblib,
séparé de reference/pairwise_model.joblib (le modèle "officiel" utilisé
par défaut) -- ne JAMAIS écraser ce dernier directement : voir
reference/bench_bootstrap.py pour la procédure de gating avant toute
promotion.
"""
from pathlib import Path

from train_pairwise_model import fit_and_save

HERE = Path(__file__).resolve().parent


def main():
    fit_and_save(
        data_path=HERE / "bootstrap_dataset.npz",
        out_path=HERE / "bootstrap_model.joblib",
        unit="(rang [0,1])",
    )


if __name__ == "__main__":
    main()
