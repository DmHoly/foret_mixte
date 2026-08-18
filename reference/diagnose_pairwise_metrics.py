"""Diagnostic demandé par Mehdi (18/08) après le rejet au gating de la
génération 1 du bootstrap : le modèle avait le meilleur R²/MAE jamais
mesuré (0.336/4.40) mais a perdu plus largement contre B (7/30) que les
deux modèles précédents, moins bons hors ligne (k=1 : 13-15/30, k=5 :
9/30). Hypothèse à tester : R²/MAE global est dominé par les paires
"faciles" (grand écart réel, n'importe quel modèle raisonnable les classe
correctement) et masque une précision DÉGRADÉE sur les paires SERRÉES --
exactement celles où UCT a besoin de discriminer pour choisir entre deux
branches proches. Si la précision de signe sur les paires serrées empire
d'un modèle à l'autre alors que R²/MAE global s'améliore, ça expliquerait
le paradoxe sans invoquer la seule hypothèse "surconfiance hors
distribution" déjà avancée dans MODELS.md.

Reproduit exactement le split train/test de train_pairwise_model.py
(GroupShuffleSplit par game_idx, test_size=0.15, random_state=0) pour
évaluer chaque modèle sur SES PROPRES paires de test, jamais vues à
l'entraînement.

CORRECTIF (18/08, après coup) : la précision de signe EXCLUT désormais
les paires à diff réel EXACTEMENT nul (candidats dupliqués -- ex. deux
exemplaires identiques d'une carte en main -- qui produisent un état et
un résultat de rollout identiques). Ces paires représentaient ~23% du
dataset (7804/35940 sur pairwise_dataset.npz) et un modèle linéaire SANS
INTERCEPT les "réussit" gratuitement par construction (`w.(fi-fi) = 0`,
`sign(0)==sign(0)`), quels que soient ses poids appris -- alors qu'un
modèle non linéaire (arbre, MLP) n'a structurellement aucune raison de
tomber pile sur 0 pour ce même point. Une première version de ce
diagnostic ne les excluait pas, ce qui gonflait artificiellement le score
"paires serrées" de tous les modèles linéaires (k=5 : 69.4% -> 47.9% une
fois corrigé, sous le hasard) et rendait la comparaison avec un modèle
non linéaire structurellement biaisée en faveur du linéaire. Voir
reference/MODELS.md pour l'historique de la correction.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent

MODELS = [
    ("k=1 (bruité, ancien modèle vivant)", "pairwise_dataset_k1_noisy.npz", "pairwise_model_k1_noisy.joblib", "13-15/30"),
    ("k=5 (modèle vivant actuel)", "pairwise_dataset.npz", "pairwise_model.joblib", "9/30"),
    ("bootstrap gen1 (trajectoires MCTS)", "pairwise_dataset_bootstrap_gen1.npz", "pairwise_model_bootstrap_gen1.joblib", "7/30"),
]

# Bornes en points de |diff réel| : "serré" = les deux coups sont presque
# équivalents, c'est justement le cas où se discriminer compte le plus
# pour UCT ; "large" = un coup domine clairement l'autre, presque
# n'importe quel modèle le classe correctement.
BINS = [(0, 3), (3, 8), (8, 20), (20, 1e9)]


def evaluate(dataset_name, model_name):
    data = np.load(HERE / dataset_name, allow_pickle=True)
    Xd, yd, groups = data["Xd"], data["yd"], data["game_idx"]
    _, test_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=0.15, random_state=0).split(Xd, yd, groups))
    Xte, yte = Xd[test_idx], yd[test_idx]

    bundle = joblib.load(HERE / model_name)
    w = np.asarray(bundle["weights"], dtype=np.float32)
    pred = Xte @ w

    mae = float(np.mean(np.abs(pred - yte)))
    r2 = 1.0 - np.sum((pred - yte) ** 2) / np.sum((yte - yte.mean()) ** 2)
    nonzero_all = yte != 0
    sign_acc = float(np.mean(np.sign(pred[nonzero_all]) == np.sign(yte[nonzero_all])))

    n_ties = int(np.sum(yte == 0))
    print(f"  n_test={len(yte)} (dont {n_ties} egalites exactes exclues du signe)  "
          f"R^2={r2:.3f}  MAE={mae:.2f}  "
          f"precision de signe (tout confondu)={sign_acc:.1%}")
    nonzero = yte != 0
    for lo, hi in BINS:
        mask = nonzero & (np.abs(yte) >= lo) & (np.abs(yte) < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        acc = float(np.mean(np.sign(pred[mask]) == np.sign(yte[mask])))
        mae_bin = float(np.mean(np.abs(pred[mask] - yte[mask])))
        hi_label = f"{hi:.0f}" if hi < 1e9 else "inf"
        print(f"    |diff reel| in [{lo:>2.0f}, {hi_label:>3s}) pts : "
              f"n={n:5d}  precision signe={acc:.1%}  MAE={mae_bin:5.2f}")


def main():
    for label, dataset_name, model_name, gate_result in MODELS:
        print(f"== {label} -- resultat gating vs B : {gate_result} ==")
        evaluate(dataset_name, model_name)
        print()


if __name__ == "__main__":
    main()
