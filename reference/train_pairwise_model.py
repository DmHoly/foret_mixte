"""Entraîne la fonction de valeur linéaire sur reference/pairwise_dataset.npz
(diff de features -> diff de gain réel, common random numbers). Voir
gen_pairwise_dataset.py pour la justification. Sauvegarde le vecteur de
poids dans reference/pairwise_model.joblib ; `value(état) = w . feat(état)`.

Usage : `python train_pairwise_model.py [dataset.npz] [model_out.joblib] [tau]`
(les trois arguments sont optionnels, défaut = le dataset/modèle vivants,
sans repondération ; utilisé avec des chemins explicites par la piste
bootstrap pour entraîner un modèle candidat sans écraser le modèle vivant
avant gating, voir gen_pairwise_dataset_bootstrap.py).

Métrique de référence : en plus de R²/MAE (utile pour comparer aux
chiffres historiques de MODELS.md), affiche la précision de SIGNE par
tranche de |diff réel|. Diagnostiqué le 18/08
(reference/diagnose_pairwise_metrics.py) : R²/MAE global s'améliore à
chaque itération du modèle (k=1 -> k=5 -> bootstrap gen1) alors que le
résultat réel au gating contre B empire dans le même temps -- parce que
R²/MAE est dominé par les paires à GROS écart (peu nombreuses, faciles à
classer, mais qui pèsent lourd dans l'erreur au carré), au prix d'une
précision dégradée sur les paires SERRÉES -- exactement celles où UCT a
besoin de discriminer entre deux branches voisines. Le classement par
précision-de-signe-sur-paires-serrées colle beaucoup mieux au résultat
réel du gating que R²/MAE.

Option `tau` (repondération) : teste l'hypothèse que reponderer les
exemples d'entraînement vers les petits |diff| (poids = 1/(|y|+tau))
corrigerait ce biais. Résultat empirique (18/08, sur pairwise_dataset.npz) :
NÉGATIF -- testé sur une plage tau in [1.5, 10], en pondération extrême
(50x sur les paires serrées), et même en entraînant EXCLUSIVEMENT sur les
paires serrées (bypass complet de la pondération) : la précision de signe
sur |diff reel| < 3 reste bloquée à 69.2-69.6%, statistiquement
indiscernable du modèle non pondéré (69.4%). Ce n'est donc PAS un problème
de fonction de perte : c'est un plafond du modèle linéaire sur ces 78
features pour ce niveau de discrimination fine (ou, en partie, du bruit
irréductible dans l'étiquette elle-même -- yd vient d'une moyenne de
k_rollout rollouts, pas d'une vérité exacte). Le paramètre reste
disponible (défaut : pas de repondération, comportement inchangé) mais ne
doit pas être présenté comme une correction.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent

# Tranches en points de |diff réel| pour la précision de signe -- "serré"
# (< 3 pts) est le cas le plus fréquent et le plus décisif pour UCT ;
# "large" (>= 20 pts) est presque toujours facile à classer correctement.
SIGN_ACC_BINS = [(0, 3), (3, 8), (8, 20), (20, 1e9)]


def sign_accuracy_report(pred, y):
    """Precision de signe, EGALITES EXACTES EXCLUES (y == 0 -- typiquement
    des candidats dupliques menant a un resultat identique). Un modele
    lineaire SANS INTERCEPT les "reussit" gratuitement par construction
    (w.0 = 0) quels que soient ses poids, ce qui gonfle artificiellement
    son score si on les compte -- diagnostique le 18/08 en comparant a un
    modele non lineaire, qui n'a structurellement aucune raison de tomber
    pile sur 0 (voir reference/MODELS.md)."""
    lines = []
    nonzero = y != 0
    n_ties = int(np.sum(~nonzero))
    overall = float(np.mean(np.sign(pred[nonzero]) == np.sign(y[nonzero])))
    lines.append(f"Precision de signe (test, tout confondu, {n_ties} egalites exactes exclues) : {overall:.1%}")
    for lo, hi in SIGN_ACC_BINS:
        mask = nonzero & (np.abs(y) >= lo) & (np.abs(y) < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        acc = float(np.mean(np.sign(pred[mask]) == np.sign(y[mask])))
        hi_label = f"{hi:.0f}" if hi < 1e9 else "inf"
        lines.append(f"  |diff reel| in [{lo:>2.0f}, {hi_label:>3s}) pts : "
                      f"n={n:5d}  precision signe={acc:.1%}")
    return "\n".join(lines)


def main():
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "pairwise_dataset.npz"
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "pairwise_model.joblib"
    tau = float(sys.argv[3]) if len(sys.argv) > 3 else None

    data = np.load(dataset_path, allow_pickle=True)
    Xd, yd = data["Xd"], data["yd"]
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

    sample_weight = None
    if tau is not None:
        sample_weight = 1.0 / (np.abs(ytr) + tau)
        print(f"Repondere vers les paires serrees : poids = 1/(|diff|+{tau:g}) "
              f"-- voir la docstring pour le resultat empirique (negatif)")

    # fit_intercept=False : un décalage constant sur un état n'a de sens que
    # différencé (voir docstring de gen_pairwise_dataset.py), donc pas
    # d'intercept ici -- seuls les poids par feature sont identifiables.
    model = RidgeCV(alphas=np.logspace(-2, 3, 20), fit_intercept=False)
    model.fit(Xtr, ytr, sample_weight=sample_weight)

    pred = model.predict(Xte)
    mae = np.mean(np.abs(pred - yte))
    r2 = model.score(Xte, yte)
    baseline_mae = np.mean(np.abs(yte))  # prédire "0 différence" tout le temps

    print(f"alpha choisi : {model.alpha_:.3g}")
    print(f"Test R^2  : {r2:.4f}")
    print(f"Test MAE  : {mae:.2f} pts (baseline 'aucune différence' : {baseline_mae:.2f} pts)")
    print(sign_accuracy_report(pred, yte))

    joblib.dump({"weights": model.coef_, "feature_names": feature_names}, out_path)
    print(f"Modèle sauvegardé dans {out_path}")


if __name__ == "__main__":
    main()
