"""Teste une idée alternative au modèle contrastif : entraîner un modèle de
valeur ABSOLU (features d'état -> gain restant réel jusqu'à la fin de la
partie) uniquement sur les états du joueur GAGNANT de chaque partie
(question de Mehdi, 17/08 : "on regarde comment le bot qui gagne fait pour
gagner, et on entraîne un modèle à reproduire cette stratégie").

Deux axes de comparaison avec la méthode actuelle (`gen_value_dataset.py`,
qui garde les états des DEUX joueurs, gagnant et perdant) :

  1. R²/MAE offline classique : le modèle sait-il prédire le gain restant ?
  2. Le test qui compte vraiment pour MCTS : sait-il ORDONNER deux coups
     candidats posés depuis le MÊME état ? Vérité terrain = le vrai écart
     mesuré par rollout court à graines communes dans
     `reference/pairwise_dataset.npz` (voir `gen_pairwise_dataset.py`).

Résultat (voir reference/MODELS.md pour la version consignée) : négatif
sur les deux axes. Restreindre au gagnant ne corrige pas le problème
diagnostiqué pour le modèle absolu "tous joueurs" (README, section
"Fonctions de valeur pour MCTS", approche 1) -- ça divise juste le
dataset par deux sans rien changer au bruit de fond : le score final
d'une partie dépend de dizaines de pioches et de coups adverses après
CHAQUE décision, que la trajectoire regardée soit celle du gagnant ou
non. C'est ce bruit de crédit-assignment, pas "quel joueur on regarde",
qui noie le signal -- raison d'être de l'approche contrastive (comparer
deux coups depuis le même état, mêmes pioches ensuite via common random
numbers) utilisée par le reste de reference/.

Ne produit aucun fichier .joblib/.npz : diagnostic jetable, à relancer au
besoin (déterministe, seed fixe).
"""
import random
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reference"))

import game as G
import search as S
import features as F


def collect(n_games, seed0, epsilon=0.4, sample_every=2):
    """Auto-jeu greedy bruité (même méthode que gen_value_dataset.py),
    retourne (game_idx, feats, gain_restant, is_winner) par snapshot."""
    rows = []
    for gi in range(n_games):
        seed = seed0 + gi
        g = G.Game(n_players=2, seed=seed)
        rng_traj = random.Random(seed)
        snapshots = []
        turns = 0
        while not g.over and turns < 400:
            cur = g.current
            if turns % sample_every == 0:
                feats = F.extract_features(g, cur)
                score_now = g.scores()[cur]
                snapshots.append((cur, score_now, feats))
            action = S.greedy_action(g, rng_traj, epsilon=epsilon)
            g.apply(action)
            turns += 1
        final = g.scores()
        winner = 0 if final[0] > final[1] else (1 if final[1] > final[0] else None)
        for cur, score_now, feats in snapshots:
            rows.append((gi, feats, final[cur] - score_now, cur == winner))
    return rows


def fit_eval(X, y, groups, label):
    """Ridge avec split au niveau des PARTIES (groups), comme
    train_pairwise_model.py -- évite la fuite de snapshots corrélés d'une
    même partie entre train et test."""
    train_idx, test_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=0).split(X, y, groups))
    model = RidgeCV(alphas=np.logspace(-2, 3, 20))
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    r2 = model.score(X[test_idx], y[test_idx])
    mae = np.mean(np.abs(pred - y[test_idx]))
    baseline_mae = np.mean(np.abs(y[test_idx] - y[train_idx].mean()))
    print(f"[{label}] n_train={len(train_idx)} n_test={len(test_idx)} "
          f"R2={r2:.4f} MAE={mae:.2f} (baseline moyenne={baseline_mae:.2f})")
    return model


def main():
    print("=== Génération (300 parties, greedy bruité, sample_every=2) ===")
    rows = collect(300, seed0=50000)
    print(f"{len(rows)} snapshots, {len(set(r[0] for r in rows))} parties")

    Xa = np.array([r[1] for r in rows], dtype=np.float32)
    ya = np.array([r[2] for r in rows], dtype=np.float32)
    ga = np.array([r[0] for r in rows], dtype=np.int32)

    rows_w = [r for r in rows if r[3]]
    Xw = np.array([r[1] for r in rows_w], dtype=np.float32)
    yw = np.array([r[2] for r in rows_w], dtype=np.float32)
    gw = np.array([r[0] for r in rows_w], dtype=np.int32)
    print(f"dont {len(rows_w)} snapshots côté gagnant\n")

    print("=== 1. Offline (le modèle sait-il prédire le gain restant ?) ===")
    m_all = fit_eval(Xa, ya, ga, "TOUS les joueurs (méthode actuelle)")
    m_win = fit_eval(Xw, yw, gw, "GAGNANT seulement")

    print("\n=== 2. Ce qui compte pour MCTS : ordonner 2 coups au même nœud ===")
    data = np.load(ROOT / "reference" / "pairwise_dataset.npz", allow_pickle=True)
    Xpa, Xpb, yd = data["Xa"], data["Xb"], data["yd"]
    # les 77 premières colonnes = features.py ; la 78e = raw_score, ajoutée
    # seulement pour le modèle pairwise -- absente des modèles ci-dessus.
    Xpa77, Xpb77 = Xpa[:, :-1], Xpb[:, :-1]

    for label, model in (("TOUS les joueurs", m_all), ("GAGNANT seulement", m_win)):
        pred_diff = model.predict(Xpa77) - model.predict(Xpb77)
        corr = np.corrcoef(pred_diff, yd)[0, 1]
        same_sign = np.mean(np.sign(pred_diff) == np.sign(yd))
        mae_diff = np.mean(np.abs(pred_diff - yd))
        print(f"[{label}] corrélation avec le vrai diff (rollout court)={corr:.3f} "
              f"| accord de signe={same_sign:.1%} | MAE diff={mae_diff:.2f} pts "
              f"(écart-type du vrai diff={np.std(yd):.2f})")

    # Repère : le modèle pairwise officiel, entraîné ET testé sur CE dataset
    # (donc optimiste/pas comparable à armes égales -- juste une échelle).
    pw = joblib.load(ROOT / "reference" / "pairwise_model.joblib")
    w = np.asarray(pw["weights"], dtype=np.float32)
    pred_diff_pw = Xpa @ w - Xpb @ w
    corr_pw = np.corrcoef(pred_diff_pw, yd)[0, 1]
    same_sign_pw = np.mean(np.sign(pred_diff_pw) == np.sign(yd))
    print(f"[repère : pairwise_model officiel, fit sur ces mêmes données] "
          f"corrélation={corr_pw:.3f} | accord de signe={same_sign_pw:.1%}")


if __name__ == "__main__":
    main()
