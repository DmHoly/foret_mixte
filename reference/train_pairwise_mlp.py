"""Entraîne un modèle de valeur NON linéaire par contraste (voir
gen_pairwise_dataset.py pour la méthodologie des paires à seed commune).

Pourquoi pas juste `MLPRegressor` sur les diffs comme `train_pairwise_model.py`
le fait avec `Ridge` : parce que ça ne marche que si le modèle est LINÉAIRE.
`train_pairwise_model.py` régresse `poids . (feat_A - feat_B) ~ gain_A -
gain_B`, et comme le produit scalaire est linéaire, `poids . (feat_A - feat_B)
== poids . feat_A - poids . feat_B` : les poids appris définissent aussi une
fonction de valeur PAR ÉTAT (`value(état) = poids . feat(état)`), utilisable
seule sur n'importe quel nœud de l'arbre MCTS (pas seulement en comparant
deux candidats du même nœud). Avec un MLP, cette égalité est fausse :
`MLP(feat_A - feat_B) != MLP(feat_A) - MLP(feat_B)` en général -- un MLP
entraîné directement sur les diffs serait un bon DISCRIMINATEUR de paires,
mais inutilisable comme fonction de valeur par état.

Solution ("réseau siamois") : un seul MLP `f`, appliqué séparément à
`feat_A` et `feat_B`, entraîné pour que `f(feat_A) - f(feat_B)` approche la
diff de gain observée. `f` reste directement utilisable seul en sortie
(`value(état) = f(feat(état))`), et le calcul se fait par un simple passage
avant sur `feat_A` et `feat_B` empilés (les poids sont partagés).

Pas de framework d'autodiff dans l'environnement (torch/tensorflow
indisponibles) : rétropropagation écrite à la main pour un petit MLP
(2 couches cachées), optimisée par L-BFGS (scipy), largement suffisant vu
la taille du réseau (~7000 paramètres) et du dataset (~20k paires).
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent


def _group_split(*arrays, groups, test_size, random_state):
    idx_tr, idx_te = next(GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state).split(arrays[0], groups=groups))
    out = []
    for a in arrays:
        out += [a[idx_tr], a[idx_te]]
    return out, groups[idx_tr]


def _init_params(n_in, h1, h2, rng):
    # He init, adaptee a des couches ReLU.
    W1 = rng.standard_normal((n_in, h1)).astype(np.float64) * np.sqrt(2.0 / n_in)
    b1 = np.zeros(h1)
    W2 = rng.standard_normal((h1, h2)).astype(np.float64) * np.sqrt(2.0 / h1)
    b2 = np.zeros(h2)
    W3 = rng.standard_normal((h2, 1)).astype(np.float64) * np.sqrt(2.0 / h2)
    b3 = np.zeros(1)
    return [W1, b1, W2, b2, W3, b3]


def _pack(params):
    return np.concatenate([p.ravel() for p in params])


def _unpack(flat, shapes):
    out, i = [], 0
    for shape in shapes:
        n = int(np.prod(shape))
        out.append(flat[i:i + n].reshape(shape))
        i += n
    return out


def _forward(params, X):
    W1, b1, W2, b2, W3, b3 = params
    Z1 = X @ W1 + b1
    A1 = np.maximum(Z1, 0.0)
    Z2 = A1 @ W2 + b2
    A2 = np.maximum(Z2, 0.0)
    Z3 = A2 @ W3 + b3
    out = Z3.ravel()
    cache = (X, Z1, A1, Z2, A2)
    return out, cache


def _backward(params, cache, dOut):
    W1, b1, W2, b2, W3, b3 = params
    X, Z1, A1, Z2, A2 = cache
    n = X.shape[0]

    dZ3 = dOut.reshape(-1, 1)
    dW3 = A2.T @ dZ3
    db3 = dZ3.sum(axis=0)

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * (Z2 > 0)
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)

    return [dW1, db1, dW2, db2, dW3, db3]


def _loss_and_grad(flat, shapes, Xa, Xb, y, l2):
    params = _unpack(flat, shapes)
    n = Xa.shape[0]
    X_stack = np.vstack([Xa, Xb])
    out_stack, cache = _forward(params, X_stack)
    f_a, f_b = out_stack[:n], out_stack[n:]
    pred_diff = f_a - f_b
    resid = pred_diff - y
    loss = float(np.mean(resid ** 2))

    dOut_a = 2.0 * resid / n
    dOut_b = -2.0 * resid / n
    dOut_stack = np.concatenate([dOut_a, dOut_b])
    grads = _backward(params, cache, dOut_stack)

    # L2 seulement sur les poids (pas les biais).
    for gi, pi in zip((0, 2, 4), (0, 2, 4)):
        loss += l2 * float(np.sum(params[pi] ** 2))
        grads[gi] = grads[gi] + 2.0 * l2 * params[pi]

    return loss, _pack(grads)


class PairwiseMLP:
    def __init__(self, n_in, h1=64, h2=32, seed=0):
        rng = np.random.default_rng(seed)
        self.shapes = None
        self.params = _init_params(n_in, h1, h2, rng)
        self.shapes = [p.shape for p in self.params]
        self.mean_ = None
        self.std_ = None

    def fit(self, Xa, Xb, y, l2=1e-2, maxiter=400):
        self.mean_ = np.vstack([Xa, Xb]).mean(axis=0)
        self.std_ = np.vstack([Xa, Xb]).std(axis=0) + 1e-6
        Xa_s = (Xa - self.mean_) / self.std_
        Xb_s = (Xb - self.mean_) / self.std_

        x0 = _pack(self.params)
        res = minimize(_loss_and_grad, x0, args=(self.shapes, Xa_s, Xb_s, y, l2),
                        jac=True, method="L-BFGS-B",
                        options={"maxiter": maxiter, "disp": False})
        self.params = _unpack(res.x, self.shapes)
        return res

    def predict_diff(self, Xa, Xb):
        Xa_s = (Xa - self.mean_) / self.std_
        Xb_s = (Xb - self.mean_) / self.std_
        n = Xa_s.shape[0]
        out, _ = _forward(self.params, np.vstack([Xa_s, Xb_s]))
        return out[:n] - out[n:]

    def value(self, X):
        """Fonction de valeur PAR ÉTAT -- ce qu'appelle un leaf_eval MCTS."""
        X_s = (np.atleast_2d(X) - self.mean_) / self.std_
        out, _ = _forward(self.params, X_s)
        return out


def _mae_r2(pred, y):
    mae = float(np.mean(np.abs(pred - y)))
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return mae, 1 - ss_res / ss_tot


def main():
    data = np.load(HERE / "pairwise_dataset.npz", allow_pickle=True)
    Xa, Xb, yd = data["Xa"], data["Xb"], data["yd"]
    feature_names = list(data["feature_names"])
    if "game_idx" not in data:
        raise SystemExit("pairwise_dataset.npz sans game_idx -- regenerer avec "
                          "gen_pairwise_dataset.py (version courante).")
    groups = data["game_idx"]

    # Split a 3 niveaux, TOUJOURS par partie (game_idx), jamais par paire :
    # train (70%) pour l'optimisation, val (15%) pour choisir les
    # hyperparametres, test (15%) laisse totalement intact jusqu'au chiffre
    # final -- eviter de choisir les hyperparametres sur le meme jeu que
    # celui qui sert a rapporter la performance (sinon on refait la meme
    # fuite, une echelle au-dessus). Voir reference/MODELS.md (16/08) pour
    # le diagnostic qui a motive ce changement.
    (Xa_trval, Xa_te, Xb_trval, Xb_te, y_trval, y_te), groups_trval = _group_split(
        Xa, Xb, yd, groups=groups, test_size=0.15, random_state=0)
    (Xa_tr, Xa_val, Xb_tr, Xb_val, y_tr, y_val), _ = _group_split(
        Xa_trval, Xb_trval, y_trval, groups=groups_trval, test_size=0.176, random_state=1)
    print(f"train={len(y_tr)}  val={len(y_val)}  test={len(y_te)} paires "
          f"({len(np.unique(groups))} parties)")

    best = None
    for h1, h2 in [(16, 8), (32, 16), (64, 32)]:
        for l2 in (1e-2, 3e-2, 1e-1, 3e-1):
            model = PairwiseMLP(Xa.shape[1], h1=h1, h2=h2, seed=0)
            model.fit(Xa_tr, Xb_tr, y_tr, l2=l2, maxiter=300)
            pred_val = model.predict_diff(Xa_val, Xb_val)
            mae_val, r2_val = _mae_r2(pred_val, y_val)
            print(f"  h=({h1},{h2}) l2={l2:g}  val R2={r2_val:.4f}  val MAE={mae_val:.2f}")
            if best is None or mae_val < best[0]:
                best = (mae_val, r2_val, h1, h2, l2)

    _, _, h1, h2, l2 = best
    # Reentraine sur train+val avec les hyperparametres choisis, jamais
    # touche au test avant ce point.
    model = PairwiseMLP(Xa.shape[1], h1=h1, h2=h2, seed=0)
    model.fit(Xa_trval, Xb_trval, y_trval, l2=l2, maxiter=300)
    pred_te = model.predict_diff(Xa_te, Xb_te)
    mae_te, r2_te = _mae_r2(pred_te, y_te)
    print(f"Meilleur (choisi sur val) : h=({h1},{h2}) l2={l2:g}")
    print(f"Test (jamais vu) : R2={r2_te:.4f}  MAE={mae_te:.2f} pts")

    out = HERE / "pairwise_mlp_model.joblib"
    joblib.dump({
        "params": model.params, "mean": model.mean_, "std": model.std_,
        "shapes": model.shapes, "feature_names": feature_names,
        "h1": h1, "h2": h2, "l2": l2,
    }, out)
    print(f"Modele sauvegarde dans {out}")


if __name__ == "__main__":
    main()
