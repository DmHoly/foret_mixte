"""Entraîne la fonction de valeur (MLPRegressor sklearn) sur
reference/value_dataset.npz, sauvegarde le pipeline (scaler+modèle) dans
reference/value_model.joblib.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent


def main():
    data = np.load(HERE / "value_dataset.npz", allow_pickle=True)
    X, y = data["X"], data["y"]
    feature_names = list(data["feature_names"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=0
    )

    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(96, 48),
            activation="relu",
            alpha=1e-3,
            early_stopping=True,
            n_iter_no_change=15,
            max_iter=500,
            random_state=0,
        ),
    )
    model.fit(X_train, y_train)

    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    pred = model.predict(X_test)
    mae = np.mean(np.abs(pred - y_test))

    print(f"Train R^2 : {train_r2:.4f}")
    print(f"Test  R^2 : {test_r2:.4f}")
    print(f"Test  MAE : {mae:.2f} points")

    out = HERE / "value_model.joblib"
    joblib.dump({"model": model, "feature_names": feature_names}, out)
    print(f"Modèle sauvegardé dans {out}")


if __name__ == "__main__":
    main()
