import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def load_data():
    """Load the main 2023 cloud budget dataset."""
    # Use absolute path to avoid any confusion
    csv_path = Path(r"C:\Users\itz4a\OneDrive\Desktop\SEM VI\Cloud\c\clouddataset\cloud_budget_2023_dataset.csv")
    if not csv_path.exists():
        raise SystemExit(f"CSV not found at {csv_path}. Adjust the path in load_data().")

    df = pd.read_csv(csv_path)
    return df


def train_val_test_split_time(df, val_size=0.15, test_size=0.15, date_col="date"):
    """Chronological train/val/test split based on a date column."""
    df = df.sort_values(date_col).reset_index(drop=True)

    n = len(df)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_val - n_test

    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]

    return train, val, test


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Standard regression metrics."""
    mse_val = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse_val),
        "rmse": float(np.sqrt(mse_val)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Binary classification metrics (for e.g. is_anomaly).
    y_prob: predicted probability for class 1.
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # For highly imbalanced datasets, ROC AUC can be useful
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc) if not np.isnan(auc) else np.nan,
        "confusion_matrix": cm,
    }


def save_regression_curves(
    history: Dict[str, list],
    title: str,
    fname_prefix: str,
) -> None:
    """Save training vs validation loss/MAE curves for models that provide history (LSTM)."""
    if not history:
        return

    out_dir = Path("metrics")
    out_dir.mkdir(exist_ok=True)

    # Loss curve
    if "loss" in history:
        plt.figure()
        plt.plot(history["loss"], label="train_loss")
        if "val_loss" in history:
            plt.plot(history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{title} Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{fname_prefix}_loss.png")
        plt.close()

    # MAE curve
    if "mae" in history:
        plt.figure()
        plt.plot(history["mae"], label="train_mae")
        if "val_mae" in history:
            plt.plot(history["val_mae"], label="val_mae")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title(f"{title} MAE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{fname_prefix}_mae.png")
        plt.close()


def build_random_forest(train, val, test, target_col="net_cost", metrics_prefix="rf_regression") -> Dict[str, Any]:
    """Train a RandomForest regressor and save metrics to disk."""
    feature_cols = [c for c in train.columns if c not in {target_col, "date"}]

    X_train = train[feature_cols]
    y_train = train[target_col]

    X_val = val[feature_cols]
    y_val = val[target_col]

    X_test = test[feature_cols]
    y_test = test[target_col]

    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(steps=[("preprocess", preprocessor), ("rf", rf)])

    model.fit(X_train, y_train)

    results: Dict[str, Dict[str, float]] = {}

    def evaluate(split_name, X, y):
        preds = model.predict(X)
        m = regression_metrics(y, preds)
        results[split_name] = m
        print(
            f"[RandomForest-{split_name}] "
            f"MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} R2={m['r2']:.3f}"
        )

    evaluate("train", X_train, y_train)
    evaluate("val", X_val, y_val)
    evaluate("test", X_test, y_test)

    out_dir = Path("metrics")
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame(results).T.to_csv(out_dir / f"{metrics_prefix}_metrics.csv")

    return {"model": model, "metrics": results}


def make_lstm_sequences(df, target_col="net_cost", date_col="date", lookback=14):
    """
    Build LSTM-ready sequences using only time + a few numeric features.
    This is a simple univariate-ish model; you can extend it as needed.
    """
    # Ensure sorted
    df = df.sort_values(date_col).reset_index(drop=True)

    # For simplicity, aggregate to daily total net_cost
    daily = df.groupby(date_col)[target_col].sum().reset_index()
    daily[target_col] = daily[target_col].astype("float32")

    values = daily[target_col].values

    X, y = [], []
    for i in range(len(values) - lookback):
        X.append(values[i:i + lookback])
        y.append(values[i + lookback])

    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)

    # Chronological split on sequences
    n = len(X)
    n_test = int(n * 0.15)
    n_val = int(n * 0.15)
    n_train = n - n_val - n_test

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_lstm(input_length):
    """Define a simple LSTM regression model."""
    model = models.Sequential(
        [
            layers.Input(shape=(input_length, 1)),
            layers.LSTM(64, return_sequences=False),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_lstm(df, target_col="net_cost", date_col="date", lookback=14, epochs=10, batch_size=64, metrics_prefix="lstm_regression"):
    """Train and evaluate an LSTM model for next-day net_cost prediction and save metrics/curves."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = make_lstm_sequences(
        df, target_col=target_col, date_col=date_col, lookback=lookback
    )

    model = build_lstm(input_length=lookback)
    model.summary()

    history_obj = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )

    history = history_obj.history
    save_regression_curves(history, title="LSTM net_cost", fname_prefix=metrics_prefix)

    results: Dict[str, Dict[str, float]] = {}

    def evaluate(split_name, X, y):
        preds = model.predict(X, verbose=0).flatten()
        m = regression_metrics(y, preds)
        results[split_name] = m
        print(
            f"[LSTM-{split_name}] "
            f"MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} R2={m['r2']:.3f}"
        )

    evaluate("train", X_train, y_train)
    evaluate("val", X_val, y_val)
    evaluate("test", X_test, y_test)

    out_dir = Path("metrics")
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame(results).T.to_csv(out_dir / f"{metrics_prefix}_metrics.csv")

    return {"model": model, "metrics": results, "history": history}


def build_rf_classifier(train, val, test, target_col="is_anomaly", metrics_prefix="rf_is_anomaly") -> Dict[str, Any]:
    """
    RandomForest classifier for binary target (e.g. is_anomaly) with
    accuracy, precision, recall, F1, ROC AUC and confusion matrix.
    Metrics for train/val/test are saved to CSV; confusion matrices saved as PNGs.
    """
    if target_col not in train.columns:
        print(f"Target column {target_col} not in dataframe; skipping classifier.")
        return {}

    feature_cols = [c for c in train.columns if c not in {target_col, "date"}]

    X_train = train[feature_cols]
    y_train = train[target_col].astype(int)

    X_val = val[feature_cols]
    y_val = val[target_col].astype(int)

    X_test = test[feature_cols]
    y_test = test[target_col].astype(int)

    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",  # anomalies may be rare
    )

    model = Pipeline(steps=[("preprocess", preprocessor), ("rf_clf", clf)])

    model.fit(X_train, y_train)

    out_dir = Path("metrics")
    out_dir.mkdir(exist_ok=True)

    results: Dict[str, Dict[str, float]] = {}

    def eval_and_plot(split_name, X, y):
        proba = model.predict_proba(X)
        # Handle case where only one class is present in this split
        if proba.shape[1] == 1:
            # Only one column => all samples are the same class; treat prob of class 1 as zeros
            y_prob = np.zeros(len(y), dtype=float)
        else:
            y_prob = proba[:, 1]
        m = classification_metrics(y, y_prob)
        results[split_name] = {k: float(v) for k, v in m.items() if k != "confusion_matrix"}

        print(
            f"[RF-Classifier-{split_name}] "
            f"Acc={m['accuracy']:.3f} Prec={m['precision']:.3f} "
            f"Rec={m['recall']:.3f} F1={m['f1']:.3f} AUC={m['roc_auc']:.3f}"
        )

        cm = m["confusion_matrix"]
        plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"{metrics_prefix} {split_name} Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["0", "1"])
        plt.yticks(tick_marks, ["0", "1"])
        plt.xlabel("Predicted label")
        plt.ylabel("True label")

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.tight_layout()
        plt.savefig(out_dir / f"{metrics_prefix}_{split_name}_confusion_matrix.png")
        plt.close()

    eval_and_plot("train", X_train, y_train)
    eval_and_plot("val", X_val, y_val)
    eval_and_plot("test", X_test, y_test)

    pd.DataFrame(results).T.to_csv(out_dir / f"{metrics_prefix}_metrics.csv")

    return {"model": model, "metrics": results}


def main():
    ...
    print("\n=== Training RandomForest baseline ===")
    rf_out = build_random_forest(train_df, val_df, test_df, target_col="net_cost")
    print("\n=== Training RandomForest classifier for is_anomaly (with confusion matrices) ===")
    clf_out = build_rf_classifier(train_df, val_df, test_df, target_col="is_anomaly")
    print("\n=== Training LSTM time-series model ===")
    lstm_out = train_lstm(df, target_col="net_cost", date_col="date", lookback=14, epochs=10, batch_size=64)
    models_dir = Path("saved_models")
    models_dir.mkdir(exist_ok=True)
    # Save RF regressor
    joblib.dump(rf_out["model"], models_dir / "rf_regressor.joblib")
    # Save RF classifier (if it was created)
    if clf_out:
        joblib.dump(clf_out["model"], models_dir / "rf_is_anomaly_classifier.joblib")
    # Save LSTM
    lstm_out["model"].save(models_dir / "lstm_net_cost.keras")


if __name__ == "__main__":
    main()

