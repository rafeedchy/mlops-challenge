from __future__ import annotations
import argparse, json, os, shutil
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

SEED = 42
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=200)
    return p.parse_args()

def main():
    args = parse_args()

    np.random.seed(SEED)

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=SEED, stratify=y
    )

    # MLflow store
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("iris-logreg")

    with mlflow.start_run(run_name=f"train_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}") as run:
        params = {
            "C": args.C,
            "max_iter": args.max_iter,
            "test_size": args.test_size,
            "seed": SEED,
        }
        mlflow.log_params(params)

        model = LogisticRegression(
            C=args.C, max_iter=args.max_iter, random_state=SEED
        )
        model.fit(Xtr, ytr)

        y_pred = model.predict(Xte)
        acc = float(accuracy_score(yte, y_pred))
        f1  = float(f1_score(yte, y_pred, average="macro"))
        mlflow.log_metrics({"accuracy": acc, "f1_macro": f1})

        # Save run-local artifacts
        run_dir = Path("runs") / run.info.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, run_dir / "model.pkl")
        stats = {
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "feature_means": np.mean(Xtr, axis=0).tolist(),
            "feature_stds": np.std(Xtr, axis=0).tolist(),
            "metrics": {"accuracy": acc, "f1_macro": f1},
        }
        (run_dir / "training_stats.json").write_text(json.dumps(stats, indent=2))
        (run_dir / "params.json").write_text(json.dumps(params, indent=2))

        # Log to MLflow
        mlflow.log_artifact(str(run_dir / "model.pkl"))
        mlflow.log_artifact(str(run_dir / "training_stats.json"))
        mlflow.log_artifact(str(run_dir / "params.json"))

        # coping the latest model and stats to ./artifacts for the API to load easily
        shutil.copy(run_dir / "model.pkl", ARTIFACTS_DIR / "model.pkl")
        shutil.copy(run_dir / "training_stats.json", ARTIFACTS_DIR / "training_stats.json")

        print(f"[OK] run_id={run.info.run_id} accuracy={acc:.4f} f1_macro={f1:.4f}")

if __name__ == "__main__":
    main()
