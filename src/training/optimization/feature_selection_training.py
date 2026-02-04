from pathlib import Path
import matplotlib.pyplot as plt

from src.models.logistic_model import build_model as build_logistic
from src.training.train import train_model
from src.evaluation.evaluate import evaluate_model
from src.training.optimization.threshold import threshold_sweep
from src.training.optimization.feature_importance import get_logistic_feature_importance


def _save_feature_importance_custom(feature_importance, prefix: str, results_dir: str = "../results"):
    """
    Save feature importance CSV + top-15 plot WITHOUT overwriting other artifacts.
    """
    metrics_dir = Path(results_dir) / "metrics"
    figures_dir = Path(results_dir) / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    feature_importance.to_csv(metrics_dir / f"{prefix}_feature_importance.csv", index=False)

    # Plot top 15
    top_features = feature_importance.head(15)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.gca().invert_yaxis()
    plt.title(f"Top 15 Important Features ({prefix})")
    plt.xlabel("Coefficient Magnitude")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{prefix}_feature_importance.png")
    plt.close()


def train_test_logistic_with_selected_features(
    X,
    y,
    selected_features,
    model_name: str = "logistic_selected_features",
    results_dir: str = "../results"
):
    """
    Step 3: Train/Test logistic regression using only SELECTED_FEATURES
    and save:
      - metrics JSON (evaluate_model)
      - threshold sweep CSV
      - feature importance CSV + top15 plot
    """

    # apply selected features (safety check)
    missing = [c for c in selected_features if c not in X.columns]
    if missing:
        raise ValueError(
            "Some SELECTED_FEATURES are not in preprocessed X columns:\n"
            + "\n".join(missing)
        )

    X_sel = X[selected_features]

    # train/test logistic
    model = build_logistic()
    model, X_train, X_test, y_train, y_test = train_model(model, X_sel, y)

    # evaluation metrics
    evaluate_model(model, X_test, y_test, model_name=model_name, results_dir=results_dir)

    # threshold sweep
    thr = threshold_sweep(model=model, X_test=X_test, y_test=y_test)
    thr_path = Path(results_dir) / "metrics" / f"{model_name}_threshold_sweep.csv"
    thr.to_csv(thr_path, index=False)

    best_row = thr.loc[
        thr["f1"].idxmax()
    ]

    print("Best Threshold (selected feature):", best_row["threshold"])
    print("Best F1 (selected feature):", best_row["f1"])
    print("Recall at Best Threshold (selected feature):", best_row["recall"])

    # feature importance
    fi = get_logistic_feature_importance(model, X_sel.columns)
    _save_feature_importance_custom(fi, prefix=model_name, results_dir=results_dir)

    return model, fi, thr
