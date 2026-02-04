import json
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score
)

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str,
    results_dir: str = "../results"
):
    """
    Evaluate trained model and save metrics.

    Args:
        model: trained sklearn model
        X_test: test features
        y_test: test labels
        model_name (str): name of the model
        results_dir (str): base directory to save results
    """

    # Predictions
    y_pred = model.predict(X_test)

    # Probabilities (for ROC-AUC if available)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(
            (y_test == "Yes").astype(int),
            y_proba
        )
    else:
        roc_auc = None

    f1 = f1_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    metrics = {
        "model": model_name,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "classification_report": report
    }

    # Save metrics
    metrics_path = Path(results_dir) / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    with open(metrics_path / f"{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics
