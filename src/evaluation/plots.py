import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str,
    results_dir: str = "../results"
):
    cm = confusion_matrix(y_true, y_pred, labels=["No", "Yes"])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {model_name}")

    plots_path = Path(results_dir) / "figures"
    plots_path.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(plots_path / f"{model_name}_confusion_matrix.png")
    plt.close()


def plot_roc_curve(
    y_true,
    y_proba,
    model_name: str,
    results_dir: str = "../results"
):
    fpr, tpr, _ = roc_curve(
        (y_true == "Yes").astype(int),
        y_proba
    )
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.legend(loc="lower right")

    plots_path = Path(results_dir) / "figures"
    plots_path.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(plots_path / f"{model_name}_roc_curve.png")
    plt.close()
