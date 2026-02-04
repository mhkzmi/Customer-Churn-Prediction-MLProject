import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def get_logistic_feature_importance(model, feature_names):

    # abs value of coefficients
    importance = np.abs(model.coef_[0])

    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    feature_importance = feature_importance.sort_values(
        by="importance",
        ascending=False
    )

    return feature_importance


def save_feature_importance(feature_importance):

    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # save csv
    feature_importance.to_csv(
        "../results/metrics/logistic_feature_importance.csv",
        index=False
    )

    # plot top 15
    top_features = feature_importance.head(15)

    plt.figure(figsize=(10,6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.gca().invert_yaxis()

    plt.title("Top 15 Important Features (Logistic Regression)")
    plt.xlabel("Coefficient Magnitude")

    plt.tight_layout()

    plt.savefig("../results/figures/logistic_feature_importance.png")
    plt.close()
