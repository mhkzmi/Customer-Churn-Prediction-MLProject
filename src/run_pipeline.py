import pandas as pd

from src.preprocessing.data_preprocessing import preprocess_data
from src.training.train import train_model
from src.evaluation.evaluate import evaluate_model

from src.models.logistic_model import build_model as build_logistic
from src.models.random_forest_model import build_model as build_rf
from src.models.gradient_boosting_model import build_model as build_gb

from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_roc_curve
)

def main():
    # Load data
    df = pd.read_csv("../data/raw/Telco-Customer-Churn.csv")

    # Preprocess
    X, y = preprocess_data(df)

    # Models to evaluate
    models = {
        "logistic_regression": build_logistic(),
        "random_forest": build_rf(),
        "gradient_boosting": build_gb(),
    }

    # Train & evaluate
    for name, model in models.items():
        trained_model, X_test, y_test = train_model(model, X, y)
        evaluate_model(
            trained_model,
            X_test,
            y_test,
            model_name=name
        )

        y_pred = trained_model.predict(X_test)

        plot_confusion_matrix(
            y_test,
            y_pred,
            model_name=name
        )

        if hasattr(trained_model, "predict_proba"):
            y_proba = trained_model.predict_proba(X_test)[:, 1]
            plot_roc_curve(
                y_test,
                y_proba,
                model_name=name
            )


if __name__ == "__main__":
    main()
