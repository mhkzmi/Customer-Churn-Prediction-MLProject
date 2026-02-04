import pandas as pd

from src.preprocessing.data_preprocessing import preprocess_data
from src.training.train import train_model
from src.training.optimization.tuning import tune_gradient_boosting
from src.evaluation.evaluate import evaluate_model
from src.training.optimization.threshold import threshold_sweep
from src.evaluation.plots import plot_threshold_vs_f1
from src.models.logistic_model import build_model as build_logistic


def main():

    print("Loading data...")
    df = pd.read_csv("../data/raw/Telco-Customer-Churn.csv")

    print("Preprocessing...")
    X, y = preprocess_data(df)

    # Split using train_model but ignore fitted model
    from sklearn.ensemble import GradientBoostingClassifier

    dummy_model = GradientBoostingClassifier()

    _, X_train, X_test, y_train, y_test = train_model(
        dummy_model,
        X,
        y
    )

    print("Starting hyperparameter tuning...")
    best_model, best_params = tune_gradient_boosting(
        X_train,
        y_train
    )

    print("Best Parameters:")
    print(best_params)

    print("Evaluating tuned model...")
    evaluate_model(
        best_model,
        X_test,
        y_test,
        model_name="gradient_boosting_tuned"
    )

    print("Running threshold optimization for Logistic Regression...")

    log_model = build_logistic()
    log_model, X_train, X_test, y_train, y_test = train_model(log_model, X, y)

    threshold_results = threshold_sweep(
        model=log_model,
        X_test=X_test,
        y_test=y_test
    )

    threshold_results.to_csv(
        "../results/metrics/logistic_threshold_sweep.csv",
        index=False
    )

    best_row = threshold_results.loc[
        threshold_results["f1"].idxmax()
    ]

    print("Best Threshold:", best_row["threshold"])
    print("Best F1:", best_row["f1"])
    print("Recall at Best Threshold:", best_row["recall"])

    plot_threshold_vs_f1(threshold_results)


if __name__ == "__main__":
    main()
