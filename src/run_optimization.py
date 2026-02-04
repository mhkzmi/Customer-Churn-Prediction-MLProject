import pandas as pd

from src.preprocessing.data_preprocessing import preprocess_data
from src.training.train import train_model
from src.training.optimization.tuning import tune_gradient_boosting
from src.evaluation.evaluate import evaluate_model


def main():

    print("Loading data...")
    df = pd.read_csv("../data/raw/Telco-Customer-Churn.csv")

    print("Preprocessing...")
    X, y = preprocess_data(df)

    y = y.map({"No": 0, "Yes": 1})

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


if __name__ == "__main__":
    main()
