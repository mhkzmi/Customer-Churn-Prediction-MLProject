import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(df: pd.DataFrame):
    """
    Preprocess raw Telco Customer Churn data.

    Steps:
    - Drop identifier column
    - Convert TotalCharges to numeric
    - Handle missing values
    - Encode categorical features
    - Scale numerical features

    Returns:
        X (pd.DataFrame): processed features
        y (pd.Series): target variable
    """

    df = df.copy()

    # Drop ID column
    df.drop(columns=["customerID"], inplace=True)

    # Convert TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Separate target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Identify feature types
    categorical_features = X.select_dtypes(include="object").columns
    numerical_features = X.select_dtypes(exclude="object").columns

    # One-hot encoding
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    # Scaling numerical features
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    # Final safety check
    X = X.fillna(0)

    return X, y
