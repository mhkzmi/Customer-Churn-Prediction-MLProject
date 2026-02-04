from sklearn.model_selection import train_test_split


def train_model(
    model,
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Train a given model using train-test split.

    Args:
        model: sklearn-like model with fit()
        X (pd.DataFrame): feature matrix
        y (pd.Series): target
        test_size (float): test split ratio
        random_state (int)

    Returns:
        model: trained model
        X_test, y_test: test data
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test
