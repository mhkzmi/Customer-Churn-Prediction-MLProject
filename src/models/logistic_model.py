from sklearn.linear_model import LogisticRegression


def build_model(random_state: int = 42):
    """
    Build Logistic Regression baseline model.
    """
    model = LogisticRegression(
        max_iter=1000,
        class_weight=None,
        random_state=random_state
    )
    return model
