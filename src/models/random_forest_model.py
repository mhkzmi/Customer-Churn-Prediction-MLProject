from sklearn.ensemble import RandomForestClassifier


def build_model(random_state: int = 42):
    """
    Build Random Forest model.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight=None,
        random_state=random_state,
        n_jobs=-1
    )
    return model
