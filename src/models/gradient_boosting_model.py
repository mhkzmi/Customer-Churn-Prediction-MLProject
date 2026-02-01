from sklearn.ensemble import GradientBoostingClassifier



def build_model(random_state: int = 42):
    """
    Build Gradient Boosting model.
    """
    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=3,
        random_state=random_state
    )
    return model
