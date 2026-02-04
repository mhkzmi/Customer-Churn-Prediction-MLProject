from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import make_scorer, f1_score

f1_scorer = make_scorer(
    f1_score
    # ,pos_label="Yes"
)

def tune_gradient_boosting(X_train, y_train):

    param_dist = {
        "n_estimators": [100, 150, 200, 250],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "min_samples_split": [2, 5, 10],
        "subsample": [0.7, 0.8, 1.0],
    }

    model = GradientBoostingClassifier(random_state=42)

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=25,
        scoring=f1_score,
        cv=5,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_