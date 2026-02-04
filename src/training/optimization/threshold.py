import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score


def threshold_sweep(model, X_test, y_test, thresholds=np.arange(0.2, 0.81, 0.01)):

    probs = model.predict_proba(X_test)[:, 1]

    results = []

    for t in thresholds:
        preds = (probs >= t).astype(int)

        f1 = f1_score(y_test, preds)
        recall = recall_score(y_test, preds)

        results.append({
            "threshold": t,
            "f1": f1,
            "recall": recall
        })

    return pd.DataFrame(results)
