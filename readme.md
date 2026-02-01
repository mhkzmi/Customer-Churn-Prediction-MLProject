# Predicting Customer Churn in Telecom Industry using Feature Engineering and Classical ML

The goal of this project is to predict customer churn in a telecom company using behavioral and contract data, 
with the objective of achieving a strong predictive performance (targeting an F1-score around 0.75).

## Dataset

The project uses the Telco Customer Churn dataset, which contains customer demographic, service usage, 
and contract-related features for predicting churn behavior.

## Data Setup
Download the Telco Customer Churn dataset from Kaggle and place it in:
data/raw/Telco-Customer-Churn.csv


## Models
- Logistic Regression (final model)
- Random Forest
- Random Forest (class-balanced)


## Evaluation
Model performance was evaluated using F1-score due to class imbalance.
The final Logistic Regression model achieved an F1-score of approximately 0.60
for predicting customer churn.


## Future Work
- Hyperparameter tuning
- Trying gradient boosting models (XGBoost, LightGBM)
- Cost-sensitive learning
- Feature selection and interaction terms
