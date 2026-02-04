# Customer Churn Prediction — End-to-End Machine Learning Project

The project emphasizes **engineering-driven experimentation**, reproducible pipelines, and business-oriented evaluation rather than notebook-only modeling.

## Project Objective

Customer churn is one of the most critical challenges in subscription-based businesses.  
The goal of this project is to build a reliable machine learning model capable of identifying customers who are likely to leave the service.

Special focus was placed on:

- Maximizing recall for churn detection
- Maintaining strong F1-score
- Building a modular and reproducible ML pipeline
- Making engineering-driven decisions through controlled experiments

---

## Business Impact

Improving churn detection enables telecom providers to proactively retain high-risk customers through targeted interventions such as personalized offers or contract adjustments.

By prioritizing recall, the selected model minimizes the risk of missing churn-prone customers — a scenario often associated with significant revenue loss.

---

## Dataset
The project uses the **Telco Customer Churn** dataset, which contains customer-level information such as:
- Demographics
- Subscription services
- Contract type and billing details

The target variable is **Churn**, indicating whether a customer left the service.

---

## Final Model

After multiple optimization experiments, the following configuration was selected as the production candidate:

**Model:** Logistic Regression  
**Features:** Full feature set  
**Optimized Threshold:** 0.31

The model was selected based on recall-driven evaluation to better align with real-world churn mitigation strategies.

### Performance:
- **F1-score:** ~0.62  
- **Recall:** ~0.75  

Threshold tuning significantly improved churn detection by prioritizing recall — a critical metric in customer retention scenarios.

---

## Project Structure
The project follows a modular design:

├── src/  
│   ├── preprocessing/    
│   ├── models/           
│   ├── training/         
│   ├── evaluation/       
│   ├── run_pipeline.py   
│   ├── run_optimization.py

This structure ensures separation of concerns and improves reproducibility and maintainability.

---

## Exploratory Data Analysis (EDA)
EDA was performed to understand the data distribution and key drivers of churn, including:
- Churn distribution and class imbalance analysis
- Numerical feature distributions (tenure, MonthlyCharges, TotalCharges)
- Categorical feature analysis (Contract, InternetService, PaymentMethod)
- Correlation analysis between numerical features

More than six meaningful visualizations were created to support these analyses.

---

## Models
The following models were implemented and compared:

- **Logistic Regression** (baseline and final model)
- **Random Forest**
- **Random Forest (class-balanced)**
- **Gradient Boosting Classifier**

Logistic Regression was selected as the final model due to its stable performance,
interpretability, and robustness under class imbalance.

---

## Evaluation
Due to the imbalanced nature of the dataset, **F1-score** was chosen as the primary evaluation metric.
Additional evaluation included:
- Confusion Matrix
- ROC Curve and ROC-AUC

All evaluation metrics and plots are stored in the `results/` directory

---

## Results

The results/ directory contains:

- JSON files with evaluation metrics for each model
- Confusion matrix and ROC curve plots

These results allow direct comparison between models without re-running the pipeline.

---

## Limitations

- Hyperparameter tuning was not exhaustively performed
- Feature selection was evaluated but did not improve recall
- Advanced ensemble models were explored only at a baseline level

The listed limitations also define clear directions for future improvements
and provide a natural extension of the current pipeline.

---

## Optimization Experiments

Several controlled experiments were conducted to improve model performance:

### ✔ Hyperparameter Tuning
Gradient boosting models were explored but did not outperform logistic regression.

### ✔ Threshold Optimization
Adjusting the decision threshold improved recall significantly without sacrificing overall model balance.

### ✔ Feature Importance Analysis
Model coefficients were analyzed to understand key churn drivers.

### ✔ Feature Selection (Experiment)
Reducing the feature set slightly simplified the model but resulted in lower recall.  
Given the business importance of identifying churned customers, the full feature set was retained.


## Key Insight

Experimentation showed that **threshold optimization delivered more impact than feature reduction**, highlighting the importance of decision policy in churn prediction systems.


## How to Run

### 1. Download Dataset
Place the dataset inside:

data/raw/Telco-Customer-Churn.csv

### 2. Train Final Pipeline
```bash
python src/run_pipeline.py

python src/run_optimization.py
```


## Future Improvements

- Cost-sensitive learning
- Advanced ensemble models
- Model calibration
- Deployment as a prediction service

---

## Engineering Highlights

- Modular ML architecture
- Experiment isolation
- Reproducible pipelines
- Artifact tracking
- Decision-driven model selection