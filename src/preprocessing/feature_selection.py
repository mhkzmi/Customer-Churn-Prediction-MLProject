# src/preprocessing/feature_selection.py

SELECTED_FEATURES = [
    # Contract
    "Contract_Two year",
    "Contract_One year",

    # Core numerical
    "tenure",
    "MonthlyCharges",
    "TotalCharges",

    # Internet & services
    "InternetService_Fiber optic",
    "InternetService_No",
    "OnlineSecurity_Yes",
    "TechSupport_Yes",

    # Billing & usage
    "PaperlessBilling_Yes",
    "PaymentMethod_Electronic check",

    # Entertainment
    "StreamingTV_Yes",
    "StreamingMovies_Yes",

    # Phone
    "PhoneService_Yes",
    "MultipleLines_Yes",

    # Demographic (kept one)
    "Dependents_Yes"
]