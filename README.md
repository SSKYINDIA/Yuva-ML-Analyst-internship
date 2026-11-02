# Telecom Churn Risk Model

A cost-aware, calibrated, and explainable machine-learning model designed for a fictional Chennai-based telecom operator. This repository forms part of the **Yuva ML Analyst Internship** deliverables.

## 📌 Project Overview
Customer churn is a critical challenge in the telecommunications industry. This project implements an end-to-end machine learning pipeline to identify high-risk churn candidates, allowing the operator to deploy targeted retention strategies proactively. The system focuses heavily on data cleaning robustness, feature engineering, and class imbalance mitigation.

## 📂 Repository Structure
*   **`Telecom_Churn_Risk_Model.ipynb`**: The main development notebook containing exploratory data analysis (EDA), model training experiments, validation, and performance evaluation.
*   **`churn_pipeline.py`**: A modular, reusable production script containing custom scikit-learn transformers (`ChurnFeatureEngineer`) and data cleaning blocks used to prevent data leakage.
*   **`churn_model.joblib`**: The finalized serialized scikit-learn pipeline (model weights and preprocessing steps) ready for inference.
*   **`comparison_table.csv`**: Performance tracking across different experimental model configurations.
*   **`lift_chart.png`**: Visual representation of model performance highlighting the true positive capture rate across population deciles.
*   **`Telecom_Churn_Risk_Model_Report.docx`**: Comprehensive business case report outlining methodology, metrics, and deployment recommendations.
*   **`Churn_Drivers_Summary.docx`**: Executive summary focusing on key risk factors driving customer attrition.

## 🚀 Key Technical Features
1. **Data Leakage Safeguards**: Built-in deterministic cleaning routines (`clean_telco`) automatically identify and drop downstream leakage columns (e.g., churn scores, satisfaction ratings, churn reasons) that aren't available at the time of prediction.
2. **Custom Feature Engineering**: Implements a stateless, groupable feature extraction class (`ChurnFeatureEngineer`) providing:
    * **Tenure Lifecycle Banding** (`is_new_customer`, `tenure_band`)
    * **Spend Patterns** (`avg_monthly_spend`, `charge_delta`, `charge_per_service`)
    * **Service Bundle Scores** (`addon_count`, `protect_count`, `stream_count`)
    * **Risk Combinations** (e.g., identifying high-risk cohorts like `fiber_no_support`)
3. **Imbalance Mitigation**: Features a custom `BalancedGB` (Gradient Boosting Classifier wrapper) that natively incorporates sample weighting (`compute_sample_weight`) during the training phase to handle skewed class distributions seamlessly.

## 🛠️ Usage & Setup
To run the inference pipeline or retrain the model, make sure Python 3.x is installed along with the required libraries:

```bash
pip install numpy pandas scikit-learn
```

### Loading and Running the Pipeline
Because the model relies on custom pipeline elements, ensure `churn_pipeline.py` sits in the same working directory as your execution script:

```python
import pandas as pd
import joblib
from churn_pipeline import clean_telco

# 1. Load your raw dataset
raw_data = pd.read_csv("your_new_customer_data.csv")

# 2. Run deterministic cleaning
X_cleaned, _, customer_ids = clean_telco(raw_data)

# 3. Load the pre-trained pipeline
pipeline = joblib.load("churn_model.joblib")

# 4. Generate predictions
churn_probabilities = pipeline.predict_proba(X_cleaned)[:, 1]
predictions = pd.DataFrame({
    'CustomerID': customer_ids,
    'Churn_Risk_Probability': churn_probabilities
})

print(predictions.head())
```
