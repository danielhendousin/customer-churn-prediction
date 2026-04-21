# Customer Churn Prediction System

A portfolio-ready machine learning project that predicts customer churn from tabular customer data.

## Highlights
- End-to-end ML workflow for binary classification
- Data cleaning and preprocessing pipeline
- Model training with Random Forest
- Evaluation with accuracy, precision, recall, F1, and ROC-AUC
- Confusion matrix and feature importance chart export
- GitHub-ready structure for university applications

## Project Structure
```text
customer_churn_prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── sample_data.csv
├── src/
│   ├── main.py
│   └── utils.py
└── assets/
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── classification_report.txt
```

## Dataset
This repo includes a small synthetic `sample_data.csv` so the project runs immediately.

You can later replace it with a real dataset such as Telco Customer Churn by keeping the target column name as `Churn`.

## How to Run
```bash
pip install -r requirements.txt
python src/main.py --data sample_data.csv
```

## Output
The script:
- trains a churn model
- prints evaluation metrics
- saves charts to `assets/`
- saves a classification report to `assets/classification_report.txt`

## 📊 Model Evaluation

### Confusion Matrix
![Confusion Matrix](assets/confusion_matrix.png)

### Feature Importance
![Feature Importance](assets/feature_importance.png)

The model performance is evaluated using classification metrics and visualized through confusion matrix and feature importance analysis.