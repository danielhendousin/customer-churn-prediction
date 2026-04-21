
import argparse
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from utils import ensure_dir, save_confusion_matrix, save_feature_importance

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Churn" not in df.columns:
        raise ValueError("Dataset must include a 'Churn' target column.")
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["Churn"] = df["Churn"].astype(str).str.strip().str.lower().map(
        {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
    )
    if df["Churn"].isna().any():
        raise ValueError("Unable to normalize some values in the 'Churn' column.")
    return df

def build_pipeline(X: pd.DataFrame):
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_split=6,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline, numeric_features, categorical_features

def get_feature_names(pipeline, numeric_features, categorical_features):
    preprocessor = pipeline.named_steps["preprocessor"]
    cat_names = []
    if categorical_features:
        onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = onehot.get_feature_names_out(categorical_features).tolist()
    return numeric_features + cat_names

def main():
    parser = argparse.ArgumentParser(description="Customer churn prediction baseline project.")
    parser.add_argument("--data", type=str, default="sample_data.csv", help="Path to CSV dataset.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion.")
    args = parser.parse_args()

    output_dir = ensure_dir("assets")
    df = load_data(args.data)

    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    pipeline, numeric_features, categorical_features = build_pipeline(X)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Customer Churn Prediction Results ===")
    print(f"Samples: {len(df)}")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1 Score : {f1:.3f}")
    print(f"ROC-AUC  : {roc_auc:.3f}")

    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    save_confusion_matrix(cm, labels=["No Churn", "Churn"], output_path=output_dir / "confusion_matrix.png")

    feature_names = get_feature_names(pipeline, numeric_features, categorical_features)
    importances = pipeline.named_steps["model"].feature_importances_
    save_feature_importance(feature_names, importances, output_path=output_dir / "feature_importance.png")

    print("\nSaved report to: assets/classification_report.txt")
    print("Saved charts to: assets/confusion_matrix.png and assets/feature_importance.png")

if __name__ == "__main__":
    main()
