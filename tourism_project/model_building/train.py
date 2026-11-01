import os
import joblib
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)

# 1. Load the train and test splits produced by the previous job
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").values.ravel()
ytest = pd.read_csv("ytest.csv").values.ravel()

# Identify numerical and categorical columns
categorical_cols = Xtrain.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = Xtrain.select_dtypes(exclude=["object"]).columns.tolist()

# 2. Define preprocessing (with missing value imputation) and XGBoost model pipeline
num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore", sparse_output=False)
)

preprocessor = make_column_transformer(
    (num_pipeline, numerical_cols),
    (cat_pipeline, categorical_cols)
)

xgb_model = xgb.XGBClassifier(random_state=42, eval_metric="logloss")
pipeline = make_pipeline(preprocessor, xgb_model)

# Define hyperparameter grid (including scale_pos_weight for class imbalance)
param_grid = {
    "xgbclassifier__n_estimators": [50, 100, 150],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__learning_rate": [0.05, 0.1, 0.2],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__scale_pos_weight": [1, 3, 4]
}

# 3. Configure MLflow tracking and tune model
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Tourism_Package_Prediction")

with mlflow.start_run():
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )
    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_

    # 4. Evaluate the best model
    y_pred = best_model.predict(Xtest)
    acc = accuracy_score(ytest, y_pred)
    prec = precision_score(ytest, y_pred)
    rec = recall_score(ytest, y_pred)
    f1 = f1_score(ytest, y_pred)

    # Log tuned parameters and evaluation metrics to MLflow
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })
    mlflow.sklearn.log_model(best_model, "best_tourism_model")

    print("Best Parameters:", grid_search.best_params_)
    print(f"Test Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(ytest, y_pred))

    # 5. Save the best model into tourism_project/deployment/
    os.makedirs("tourism_project/deployment", exist_ok=True)
    model_save_path = "tourism_project/deployment/model.joblib"
    joblib.dump(best_model, model_save_path)
    print(f"Best model saved to {model_save_path}")
