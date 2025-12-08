import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import joblib


def ensure_directories():
    """Create folders for models and reports."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/modeling", exist_ok=True)
    os.makedirs("results", exist_ok=True)


def load_claim_data(file_path="data/raw/insurance_data.csv"):
    """Load and clean insurance claim data."""
    df = pd.read_csv(file_path, sep="|", low_memory=False)
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df["TotalPremium"] = pd.to_numeric(df["TotalPremium"], errors="coerce")
    df["TotalClaims"] = pd.to_numeric(df["TotalClaims"], errors="coerce")
    df = df[df["TotalClaims"].notna()]
    df = df[df["TotalClaims"] > 0]
    return df


def prepare_features(df):
    """Prepare numerical and categorical features."""
    num_cols = ["TotalPremium", "Cylinders", "cubiccapacity", "kilowatts", "NumberOfDoors", "CustomValueEstimate"]
    cat_cols = ["Province", "VehicleType", "make", "Model", "Gender", "CoverCategory", "CoverType"]

    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    X = df[num_cols + cat_cols]
    y = df["TotalClaims"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    return X, y, preprocessor


def train_and_evaluate_models(X, y, preprocessor):
    """Train Linear Regression, Random Forest, and XGBoost models."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=150, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=120, learning_rate=0.1, random_state=42)
    }

    results = []
    for name, model in models.items():
        print(f"Training {name}...")
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)

        print(f"{name}: RMSE={rmse:.2f}, R²={r2:.2f}")
        results.append({"Model": name, "RMSE": rmse, "R2": r2})

        joblib.dump(pipe, f"models/{name}_pipeline.pkl")

    results_df = pd.DataFrame(results)
    results_df.to_csv("reports/modeling/model_performance_summary.csv", index=False)
    return results_df


def shap_analysis(best_model_name, X_sample):
    """Perform SHAP feature importance analysis."""
    print(f"\nPerforming SHAP analysis for {best_model_name}...")
    pipe = joblib.load(f"models/{best_model_name}_pipeline.pkl")
    model = pipe.named_steps["model"]
    preprocessor = pipe.named_steps["preprocessor"]

    X_trans = preprocessor.transform(X_sample)
    explainer = shap.Explainer(model, X_trans)
    shap_values = explainer(X_trans)

    shap.summary_plot(shap_values, X_trans, feature_names=preprocessor.get_feature_names_out(), show=False)
    plt.title(f"SHAP Feature Importance - {best_model_name}")
    plt.savefig("reports/modeling/shap_feature_importance.png", bbox_inches="tight")
    plt.close()


def save_report(results_df):
    """Save model evaluation summary as text."""
    best_model = results_df.sort_values(by="RMSE").iloc[0]
    with open("reports/modeling/predictive_pricing_report.txt", "w") as f:
        f.write("=== Predictive Pricing and Risk Modeling Report ===\n\n")
        f.write(results_df.to_string(index=False))
        f.write(f"\n\nBest Model: {best_model['Model']} (RMSE={best_model['RMSE']:.2f}, R²={best_model['R2']:.2f})\n")
        f.write("\nSHAP feature importance identifies the most influential risk factors driving claim severity.\n")


if __name__ == "__main__":
    ensure_directories()
    df = load_claim_data()
    X, y, preprocessor = prepare_features(df)
    results_df = train_and_evaluate_models(X, y, preprocessor)
    save_report(results_df)

    best_model = results_df.loc[results_df["RMSE"].idxmin(), "Model"]
    shap_analysis(best_model, X.sample(200, random_state=42))
    print("\n✅ Predictive modeling completed successfully. Reports saved in /reports/modeling/")
