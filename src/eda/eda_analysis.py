import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.data.load_data import load_insurance_data
from src.data.preprocess import preprocess_data

def ensure_reports_dir():
    os.makedirs("reports", exist_ok=True)

def data_summary(df: pd.DataFrame):
    print("===== DATA STRUCTURE =====")
    print(df.info())
    print("\n===== DESCRIPTIVE STATISTICS =====")
    print(df.describe().T[['mean','std','min','max']])
    print("\n===== DATA TYPES =====")
    print(df.dtypes.value_counts())

def check_missing_values(df: pd.DataFrame):
    print("\n===== MISSING VALUES =====")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print(missing)
    else:
        print("No missing values detected.")

def univariate_analysis(df: pd.DataFrame):
    ensure_reports_dir()
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    cat_cols = df.select_dtypes(include=['object','category']).columns

    for col in num_cols:
        plt.figure(figsize=(8,4))
        sns.histplot(df[col], bins=50, kde=True, color='steelblue')
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(f"reports/univariate_{col}_hist.png")
        plt.close()

    for col in cat_cols[:6]:
        plt.figure(figsize=(8,4))
        df[col].value_counts().head(10).plot(kind='bar', color='teal')
        plt.title(f"Frequency of {col}")
        plt.tight_layout()
        plt.savefig(f"reports/univariate_{col}_bar.png")
        plt.close()

def bivariate_analysis(df: pd.DataFrame):
    ensure_reports_dir()
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='TotalPremium', y='TotalClaims', hue='Province', data=df)
    plt.title("TotalPremium vs TotalClaims by Province")
    plt.tight_layout()
    plt.savefig("reports/bivariate_premium_claims_scatter.png")
    plt.close()

    num_df = df.select_dtypes(include=['int64','float64'])
    plt.figure(figsize=(10,8))
    sns.heatmap(num_df.corr(), annot=False, cmap='coolwarm', center=0)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("reports/correlation_heatmap.png")
    plt.close()

def trends_over_geography(df: pd.DataFrame):
    ensure_reports_dir()
    geo_summary = df.groupby("Province")[["TotalPremium","TotalClaims"]].mean().sort_values(by="TotalClaims", ascending=False)
    print("\n===== AVERAGE PREMIUM & CLAIMS BY PROVINCE =====")
    print(geo_summary)

    plt.figure(figsize=(12,6))
    sns.barplot(x=geo_summary.index, y=geo_summary["TotalClaims"], color='coral')
    plt.title("Average TotalClaims by Province")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("reports/trends_claims_province.png")
    plt.close()

def detect_outliers(df: pd.DataFrame):
    ensure_reports_dir()
    num_cols = ['TotalPremium','TotalClaims','CustomValueEstimate']
    for col in num_cols:
        if col in df.columns:
            plt.figure(figsize=(8,4))
            sns.boxplot(x=df[col], color='orchid')
            plt.title(f"Outlier Detection: {col}")
            plt.tight_layout()
            plt.savefig(f"reports/outlier_{col}_box.png")
            plt.close()

def creative_visualizations(df: pd.DataFrame):
    ensure_reports_dir()
    df['ClaimSeverity'] = df['TotalClaims'] / (df['TotalPremium'] + 1)
    plt.figure(figsize=(10,6))
    sns.barplot(x='VehicleType', y='ClaimSeverity', data=df, palette='coolwarm')
    plt.title("Average Claim Severity by Vehicle Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("reports/creative_claim_severity_vehicle.png")
    plt.close()

    if 'TransactionMonth' in df.columns:
        monthly_trend = df.groupby('TransactionMonth')[['TotalClaims']].sum().reset_index()
        plt.figure(figsize=(10,6))
        sns.lineplot(x='TransactionMonth', y='TotalClaims', data=monthly_trend, marker='o', color='green')
        plt.title("Monthly Trend of Total Claims")
        plt.tight_layout()
        plt.savefig("reports/creative_monthly_claims_trend.png")
        plt.close()

    df['LossRatio'] = df['TotalClaims'] / (df['TotalPremium']+1)
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Gender', y='LossRatio', data=df, palette='pastel')
    plt.title("Loss Ratio Distribution by Gender")
    plt.tight_layout()
    plt.savefig("reports/creative_loss_ratio_gender.png")
    plt.close()

def run_eda_pipeline(df: pd.DataFrame):
    print("🚀 Starting EDA Pipeline...")
    data_summary(df)
    check_missing_values(df)
    univariate_analysis(df)
    bivariate_analysis(df)
    trends_over_geography(df)
    detect_outliers(df)
    creative_visualizations(df)
    print("✅ EDA Pipeline completed. Plots saved in 'reports/' folder.")

if __name__ == "__main__":
    df = load_insurance_data("data/raw/insurance_data.csv")
    df = preprocess_data(df)
    run_eda_pipeline(df)
