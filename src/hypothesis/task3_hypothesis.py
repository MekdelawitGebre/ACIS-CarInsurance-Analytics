import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind

# ----------------------
# Load Data Safely
# ----------------------
def load_data(file_path="data/raw/insurance_data.csv"):
    """
    Load insurance data from CSV safely.
    Handles pipe-delimited data, parses dates, and skips malformed lines.
    """
    df = pd.read_csv(
        file_path,
        sep='|',
        parse_dates=['TransactionMonth', 'VehicleIntroDate'],
        dayfirst=True,
        dtype=str,        # Read all as string to avoid dtype issues
        low_memory=False,
        on_bad_lines='skip'
    )
    
    # Convert numeric columns
    numeric_cols = ['TotalPremium', 'TotalClaims']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# ----------------------
# Compute KPIs
# ----------------------
def compute_kpis(df):
    """
    Compute Claim Frequency, Claim Severity, and Margin
    """
    # Add flag if claim exists
    df['HasClaim'] = df['TotalClaims'] > 0
    
    # Claim Frequency: proportion of policies with at least 1 claim
    claim_frequency = df['HasClaim'].mean()
    
    # Claim Severity: average claim amount given a claim occurred
    claim_severity = df.loc[df['HasClaim'], 'TotalClaims'].mean()
    
    # Margin: TotalPremium - TotalClaims
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    
    return df, claim_frequency, claim_severity

# ----------------------
# Statistical Tests
# ----------------------
def test_province_risk(df):
    """
    Test H0: No risk difference across provinces
    Using Chi-squared for claim frequency and ANOVA (t-test) for claim severity
    """
    print("\n--- Province Risk Analysis ---")
    provinces = df['Province'].unique()
    
    # Claim Frequency Chi-squared
    freq_table = pd.crosstab(df['Province'], df['HasClaim'])
    chi2, p, _, _ = chi2_contingency(freq_table)
    print(f"Claim Frequency Chi2 Test p-value: {p:.4f}")
    if p < 0.05:
        print("Reject H0: Risk differs across provinces.")
    else:
        print("Fail to reject H0: No significant frequency difference across provinces.")
    
    # Claim Severity ANOVA (pairwise t-tests simplified)
    severity_groups = [df.loc[df['Province']==prov, 'TotalClaims'].dropna() for prov in provinces]
    # Use simple pairwise t-test between first 2 provinces for demonstration
    if len(severity_groups) >= 2:
        t_stat, p_val = ttest_ind(severity_groups[0], severity_groups[1], equal_var=False)
        print(f"Claim Severity T-test between {provinces[0]} and {provinces[1]} p-value: {p_val:.4f}")
        if p_val < 0.05:
            print("Reject H0: Claim severity differs between provinces.")
        else:
            print("Fail to reject H0: No significant severity difference between provinces.")

def test_zipcode_margin(df):
    """
    Test H0: No significant margin difference between zip codes
    """
    print("\n--- ZipCode Margin Analysis ---")
    zip_codes = df['PostalCode'].unique()
    if len(zip_codes) >= 2:
        group1 = df.loc[df['PostalCode']==zip_codes[0], 'Margin'].dropna()
        group2 = df.loc[df['PostalCode']==zip_codes[1], 'Margin'].dropna()
        t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
        print(f"Margin T-test between zip {zip_codes[0]} and {zip_codes[1]} p-value: {p_val:.4f}")
        if p_val < 0.05:
            print("Reject H0: Margin differs between zip codes.")
        else:
            print("Fail to reject H0: No significant margin difference.")

def test_gender_risk(df):
    """
    Test H0: No significant risk difference between Women and Men
    """
    print("\n--- Gender Risk Analysis ---")
    genders = df['Gender'].unique()
    if 'Male' in genders and 'Female' in genders:
        male_claims = df.loc[df['Gender']=='Male', 'TotalClaims'].dropna()
        female_claims = df.loc[df['Gender']=='Female', 'TotalClaims'].dropna()
        t_stat, p_val = ttest_ind(male_claims, female_claims, equal_var=False)
        print(f"Claim Amount T-test Male vs Female p-value: {p_val:.4f}")
        if p_val < 0.05:
            print("Reject H0: Risk differs between genders.")
        else:
            print("Fail to reject H0: No significant risk difference between genders.")
    else:
        print("Insufficient gender data to run test.")

# ----------------------
# Main Execution
# ----------------------
if __name__ == "__main__":
    print("Loading data...")
    df = load_data("data/raw/insurance_data.csv")
    df, claim_freq, claim_sev = compute_kpis(df)
    print(f"\nOverall Claim Frequency: {claim_freq:.4f}")
    print(f"Overall Claim Severity: {claim_sev:.2f}")

    # Run all statistical tests
    test_province_risk(df)
    test_zipcode_margin(df)
    test_gender_risk(df)

    print("\nTask 3 hypothesis tests completed.")
