# src/hypothesis/task3_hypothesis.py

import pandas as pd
import numpy as np
from scipy import stats

# Load dataset via DVC (Task 2)
def load_data(file_path="data/raw/insurance_data.csv"):
    df = pd.read_csv(file_path)
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    return df

# Compute key metrics
def compute_metrics(df):
    df['ClaimFrequency'] = np.where(df['TotalClaims'] > 0, 1, 0)
    df['ClaimSeverity'] = df.apply(lambda row: row['TotalClaims'] if row['TotalClaims'] > 0 else np.nan, axis=1)
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    return df

# Hypothesis tests

# H0: No risk differences across provinces
def test_province_risk(df):
    provinces = df['Province'].dropna().unique()
    results = {}
    for province in provinces:
        sub = df[df['Province'] == province]['ClaimFrequency']
        others = df[df['Province'] != province]['ClaimFrequency']
        t_stat, p_val = stats.ttest_ind(sub, others, equal_var=False, nan_policy='omit')
        results[province] = p_val
    return results

# H0: No risk differences between zip codes
def test_zipcode_risk(df):
    zipcodes = df['PostalCode'].dropna().unique()
    results = {}
    for zipcode in zipcodes[:10]:  # limit to first 10 zip codes for speed
        sub = df[df['PostalCode'] == zipcode]['ClaimFrequency']
        others = df[df['PostalCode'] != zipcode]['ClaimFrequency']
        t_stat, p_val = stats.ttest_ind(sub, others, equal_var=False, nan_policy='omit')
        results[zipcode] = p_val
    return results

# H0: No margin difference between zip codes
def test_zipcode_margin(df):
    zipcodes = df['PostalCode'].dropna().unique()
    results = {}
    for zipcode in zipcodes[:10]:
        sub = df[df['PostalCode'] == zipcode]['Margin']
        others = df[df['PostalCode'] != zipcode]['Margin']
        t_stat, p_val = stats.ttest_ind(sub, others, equal_var=False, nan_policy='omit')
        results[zipcode] = p_val
    return results

# H0: No risk difference between Women and Men
def test_gender_risk(df):
    male = df[df['Gender'].str.lower() == 'male']['ClaimFrequency']
    female = df[df['Gender'].str.lower() == 'female']['ClaimFrequency']
    t_stat, p_val = stats.ttest_ind(male, female, equal_var=False, nan_policy='omit')
    return p_val

# Main execution
if __name__ == "__main__":
    df = load_data()
    df = compute_metrics(df)

    province_results = test_province_risk(df)
    zipcode_risk_results = test_zipcode_risk(df)
    zipcode_margin_results = test_zipcode_margin(df)
    gender_risk_result = test_gender_risk(df)

    print("Province Claim Frequency p-values:", province_results)
    print("Sample ZipCode Claim Frequency p-values:", zipcode_risk_results)
    print("Sample ZipCode Margin p-values:", zipcode_margin_results)
    print("Gender Claim Frequency p-value:", gender_risk_result)

    # Interpretation
    print("\nInterpretation:")
    for prov, p in province_results.items():
        print(f"Province {prov}: {'Reject H0' if p<0.05 else 'Fail to reject H0'} (p={p:.4f})")
    if gender_risk_result < 0.05:
        print("Gender difference significant: Reject H0")
    else:
        print("No gender difference: Fail to reject H0")
