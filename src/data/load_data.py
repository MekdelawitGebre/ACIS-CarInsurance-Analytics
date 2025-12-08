import pandas as pd
from datetime import datetime

def parse_vehicle_intro(date_str):
    """Parse VehicleIntroDate in month/year format (e.g., '6/2002')"""
    try:
        return pd.to_datetime(date_str, format='%m/%Y', errors='coerce')
    except:
        return pd.NaT

def load_insurance_data(file_path: str = "data/raw/insurance_data.csv") -> pd.DataFrame:
    """
    Load insurance CSV data with proper parsing for dates and mixed types.
    """
    try:
        # Read CSV with pipe separator, without using deprecated date_parser
        df = pd.read_csv(
            file_path,
            sep='|',
            parse_dates=['TransactionMonth'],   # Only parse TransactionMonth here
            dayfirst=True,                      # Optional: if day comes first
            low_memory=False                     # Avoid DtypeWarning for large/mixed columns
        )

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Parse VehicleIntroDate separately
        if 'VehicleIntroDate' in df.columns:
            df['VehicleIntroDate'] = df['VehicleIntroDate'].apply(parse_vehicle_intro)

        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"{file_path} not found.")
    except ValueError as e:
        raise ValueError(f"Error reading CSV: {e}")
