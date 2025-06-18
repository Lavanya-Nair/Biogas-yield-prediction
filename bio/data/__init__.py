import pandas as pd
import os

def load_data():
    """
    Load the biogas dataset from Excel file
    """
    try:
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'agstar-livestock-ad-database.xlsx')
        df = pd.read_excel(file_path)
        print("Data loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None