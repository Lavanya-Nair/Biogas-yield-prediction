import os
import pandas as pd
from preprocessing import preprocess_data
from models import train_model
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Get the absolute path to the Excel file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(os.path.dirname(current_dir), 'agstar-livestock-ad-database.xlsx')
    
    # Load data
    try:
        df = pd.read_excel(file_path)
        print("Data loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        
        # Preprocess data
        X, y, scaler = preprocess_data(df)
        
        # Train model
        results = train_model(X, y)
        
        # Print results
        print("\nModel Performance:")
        print(f"RMSE: {results['metrics']['rmse']:.2f}")
        print(f"R² Score: {results['metrics']['r2']:.2f}")
        
        # Create visualizations directory if it doesn't exist
        vis_dir = os.path.join(current_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame(
            results['feature_importance'].items(), 
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=True)
        
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.title('Feature Importance in Biogas Prediction')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'feature_importance.png'))
        plt.close()
        
    except FileNotFoundError:
        print(f"Error: Could not find the Excel file at {file_path}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()