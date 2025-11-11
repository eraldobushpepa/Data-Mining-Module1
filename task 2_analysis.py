import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler # <-- CHANGED
import os

def task_2_preparation(input_filepath, output_filepath):
    """
    Performs all "Data Preparation" (Task 1.2 Cleaning & 1.4 Transformation).
    This script loads the RAW data, fixes all problems identified in
    the analysis script, and saves the final prepared dataset.
    
    This version uses RobustScaler and imputes medium-missing columns.
    """
    print(f"--- Task 2: Data Preparation & Transformation ---")
    
    # --- 1. Load Data ---
    print(f"\nLoading dataset from: {input_filepath}")
    try:
        df = pd.read_csv(input_filepath)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"\n--- Error: file not found ---")
        return


    # --- 2. Clip Extreme Outliers (Using IQR Formula) ---
    # We still do this to clean the data before any transformations
    print("\n--- 2. Clipping Extreme 'YearPublished' Outliers (using IQR) ---")
    
    Q1 = df['YearPublished'].quantile(0.25)
    Q3 = df['YearPublished'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    
    print(f"- Calculated Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
    print(f"- Using formula-based Lower Bound: {lower_bound:.2f}")

    # Find outliers based on the formula
    outlier_count = (df['YearPublished'] < lower_bound).sum()
    
    if outlier_count > 0:
        print(f"- Found {outlier_count} extreme outliers (< {lower_bound:.2f}), setting to NaN.")
        df.loc[df['YearPublished'] < lower_bound, 'YearPublished'] = np.nan
    else:
        print("- No extreme 'YearPublished' outliers found.")

    # --- 3. Handle 'Rating' Column (Task 1.4) ---
    print("\n--- 3. Transforming 'Rating' Column ---")
    if 'Rating' in df.columns and df['Rating'].dtype == 'object':
        rating_map = {'Low': 1, 'Medium': 2, 'High': 3}
        df['Rating_Ordinal'] = df['Rating'].map(rating_map)
        # We will drop the original 'Rating' column later
    
    # --- 4. Drop Useless Columns (Task 1.4) ---
    # This includes identifiers, text, high-missing-value columns, and ranks
    print("\n--- 4. Dropping Useless Columns ---")
    cols_to_drop = [
        # Identifiers / Text
        'BGGId', 'Name', 'Description', 'ImagePath', 'GoodPlayers', 'ComMinPlaytime'
        # High Missing % (from our analysis)
        'Family', 
        # Ranks (Outcome, not Feature)
        'Rank:strategygames', 'Rank:abstracts', 'Rank:familygames',
        'Rank:thematic', 'Rank:cgs', 'Rank:wargames',
        'Rank:partygames', 'Rank:childrensgames',
        # Original 'Rating' text column
        'Rating' 
    ]
    df_transformed = df.drop(columns=cols_to_drop, errors='ignore')
    print(f"- Dropped {len(df.columns) - len(df_transformed.columns)} columns.")

    # --- 5. Impute Remaining Missing Values (Task 1.2) ---
    # This will now also impute 'LanguageEase' and 'ComAgeRec'
    print("\n--- 5. Imputing Remaining Missing Values ---")
    for col in df_transformed.columns:
        if df_transformed[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_transformed[col]):
                median_val = df_transformed[col].median()
                df_transformed[col] = df_transformed[col].fillna(median_val)
                print(f"- Filled missing numeric values in '{col}' with median.")
            elif pd.api.types.is_object_dtype(df_transformed[col]):
                mode_val = df_transformed[col].mode()[0]
                df_transformed[col] = df_transformed[col].fillna(mode_val)
                print(f"- Filled missing text values in '{col}' with mode.")

    # --- 6. Log Transformation (Task 1.4) ---
    print(f"\n--- 6. Applying Log Transform to skewed columns ---")
    
    #Get numeric columns
    numeric_cols=df_transformed.select_dtypes(include=np.number).columns
    
    #Calculate skewnees for all numeric columns, we check columns with >1 unique value
    skewed_cols=[]
    if not numeric_cols.empty:
        skewness=df_transformed[numeric_cols].skew()
        
        #define threshold
        skew_threshold=1.0
        
        #get list of columns where skewness>threshold
        skewed_cols=skewness[abs(skewness) > skew_threshold].index
    
    if not skewed_cols.empty:
        print(f"Found {len(skewed_cols)} skewed columns to transform: {list(skewed_cols)}")
        for col in skewed_cols:
            df_transformed[col]=np.log1p(df_transformed[col])
        print("- Log transformation completed -")
    else:
        print(" - No highly skewed columns to transform")
    

    # --- 7. Robust Scaling (Task 1.4) ---
    print("\n--- 7. Applying RobustScaler (handles outliers) ---")
    numeric_cols_final = df_transformed.select_dtypes(include=np.number)
    
    # Use RobustScaler instead of MinMaxScaler
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(numeric_cols_final)
    
    df_prepared = pd.DataFrame(scaled_data, columns=numeric_cols_final.columns)
    print("- Scaling complete -")

    # --- 8. Save Prepared Dataset ---
    try:
        df_prepared.to_csv(output_filepath, index=False)
        print(f"\n--- SUCCESS ---")
        print(f"Final prepared dataset saved to: {output_filepath}")
        print("This file is ready for Part 2: Clustering.")
    except Exception as e:
        print(f"\n--- Error: Could not save file. {e} ---")

if __name__ == "__main__":
    input_dataset = "dm1_25_26_dataset/DM1_game_dataset.csv"
    output_dataset = "dm1_prepared_dataset.csv"
    task_2_preparation(input_dataset, output_dataset)