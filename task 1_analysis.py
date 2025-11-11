import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
from scipy.stats import zscore

def task_1_analysis(input_filepath):
    """
    Performs all "Data Understanding" (Task 1.1, 1.2, 1.3) in one place.
    This script loads the RAW data and generates all stats and plots
    to understand the problems. It does NOT save a new CSV.
    """
    print(f"--- Task 1: Data Understanding & Analysis ---")
    
    # --- 1. Load Data ---
    print(f"\nLoading dataset from: {input_filepath}")
    try:
        df = pd.read_csv(input_filepath)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"\n--- Error: file not found ---")
        print(f"Please ensure the file is at the path: {input_filepath}")
        return
    
    # --- 2. Task 1.1: Initial Exploration ---
    print("\n--- 2. Initial Exploration (Task 1.1) ---")
    num_records, num_attributes = df.shape
    print(f"Total number of records: {num_records}")
    print(f"Total number of attributes: {num_attributes}")
    
    print("\n--- Technical Info (df.info()) ---")
    df.info()

    # --- 2b. Variable Classification ---
    print("\n--- 2b. Variable Semantics & Classification (Automatic) ---")
    
    categorical_cols = []
    continuous_cols = []
    discrete_cols = []
    
    # Define a threshold for "low cardinality"
    # If an int column has <= 10 unique values, we'll treat it as categorical/discrete
    low_cardinality_threshold = 10 
    
    for col in df.columns:
        if df[col].dtype == 'object':
            categorical_cols.append(col)
        elif df[col].dtype == 'float64':
            continuous_cols.append(col)
        elif df[col].dtype == 'int64':
            # This is the "smart" part
            num_unique = df[col].nunique()
            
            if col == 'BGGId': # Rule 1: Special case for known IDs
                categorical_cols.append(col)
            elif num_unique <= 2: # Rule 2: Binary flags (e.g., Cat:War, Kickstarted)
                categorical_cols.append(col)
            elif num_unique <= low_cardinality_threshold: # Rule 3: Low-cardinality ints (e.g., MinPlayers)
                discrete_cols.append(col) # Could also be categorical, but discrete is fine
            else: # Rule 4: High-cardinality ints (e.g., NumOwned, YearPublished, Ranks)
                discrete_cols.append(col)
                
    print(f"Categorical (Object, ID, or Flag): {len(categorical_cols)} columns")
    print(f"  {categorical_cols}") 
    print(f"Continuous (Float): {len(continuous_cols)} columns")
    print(f"  {continuous_cols}") 
    print(f"Discrete (Counts or High-Cardinality Ints): {len(discrete_cols)} columns")
    print(f"  {discrete_cols}") 

    # --- 2c. Distribution Skewness & Kurtosis ---
    print("\n--- 2c. Distribution Statistics (Skew & Kurtosis) ---")
    
    # We analyze columns that are continuous or discrete (and not just flags/IDs)
    # We'll select numeric columns that have more than 1 unique value
    cols_to_analyze = [
        col for col in continuous_cols if df[col].nunique() > 1
    ]
    cols_to_analyze += [
        col for col in discrete_cols if (df[col].nunique() > 1 and col != 'BGGId')
    ]
    
    # Ensure we only check columns that actually exist in the dataframe
    cols_to_analyze = [col for col in cols_to_analyze if col in df.columns]
    
    dist_stats = pd.DataFrame({
        'Skewness': df[cols_to_analyze].skew(),
        'Kurtosis': df[cols_to_analyze].kurtosis()
    })
    
    print("Analyzing numeric columns with > 1 unique value.")
    print("Skewness indicates symmetry (0=Normal, >0=Right-skew, <0=Left-skew)")
    print("Kurtosis indicates 'tailedness' (0=Normal, >0=Heavy-tails, <0=Light-tails)")
    print(dist_stats.to_string())
    print("\nNote: High Skewness/Kurtosis confirms columns like 'YearPublished', 'NumOwned' are not normally distributed.")
    print("This justifies using the IQR method for outlier detection instead of Z-Score.")


    print("\n--- Descriptive Statistics (Numeric) ---")
    print(df.describe())

    # --- 3. Task 1.2: Data Quality Assessment (Finding Problems) ---
    print("\n--- 3. Data Quality Assessment (Task 1.2) ---")
    
    # Create plot directory
    plot_directory = "analysis_plots"
    os.makedirs(plot_directory, exist_ok=True)
    print(f"Saving all analysis plots to '{plot_directory}' folder...")

    # --- 3a. Finding Duplicates ---
    print("\n--- 3a. Finding Duplicates ---")
    num_duplicates = df.duplicated().sum()
    print(f"Total duplicate rows found: {num_duplicates}")
    if num_duplicates > 0:
        print("These should be dropped in the preparation task.")

    # --- 3b. Finding Missing Values ---
    print("\n--- 3b. Finding Missing Values ---")
    missing_counts = df.isnull().sum()
    total_records = len(df)
    missing_percentage = (missing_counts / total_records) * 100
    missing_report = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing Percentage (%)': missing_percentage
    })
    missing_report = missing_report[missing_report['Missing Count'] > 0]
    missing_report = missing_report.sort_values(by='Missing Percentage (%)', ascending=False)
    print("Columns with missing values (Count and percentage):")
    print(missing_report.to_string(formatters={'Missing Percentage (%)': '{:.2f}%'.format}))

    # Generate and save missing value heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(df.isnull(), cbar=True, cbar_kws={'label': 'Missing Data (1=Missing)'}, yticklabels=False, xticklabels=False, cmap='viridis')
    plt.title('Heatmap of Missing Values per Column')
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plot_filename = os.path.join(plot_directory, 'heatmap_missing_values.png')
    
    
    plt.savefig(plot_filename)
    plt.close()
    print(f"- Missing value heatmap saved to: {plot_filename}")

    # --- 3c. Finding Outliers (Statistical) ---
    print("\n--- 3c. Finding Outliers (Statistical) ---")
    numeric_cols = df.select_dtypes(include=np.number)
    
    # Method 1: Z-Score (Best for normally distributed data)
    z_scores = numeric_cols.apply(zscore, nan_policy='omit')
    z_outliers = (np.abs(z_scores) > 3).sum()
    print("\nPotential outliers (Z-Score from <-3 to >3) per column:")
    print(z_outliers[z_outliers > 0].to_string())

    # Method 2: IQR (Best for skewed data, as identified by Skew/Kurtosis)
    print("\nPotential outliers (IQR Method) per column:")
    iqr_outlier_counts = {}
    for col in numeric_cols.columns:
        if not df[col].isnull().all():
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (1.5 * IQR)
            upper_bound = Q3 + (1.5 * IQR)
            
            outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers_count > 0:
                iqr_outlier_counts[col] = outliers_count
                
    iqr_report = pd.Series(iqr_outlier_counts, name="Outlier Count (IQR)").sort_values(ascending=False)
    print(iqr_report.to_string())
    print("\nNote: 'YearPublished' shows 1143 outliers, confirming the -3500 value and other old dates are a problem.")


    # --- 3d. Finding Outliers & Errors (Visual Plots) ---
    print("\n--- 3d. Finding Outliers & Errors (Visual Plots) ---")

    # Box Plots (for outliers)
    for col in numeric_cols.columns:
        plt.figure(figsize=(10, 6))
        if not df[col].isnull().all():
            plt.boxplot(df[col].dropna(), vert=False, whis=1.5)
            plt.title(f'Box Plot for: {col}')
            sanitized_col_name = col.replace(':', '_') #this is to solve rank:problem column
            filename = f'boxplot_{sanitized_col_name}.png'
            plot_filename = os.path.join(plot_directory, filename)
            plt.savefig(plot_filename)
            plt.close()
    print("- Box plots saved.")

    # --- 4. Task 1.3: Distribution & Relationship Analysis ---
    
    # Histograms (for analyzing skewness)
    num_bins = int(1 + 3.322 * math.log10(num_records)) # Sturges' Rule
    print(f"- Generating histograms (using {num_bins} bins)...")
    for col in numeric_cols.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col].dropna(), kde=True, bins=num_bins)
        plt.title(f'Histogram of {col}')
        sanitized_col_name = col.replace(':', '_')
        plot_filename = os.path.join(plot_directory, f'hist_{sanitized_col_name}.png')
        plt.savefig(plot_filename)
        plt.close()
    print("- Histograms saved.")
    
    # Pairwise Correlation Matrix
    corr_matrix = numeric_cols.corr().abs()
    
    #Generate and save heatmap plot
    plt.figure(figsize=(24,20))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f")
    plt.title('Full Correlation Matrix Heatmap')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plot_filename = os.path.join(plot_directory, 'heatmap_correlation_matrix.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"Full correlation heatmap saved to: {plot_filename}")
    
    #print the list
    upper_triangle_mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    highly_correlated = corr_matrix.where(upper_triangle_mask) \
                                 .stack() \
                                 .reset_index()
    highly_correlated.columns = ['Feature 1', 'Feature 2', 'Correlation']
    
    threshold = 0.8
    highly_correlated = highly_correlated[
        (highly_correlated['Correlation'] > threshold) &
        (highly_correlated['Feature 1'] != highly_correlated['Feature 2'])
    ]
    highly_correlated = highly_correlated.sort_values(by='Correlation', ascending=False)
    
    if highly_correlated.empty:
        print(f"\nNo feature pairs found with correlation > {threshold}")
        print("This is good, it means features are not redundant")
    else:
        print(f"\n--- Highly correlated Pairs (Threshold > {threshold}) ---")
        print("These features are redundant, consider dropping one from each pair in Task 2")
        print(highly_correlated.to_string(index=False))

if __name__ == "__main__":
    input_dataset = "dm1_25_26_dataset/DM1_game_dataset.csv"
    task_1_analysis(input_dataset)