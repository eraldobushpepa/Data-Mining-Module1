import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc

def task_3_hierarchical_analysis(input_filepath):
    """
    Performs Hierarchical Clustering (Task 3.3).
    - Generates a dendrogram
    - Fits a model and analyzes the results.
    """
    print(f"--- Task 3: Hierarchical Clustering ---")

    # --- 1. Configuration ---
    
    # Use the same 3-feature set that gave us our best K-Means score
    attributes_to_use=[
        'GameWeight',
        'MfgPlaytime',
        'NumOwned', 
    ]
    
    # We will test k=4 to compare directly with our best K-Means result
    N_CLUSTERS = 4 
    
    plot_directory = "clustering_plots"
    os.makedirs(plot_directory, exist_ok=True)
    print(f"Saving plots to: {plot_directory}")

    # --- 2. Load Data ---
    print(f"\nLoading dataset from: {input_filepath}...")
    try:
        df = pd.read_csv(input_filepath)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"--- Error: file not found at '{input_filepath}' ---")
        return

    # Filter to our 3 selected features
    available_cols = [col for col in attributes_to_use if col in df.columns]
    features_df = df[available_cols]
    print(f"Clustering will use {len(available_cols)} features: {available_cols}")

    # --- 3. Generate Dendrogram ---
    # To avoid crashing by plotting all 21k points, we plot on a 1% sample (219 points)
    print(f"\n--- 3. Generating Dendrogram (on a 1% sample) ---")
    try:
        # Using a small sample for the dendrogram is standard practice
        sample_df = features_df.sample(frac=0.01, random_state=42)
        
        plt.figure(figsize=(15, 10))
        plt.title("Hierarchical Clustering Dendrogram (1% Sample, Linkage='ward')")
        plt.xlabel("Data Points (Sampled)")
        plt.ylabel("Distance (Euclidean)")
        
        # 'ward' linkage minimizes variance, similar to K-Means
        dend = shc.dendrogram(shc.linkage(sample_df, method='ward'))
        
        plot_filename = os.path.join(plot_directory, "hierarchical_dendrogram.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Dendrogram saved to: {plot_filename}")

    except Exception as e:
        print(f"Error generating dendrogram: {e}")

    # --- 4. Run Final Hierarchical Model ---
    print(f"\n--- 4. Running Final Model (on full dataset) ---")
    print(f"Using parameters: n_clusters={N_CLUSTERS}, linkage='ward'")
    
    # We use AgglomerativeClustering for the actual fitting
    model = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward')
    clusters = model.fit_predict(features_df)
    
    df['Cluster'] = clusters
    
    # --- 5. Analyze Clusters ---
    print(f"\n--- 5. Cluster Analysis ---")
    
    try:
        silhouette = silhouette_score(features_df, clusters)
        print(f"Silhouette Score: {silhouette:.4f}")

        print("\n--- Cluster Centroids (Mean Values) ---")
        cluster_analysis = df.groupby('Cluster')[available_cols].mean()
        cluster_analysis['Cluster_Size'] = df['Cluster'].value_counts()
        
        # Transpose for clean printing
        print(cluster_analysis.T.to_string())
             
    except Exception as e:
        print(f"Error in analysis: {e}")
        print("This can happen if clustering is very imbalanced (e.g., 1 cluster with 1 point).")


    print("\n--- Hierarchical Analysis Complete ---")

if __name__ == "__main__":
    input_dataset = "dm1_prepared_dataset.csv"
    task_3_hierarchical_analysis(input_dataset)