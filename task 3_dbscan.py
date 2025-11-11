import pandas as pd
import numpy as np
from kneed import KneeLocator

import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

def task_3_dbscan_analysis(input_filepath):
    """
    Performs DBSCAN Clustering (Task 3.2).
    - Helps find optimal parameters (eps, min_samples)
    - Runs DBSCAN and analyzes the resulting clusters.
    """
    print(f"--- Task 3: DBSCAN Clustering ---")

    # --- 1. Configuration ---
    
    # Use the same 3-feature set that gave us our best K-Means score
    attributes_to_use=[
        'GameWeight',
        'MfgPlaytime',
        'NumOwned', 
    ]
    
    # --- Parameter selection ---
    # For min_samples, a common rule of thumb is (2 * num_features)
    # 2 * 3 features = 6
    CHOSEN_MIN_SAMPLES = 6
    
    # --- !!! IMPORTANT !!! ---
    # Me must find the best 'eps' value.
    # Run this script once. Open 'clustering_plots/k_distance_plot.png'.
    # Find the 'elbow' (point of sharpest curve) on the y-axis.
    # Set CHOSEN_EPS to that value (e.g., 0.2) and run again.
    # run with 0.3 and gag cluster 0 will ahve 94% of data

    CHOSEN_EPS = 0.1456
    
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

    # --- 3. Find Optimal 'eps' (k-distance plot) ---
    print(f"\n--- 3. Finding Optimal 'eps' ---")
    print(f"Calculating k-distance plot for min_samples = {CHOSEN_MIN_SAMPLES}...")
    
    neighbors = NearestNeighbors(n_neighbors=CHOSEN_MIN_SAMPLES)
    neighbors_fit = neighbors.fit(features_df)
    distances, indices = neighbors_fit.kneighbors(features_df)
    
    kth_distances = sorted(distances[:, CHOSEN_MIN_SAMPLES-1], reverse=True)
    
    


    # We need an x-axis for the KneeLocator, which is just the point index
    x_axis = range(1, len(kth_distances) + 1)

    # Reverse the list so the curve is "concave" and "decreasing"
    y_axis = sorted(kth_distances, reverse=False) 

    try:
        kn = KneeLocator(
            x_axis, 
            y_axis, 
            curve='convex',     # Shape of the curve
            direction='increasing', # How the y-axis (distance) is moving
            interp_method='polynomial' # Smoother method for finding the knee
        )

        automated_eps = kn.elbow_y # This is the y-value (distance) at the elbow

        if automated_eps:
            print(f"\n--- Automated 'eps' Finder ---")
            print(f"Optimal 'eps' (elbow) found at: {automated_eps:.4f}")
            print(f"==> ACTION: Set CHOSEN_EPS = {automated_eps:.4f} and re-run.")
            # I can even set it automatically if I want:
            # CHOSEN_EPS = automated_eps 
        else:
            print("\n--- Automated 'eps' Finder ---")
            print("Could not automatically find an elbow. Please check the plot manually.")

    except Exception as e:
        print(f"Error during automated elbow finding: {e}")
    
   
    plt.figure(figsize=(20, 12))
    plt.plot(range(1, len(kth_distances) + 1), kth_distances)
    plt.title(f'k-Distance Plot (k = {CHOSEN_MIN_SAMPLES})')
    plt.xlabel('Points (sorted by distance)')
    plt.ylabel(f'Distance to {CHOSEN_MIN_SAMPLES}-th Neighbor (eps)')
    plt.grid(True)
    
    plot_filename = os.path.join(plot_directory, "k_distance_plot.png")
    plt.savefig(plot_filename)
    plt.close()
    
    print(f"Plot saved to: {plot_filename}")
    print(f"==> ACTION: Open this plot, find the 'elbow', and update 'CHOSEN_EPS' in this script if needed.")

    # --- 4. Run Final DBSCAN Model ---
    print(f"\n--- 4. Running Final DBSCAN Model ---")
    print(f"Using parameters: eps={CHOSEN_EPS}, min_samples={CHOSEN_MIN_SAMPLES}")
    
    dbscan_model = DBSCAN(eps=CHOSEN_EPS, min_samples=CHOSEN_MIN_SAMPLES)
    clusters = dbscan_model.fit_predict(features_df)
    
    df['Cluster'] = clusters
    
    # --- 5. Analyze Clusters ---
    print(f"\n--- 5. Cluster Analysis ---")
    
    unique_labels = set(clusters)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(clusters).count(-1)
    
    print(f"Total clusters found: {n_clusters}")
    print(f"Total noise points: {n_noise} (%.2f%% of data)" % (100 * n_noise / len(df)))
    
    if n_clusters > 1:
        non_noise_mask = df['Cluster'] != -1
        silhouette = silhouette_score(features_df[non_noise_mask], df[non_noise_mask]['Cluster'])
        print(f"Silhouette Score (excl. noise): {silhouette:.4f}")
    elif n_clusters == 1:
        print("Silhouette Score: Not applicable (only one cluster found).")
    else:
        print("Silhouette Score: Not applicable (no clusters found).")

    if n_clusters > 0:
        print("\n--- Cluster Centroids (Mean Values) ---")
        cluster_analysis = df[df['Cluster'] != -1].groupby('Cluster')[available_cols].mean()
        cluster_analysis['Cluster_Size'] = df[df['Cluster'] != -1]['Cluster'].value_counts()
        print(cluster_analysis.T.to_string())
    else:
        print("\nNo clusters were found to analyze (all points are noise).")
        print("==> Try increasing 'CHOSEN_EPS' or decreasing 'CHOSEN_MIN_SAMPLES'.")

    print("\n--- DBSCAN Analysis Complete ---")

if __name__ == "__main__":
    input_dataset = "dm1_prepared_dataset.csv"
    task_3_dbscan_analysis(input_dataset)