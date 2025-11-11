import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates # <-- IMPORT THIS
from mpl_toolkits.mplot3d import Axes3D # <-- ADD THIS IMPORT
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def task_3_kmeans_selected_features(input_filepath):
    """
    Perform KMeans Clustering on a *manually selected*
    subset of features to see if it's better.
    """
    print(f"--- Task 3 K-Means (Selected Features) ---")
    
    #1. Config
    #Starting list of manually selected features
    attributes_to_use=[
        'GameWeight',
        'MfgPlaytime',
        'NumOwned', 
    ]
    
    max_k_test=20
    random_state=42
    plot_directory="clustering_plots"
    os.makedirs(plot_directory, exist_ok=True)
    
    #2. Load data
    print(f"\nLoading dataset from: {input_filepath}...")
    try:
        df=pd.read_csv(input_filepath)
        print("Dataset loaded successfully")
    except FileNotFoundError:
        print(f"--- Error: file not found at '{input_filepath}' ---")
        return
    
    # Filter the DataFrame to only our selected features
    available_cols = [col for col in attributes_to_use if col in df.columns]
    if len(available_cols) != len(attributes_to_use):
        print(f"Warning: Not all columns were available.")
        
    features_df = df[available_cols]
    print(f"This clustering will use {len(available_cols)} features: {available_cols}")
    
    #3. finding best k (elbow+silhoutte)
    print(f"\nFinding best 'k' by testing k=2 to k={max_k_test}...")
    kmeans_results={
        "k":[],
        "inertia":[],
        "silhouette":[]
    }
    
    for k in range(2, max_k_test +1):
        kmeans_model=KMeans(n_clusters=k,
                            n_init='auto',
                            max_iter=400,
                            random_state=random_state)
        kmeans_model.fit(features_df)
        
        #store results
        kmeans_results["k"].append(k)
        kmeans_results["inertia"].append(kmeans_model.inertia_)
        kmeans_results["silhouette"].append(silhouette_score(features_df, kmeans_model.labels_))
    
    #print table result
    results_df=pd.DataFrame(kmeans_results)
    print("\n--- K-Means Evaluation Metrics (Selected Features) ---")
    print(results_df.to_string(index=False))
    
    #4. plot metrics
    fig, (ax1, ax2)=plt.subplots(1,2, figsize=(15,6))
    
    #plot1 Elbow method (inertia)
    ax1.plot(kmeans_results["k"], kmeans_results["inertia"], 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (sum of squared distances)')
    ax1.set_title('Elbow method for optimal k (Selected Features)')
    
    #plot2 silhouette score
    ax2.plot(kmeans_results["k"], kmeans_results["silhouette"], 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette score')
    ax2.set_title('Silhouette score for optimal k (Selected Features)')
    
    plt.suptitle("K-Means Evaluation (Selected Features)", fontsize=16)
    plot_filename=os.path.join(plot_directory, "kmeans_evaluation_selected_features.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"\nEvaluation plot saved to: {plot_filename}")
    
    #5. Run final model & analyze clusters
    #find best 'k'
    best_k=int(results_df.loc[results_df['silhouette'].idxmax()]['k'])
    print(f"\n--- Running final model (Selected Features) ---")
    print(f"Best 'k' based on highest silhouette score: {best_k}")
    
    final_kmeans=KMeans(n_clusters=best_k,
                        n_init=10,
                        random_state=random_state)
    final_kmeans.fit(features_df)
    
    # --- IMPORTANT: Add cluster labels back to the original 'df' ---
    # We use 'df' for plotting because 'features_df' was just for fitting
    df['Cluster']=final_kmeans.labels_
    
    print(f"\n--- Cluster analysis (k={best_k}) (Selected Features) ---")
    cluster_analysis = pd.DataFrame(final_kmeans.cluster_centers_, columns=available_cols)
    cluster_analysis['Cluster_Size'] = pd.Series(final_kmeans.labels_).value_counts()
    
    print(cluster_analysis.T.to_string())
    
    # --- 6. Generate Parallel Coordinate Plot ---
    print(f"\n--- 6. Generating Centroid Plot ---")
    try:
        # We need to add a 'Cluster' column (as a string) for the plot to use as a label
        plot_data = cluster_analysis.drop(columns=['Cluster_Size'])
        plot_data['Cluster'] = [f'Cluster {i}' for i in plot_data.index]
        
        plt.figure(figsize=(15, 8))
        parallel_coordinates(plot_data, 'Cluster', colormap='viridis')
        
        plt.title(f'Parallel Coordinate Plot of {best_k} Cluster Centroids')
        plt.xlabel('Features')
        plt.ylabel('Scaled Value (from RobustScaler)')
        plt.legend(loc='center right', bbox_to_anchor=(1.15, 0.5))
        plt.tight_layout()
        
        plot_filename = os.path.join(plot_directory, "kmeans_centroid_plot.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Centroid plot saved to: {plot_filename}")
        
    except Exception as e:
        print(f"Error generating parallel coordinate plot: {e}")


    # --- 7. Generate 3D Scatter Plot ---
    print(f"\n--- 7. Generating 3D Scatter Plot (on 10% sample) ---")
    try:
        # Check if we actually have 3 features
        if len(available_cols) != 3:
            print("Skipping 3D plot: Requires exactly 3 features.")
        else:
            # We sample for performance and clarity (10% of data)
            sample_df = df.sample(frac=0.1, random_state=random_state)
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get feature names from the list
            f1 = available_cols[0] # GameWeight
            f2 = available_cols[1] # MfgPlaytime
            f3 = available_cols[2] # NumOwned
            
            # Plot the sampled data points, colored by their assigned cluster
            scatter = ax.scatter(
                sample_df[f1], 
                sample_df[f2], 
                sample_df[f3], 
                c=sample_df['Cluster'], 
                cmap='viridis',  # Use the same colormap as parallel plot
                alpha=0.5       # Add transparency
            )
            
            ax.set_title(f'3D Cluster Plot (k={best_k}, 10% Sample)')
            ax.set_xlabel(f1)
            ax.set_ylabel(f2)
            ax.set_zlabel(f3)
            
            # Create a legend
            legend1 = ax.legend(*scatter.legend_elements(),
                                title="Clusters")
            ax.add_artist(legend1)
            
            plot_filename = os.path.join(plot_directory, "kmeans_3d_scatter_plot.png")
            plt.savefig(plot_filename)
            plt.close()
            print(f"3D Scatter plot saved to: {plot_filename}")

    except Exception as e:
        print(f"Error generating 3D scatter plot: {e}")
    

    
    print("\n--- K-Means Analysis (Selected Features) Complete ---")
    
if __name__=="__main__":
    input_dataset="dm1_prepared_dataset.csv"
    task_3_kmeans_selected_features(input_dataset)