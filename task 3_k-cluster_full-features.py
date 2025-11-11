import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def task_3_kmeans_analysis(input_filepath):
    """
    Perform KMeans Clustering
    FInd the best 'k' and analyzes the clusters
    
    """
    print(f"--- Task 3 KMeans Clustering ---")
    
    #1. Config
    max_k_test=20
    random_state=42
    plot_directory="clustering_plots" 
    os.makedirs(plot_directory, exist_ok=True)
    print(f"Saving plots to: {plot_directory}")

    
    #2. Load data
    print(f"\nLoading dataset from: {input_filepath}...")
    try:
        df=pd.read_csv(input_filepath)
        print("Dataset loaded successfully")
    except FileNotFoundError:
        print(f"--- Error: file not found at '{input_filepath}' ---")
        print("Remember to run 'task 2_analysis.py' first")
        return
    
    features_df = df.copy()
    attributes_to_use = features_df.columns.tolist() 
    print(f"This clustering will use {len(attributes_to_use)} features.")
    
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
    print("\n--- K-Means Evaluation Metrics ---")
    print(results_df.to_string(index=False))
    
    #4. plot metrics
    fig, (ax1, ax2)=plt.subplots(1,2, figsize=(15,6))
    
    #plot1 Elbow method (inertia)
    ax1.plot(kmeans_results["k"], kmeans_results["inertia"], 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (sum of squared distances)')
    ax1.set_title('Elbow method for optimal k')
    
    #plot2 silhouette score
    ax2.plot(kmeans_results["k"], kmeans_results["silhouette"], 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette score')
    ax2.set_title('Silhouette score for optimal k')
    
    plt.suptitle("K-Means evaluation", fontsize=16)
    plot_filename=os.path.join(plot_directory, "kmeans_evaluation.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"\nEvaluation plot saved to: {plot_filename}")
    
    #5. Run final model & analyze clusters
    #find best 'k'
    best_k=int(results_df.loc[results_df['silhouette'].idxmax()]['k'])
    print(f"\n--- Running final model ---")
    print(f"Best 'k' based on highest silhouette score: {best_k}")
    
    final_kmeans=KMeans(n_clusters=best_k,
                        n_init=10,
                        random_state=random_state)
    final_kmeans.fit(features_df)
    
    df['Cluster']=final_kmeans.labels_
    
    print(f"\n--- Cluster analysis (k={best_k}) ---")
    # Get the cluster centroids
    cluster_analysis = pd.DataFrame(final_kmeans.cluster_centers_, columns=attributes_to_use)
    
    # Add the size of each cluster
    cluster_analysis['Cluster_Size'] = pd.Series(final_kmeans.labels_).value_counts()
    
    # --- Transpose (.T) the DataFrame for clean printing ---
    print(cluster_analysis.T.to_string())
    
    print("\n--- K-Means Analysis Complete ---")
    
if __name__=="__main__":
    input_dataset="dm1_prepared_dataset.csv"
    task_3_kmeans_analysis(input_dataset)