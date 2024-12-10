import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine, make_moons
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
from matplotlib import cm

# Load Wine dataset and remove labels
wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(wine_df)

# Define a consistent colormap for clusters
cluster_cmap = cm.get_cmap("Set2")

# Function to visualize clustering results with centroids for K-means using PCA (for 2D projection)
def plot_clusters(X, labels, centroids=None, title=""):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    unique_labels = np.unique(labels)
    
    # Use consistent colors for each cluster
    colors = cluster_cmap(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        plt.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], 
                    s=50, c=[colors[i]], label=f'Cluster {label+1}', edgecolor='k')
    
    if centroids is not None:
        centroids_pca = pca.transform(centroids)
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=250, marker='*', c='red', edgecolor='black', label='Centroids')
        plt.legend()
    
    plt.title(title)
    plt.grid(True)
    plt.show()

# 1. K-means clustering with 3 clusters and k-means++ initialization
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=0)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Visualize K-means clusters with centroids
plot_clusters(X_scaled, kmeans_labels, centroids=kmeans.cluster_centers_, title='K-means Clustering with Consistent Colors')

# Elbow method to find the optimal number of clusters
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=0)
    km.fit(X_scaled)
    distortions.append(km.inertia_)

plt.plot(range(1, 11), distortions, marker='o', color='b')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.grid(True)
plt.show()

# 2. Hierarchical clustering and dendrogram
linked = linkage(X_scaled, method='complete')
plt.figure(figsize=(10, 7))
dendrogram(linked, color_threshold=0.7 * max(linked[:, 2]))
plt.title('Dendrogram for Hierarchical Clustering')
plt.grid(True)
plt.show()

# Hierarchical clustering heatmap with consistent colors
sns.clustermap(X_scaled, method='complete', cmap='coolwarm', standard_scale=1)
plt.title('Dendrogram Heatmap for Hierarchical Clustering')
plt.show()

# 3. DBSCAN clustering using make_moons
X_moons, _ = make_moons(n_samples=200, noise=0.05, random_state=0)
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_moons)

# Visualize DBSCAN on moons dataset with consistent colors
unique_dbscan_labels = np.unique(dbscan_labels)
colors = cluster_cmap(np.linspace(0, 1, len(unique_dbscan_labels)))

for i, label in enumerate(unique_dbscan_labels):
    plt.scatter(X_moons[dbscan_labels == label, 0], X_moons[dbscan_labels == label, 1], 
                c=[colors[i]], s=50, label=f'Cluster {label}', edgecolor='k')

plt.title('DBSCAN Clustering on Moons Dataset with Consistent Colors')
plt.grid(True)
plt.show()

# 4. Silhouette analysis for K-means with different number of clusters (including axvline)
def silhouette_analysis(X, range_n_clusters):
    for n_clusters in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # Initialize the KMeans
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(X)

        # Silhouette score for the average value
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(f"For n_clusters = {n_clusters}, the average silhouette score is {silhouette_avg}")

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Use consistent colors for the silhouette plot
        colors = cluster_cmap(np.linspace(0, 1, n_clusters))

        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=colors[i], edgecolor=colors[i], alpha=0.7)

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        ax1.set_title(f"Silhouette plot for {n_clusters} clusters")
        ax1.set_xlabel("Silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # Average silhouette line (axvline)
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        # Plot 2D clustering results using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='nipy_spectral', marker='o', edgecolor='k', s=50)
        ax2.set_title(f"Cluster plot for {n_clusters} clusters")

        plt.show()

# Perform silhouette analysis for K-means with 3, 4, and 5 clusters
silhouette_analysis(X_scaled, [3, 4, 5])
