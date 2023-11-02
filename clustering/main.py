import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Data Preprocessing
data = pd.read_csv("country_data.csv")

# Handle missing values (if any)
data.fillna(0, inplace=True)

# Step 2: Feature Selection
# Include relevant features. You can customize this list.
selected_features = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']


# Select the relevant columns
X = data[selected_features]

# Step 3: Clustering Algorithm Selection
# Choose the number of clusters (K)
K = 4  # You can experiment with different values of K

# Step 4: Clustering
# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=K, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 6: Visualizing Clusters using PCA
# Reduce dimensionality for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot clusters
plt.figure(figsize=(8, 6))
for cluster in range(K):
    plt.scatter(X_pca[data['Cluster'] == cluster, 0], X_pca[data['Cluster'] == cluster, 1], label=f'Cluster {cluster}')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.title('Cluster Visualization')
plt.show()

# Step 7: Analyzing and Interpreting the Clusters
# You can analyze the clusters by examining the characteristics of each cluster.

for cluster in range(K):
    cluster_data = data[data['Cluster'] == cluster]
    print(f'Cluster {cluster} - Number of Countries: {len(cluster_data)}')
    print(cluster_data[selected_features].mean())

# Step 9: Conclusions and Recommendations
# Based on the cluster analysis, draw conclusions and make recommendations for each cluster.
