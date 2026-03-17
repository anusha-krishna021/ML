import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print("Running K-Means Clustering...")

# Load dataset
data = pd.read_csv("train.csv")

# Features only (remove target)
X = data.drop("price_range", axis=1)

# Reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42)

clusters = kmeans.fit_predict(X)

print("Clustering completed")

# Plot clusters
plt.figure(figsize=(8, 6))

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clustering of Mobile Prices")

plt.show()
