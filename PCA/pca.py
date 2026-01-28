# ---------------------------------------
# PCA on Student Performance Dataset
# Features: Study Hours, Attendance, Marks
# ---------------------------------------

import numpy as np
import pandas as pd

# -----------------------------
# Step 1: Create the dataset
# -----------------------------
data = {
    'Study_Hours': [2, 3, 1, 4, 2, 5, 4, 1, 6, 3],
    'Attendance': [5, 6, 4, 7, 6, 8, 7, 5, 9, 6],
    'Marks': [45, 50, 40, 55, 48, 60, 58, 42, 65, 52]
}

df = pd.DataFrame(data)
print("Original Dataset:\n")
print(df)

# -----------------------------
# Step 2: Compute mean of each feature
# -----------------------------
mean_vector = df.mean()
print("\nMean Vector:\n")
print(mean_vector)

# -----------------------------
# Step 3: Mean-center the data
# -----------------------------
X_centered = df - mean_vector
print("\nMean Centered Data:\n")
print(X_centered)

# -----------------------------
# Step 4: Compute covariance matrix
# -----------------------------
cov_matrix = np.cov(X_centered.T)
print("\nCovariance Matrix:\n")
print(cov_matrix)

# -----------------------------
# Step 5: Compute eigenvalues and eigenvectors
# -----------------------------
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\nEigenvalues:\n")
print(eigenvalues)

print("\nEigenvectors:\n")
print(eigenvectors)

# -----------------------------
# Step 6: Sort eigenvalues in descending order
# -----------------------------
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

print("\nSorted Eigenvalues:\n")
print(eigenvalues_sorted)

print("\nSorted Eigenvectors:\n")
print(eigenvectors_sorted)

# -----------------------------
# Step 7: Explained variance ratio
# -----------------------------
explained_variance_ratio = eigenvalues_sorted / np.sum(eigenvalues_sorted)

print("\nExplained Variance Ratio:\n")
print(explained_variance_ratio)

# -----------------------------
# Step 8: Select top k components
# (Here we choose k = 2)
# -----------------------------
k = 2
W = eigenvectors_sorted[:, :k]

print("\nProjection Matrix (Top 2 Eigenvectors):\n")
print(W)

# -----------------------------
# Step 9: Project data onto principal components
# -----------------------------
Z = np.dot(X_centered, W)

pca_df = pd.DataFrame(Z, columns=['PC1', 'PC2'])
print("\nData after PCA (Reduced to 2 Dimensions):\n")
print(pca_df)
