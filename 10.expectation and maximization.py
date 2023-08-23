import numpy as np
from scipy.stats import multivariate_normal

# Generate synthetic data
np.random.seed(42)
num_samples = 300
mean1 = [2, 3]
cov1 = [[1, 0.5], [0.5, 1]]
mean2 = [7, 8]
cov2 = [[1.2, -0.6], [-0.6, 1.2]]
data1 = np.random.multivariate_normal(mean1, cov1, num_samples // 2)
data2 = np.random.multivariate_normal(mean2, cov2, num_samples // 2)
data = np.vstack((data1, data2))

# Initialize GMM parameters
num_clusters = 2
num_features = data.shape[1]
pi = np.ones(num_clusters) / num_clusters
mu = np.random.rand(num_clusters, num_features)
sigma = [np.eye(num_features)] * num_clusters

# EM Algorithm
num_iterations = 50
for _ in range(num_iterations):
    # E-step: Compute responsibilities
    responsibilities = np.zeros((num_samples, num_clusters))
    for k in range(num_clusters):
        responsibilities[:, k] = pi[k] * multivariate_normal.pdf(data, mean=mu[k], cov=sigma[k])
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    
    # M-step: Update parameters
    N_k = responsibilities.sum(axis=0)
    pi = N_k / num_samples
    mu = (responsibilities.T @ data) / N_k[:, np.newaxis]
    for k in range(num_clusters):
        diff = data - mu[k]
        sigma[k] = (diff.T @ (diff * responsibilities[:, k][:, np.newaxis])) / N_k[k]

# Print the estimated parameters
print("Estimated Mixing Coefficients (pi):", pi)
print("Estimated Means (mu):", mu)
print("Estimated Covariances (sigma):", sigma)
