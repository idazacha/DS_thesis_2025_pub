import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def hopkins(X, sample_size=None, random_state=None):
    """
    Compute the Hopkins statistic for assessing cluster tendency.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input dataset.
    sample_size : int or None
        Number of points to sample. 
        If None, defaults to sqrt(n_samples).
    random_state : int or None
        For reproducibility.

    Returns
    -------
    H : float
        Hopkins statistic in [0, 1]. 
        ~0.5 → random (no cluster structure)
        ~1.0 → highly clusterable
        ~0.0 → uniform/regular distribution
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    n, d = X.shape

    if sample_size is None:
        sample_size = int(np.sqrt(n))  # recommended in literature

    rng = np.random.default_rng(random_state)

    # Fit nearest neighbor model to the data
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)

    # Randomly sample points from X
    rand_indices = rng.choice(n, sample_size, replace=False)
    X_sample = X[rand_indices]

    # Random points generated uniformly within feature ranges
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X_uniform = rng.uniform(X_min, X_max, size=(sample_size, d))

    # For real points: distance to nearest neighbor (excluding itself)
    u_dist, _ = nbrs.kneighbors(X_uniform, n_neighbors=1)
    w_dist, idx = nbrs.kneighbors(X_sample, n_neighbors=2)
    w_dist = w_dist[:, 1]  # skip distance to itself

    U = u_dist.sum()
    W = w_dist.sum()

    return U / (U + W + 1e-12)

def hopkins_repeated(X, runs=20):
    H_values = [hopkins(X, random_state=i) for i in range(runs)]
    return np.mean(H_values), np.std(H_values), H_values
