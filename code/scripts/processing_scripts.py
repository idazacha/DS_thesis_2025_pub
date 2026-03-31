from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def preprocess_data(df, station_col, use_pca=True, pca_components=0.9):
    stations = df[station_col].values
    X = df.drop(columns=[station_col]).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca_res = None

    if use_pca:
        pca = PCA(n_components=pca_components, random_state=0)
        Xs = pca.fit_transform(Xs)
        pca_res = {
            "explained_var_ratio": pca.explained_variance_ratio_,
            "explained_var_cumsum": pca.explained_variance_ratio_.cumsum(),
            "components": pca.components_,
            "mean": pca.mean_,
            "n_components": pca.n_components_
        }

    return stations, Xs, pca_res

def make_pca_loading_table(results, feature_names, n_components=5):
    # TO DO: Move to script
    """
    Create a PCA loading table similar to published PCA studies.
    
    results: your PCA results dict
    feature_names: list of original feature (column) names
    n_components: number of PCs to include in the table
    """
    loadings = results["components"][:n_components]  # get first PCs
    df = pd.DataFrame(
        loadings.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=feature_names
    )
    
    # Add absolute loadings to help sorting
    df["max_abs_loading"] = df.abs().max(axis=1)
    
    # Sort by strongest loading
    df = df.sort_values("max_abs_loading", ascending=False)
    
    return df


def pca_scatter_3d(X, consensus_labels, chosen_k):
    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        c=consensus_labels[f'k_{chosen_k}'],
        s=50,
    )

    ax.set(
        title="First three PCA dimensions",
        xlabel="1st Eigenvector",
        ylabel="2nd Eigenvector",
        zlabel="3rd Eigenvector",
    )
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    # Add a legend
    legend1 = ax.legend(
        scatter.legend_elements()[0],
        consensus_labels[f'k_{chosen_k}'].unique().tolist(),
        loc="upper right",
        title="Classes",
    )
    ax.add_artist(legend1)

    plt.show()