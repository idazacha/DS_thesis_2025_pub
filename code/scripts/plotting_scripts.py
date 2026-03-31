import pandas as pd
from scipy.cluster.hierarchy import leaves_list, fcluster, dendrogram
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_hex
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_heatmap(Z, C, 
                 station_names, 
                 k_clusters, 
                 do_save=False,
                 save_prefix=""):
    # Derive leaf order
    order = leaves_list(Z)
    C_ord = C[order][:, order]
    names_ord = [station_names[i] for i in order]

    fig, ax_heat = plt.subplots(figsize=(14, 14))
    clusters = fcluster(Z, k_clusters, criterion='maxclust')
    cluster_colors = sns.color_palette("tab10", k_clusters)
    clusters_ord = [clusters[i] for i in order]

    # --- BUILD NODE → CLUSTER MAPPING ---
    n_leaves = len(clusters)
    node_cluster = {}

    # Leaves: node_id is the index
    for i in range(n_leaves):
        node_cluster[i] = clusters[i]

    # Internal nodes (SciPy IDs: n, n+1, ..., n+Z.shape[0]-1)
    for i in range(Z.shape[0]):
        node_id = n_leaves + i
        left = int(Z[i, 0])
        right = int(Z[i, 1])

        left_cluster = node_cluster[left]
        right_cluster = node_cluster[right]

        # If children belong to same cluster → internal node is that cluster
        if left_cluster == right_cluster:
            node_cluster[node_id] = left_cluster
        else:
            # Mixed cluster: assign None or a fallback color
            node_cluster[node_id] = None

    # --- CUSTOM COLOR FUNCTION ---
    def color_func(node_id):
        cl = node_cluster[node_id]
        if cl is None:
            return "#000000"   # black for mixed branches
        return to_hex(cluster_colors[cl - 1])

    # --- BOTTOM: HEATMAP ---
    df = pd.DataFrame(C_ord, index=names_ord, columns=names_ord)

    # Create a divider to manually place the colorbar
    divider = make_axes_locatable(ax_heat)
    cax = divider.append_axes("left", size="5%", pad=0.5)

    sns.heatmap(
        df,
        cmap="coolwarm",
        vmin=0, vmax=1,
        ax=ax_heat,
        cbar_ax=cax,                    # <-- colorbar on LEFT
        #cbar_kws={'label': 'Co-association'}
    )

    # ----- Move y-tick labels to the RIGHT -----
    ax_heat.yaxis.tick_right()
    ax_heat.yaxis.set_label_position("right")

    # Rotate tick labels
    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=90)
    ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0)

    # Optional: remove y-axis ticks on left
    ax_heat.tick_params(left=False)

    for tick_label, cl in zip(ax_heat.get_xticklabels(), clusters_ord):
        tick_label.set_color(to_hex(cluster_colors[cl-1]))

    for tick_label, cl in zip(ax_heat.get_yticklabels(), clusters_ord):
        tick_label.set_color(to_hex(cluster_colors[cl-1]))
    
    # find boundaries where cluster label changes
    cluser_boundaries = np.where(np.diff(clusters_ord) != 0)[0] + 1

    for b in cluser_boundaries:
        ax_heat.hlines(b, 0, len(C_ord), colors='white', linewidth=2, alpha=0.5)
        ax_heat.vlines(b, 0, len(C_ord), colors='white', linewidth=2, alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(left=0.12, right=0.98)

    if do_save:
        plt.savefig(f"../results/plots/clustering/coass_{k_clusters}{save_prefix}.png")

    plt.show()

def plot_heatmap_no_k(Z, C, station_names, do_save=[False, ""]):
    # Derive leaf order
    order = leaves_list(Z)
    C_ord = C[order][:, order]
    names_ord = [station_names[i] for i in order]

    fig, ax_heat = plt.subplots(figsize=(14, 14))

    # --- BOTTOM: HEATMAP ---
    df = pd.DataFrame(C_ord, index=names_ord, columns=names_ord)

    # Create a divider to manually place the colorbar
    divider = make_axes_locatable(ax_heat)
    cax = divider.append_axes("left", size="5%", pad=0.5)

    sns.heatmap(
        df,
        cmap="coolwarm",
        vmin=0, vmax=1,
        ax=ax_heat,
        cbar_ax=cax,                    # <-- colorbar on LEFT
        cbar_kws={'label': 'Co-association'}
    )

    # ----- Move y-tick labels to the RIGHT -----
    ax_heat.yaxis.tick_right()
    ax_heat.yaxis.set_label_position("right")

    # Rotate tick labels
    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=90)
    ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0)

    # Optional: remove y-axis ticks on left
    ax_heat.tick_params(left=False)

    #plt.tight_layout()
    plt.subplots_adjust(left=0.12, right=0.98)

    if do_save[0] and do_save[1] != None:
        plt.savefig(f"../results/plots/clustering/coass_{do_save[1]}.png")

    plt.show()


def plot_dendrogram_top_and_heatmap(Z, 
                                    C, 
                                    station_names, 
                                    k_clusters, 
                                    do_save=[False, None]):

    # Derive leaf order
    order = leaves_list(Z)
    C_ord = C[order][:, order]
    names_ord = [station_names[i] for i in order]

    # Create figure with two rows
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 6], hspace=0.05)

    clusters = fcluster(Z, k_clusters, criterion='maxclust')
    cluster_colors = sns.color_palette("tab10", k_clusters)
    clusters_ord = [clusters[i] for i in order]

    # --- BUILD NODE → CLUSTER MAPPING ---
    n_leaves = len(clusters)
    node_cluster = {}

    # Leaves: node_id is the index
    for i in range(n_leaves):
        node_cluster[i] = clusters[i]

    # Internal nodes (SciPy IDs: n, n+1, ..., n+Z.shape[0]-1)
    for i in range(Z.shape[0]):
        node_id = n_leaves + i
        left = int(Z[i, 0])
        right = int(Z[i, 1])

        left_cluster = node_cluster[left]
        right_cluster = node_cluster[right]

        # If children belong to same cluster → internal node is that cluster
        if left_cluster == right_cluster:
            node_cluster[node_id] = left_cluster
        else:
            # Mixed cluster: assign None or a fallback color
            node_cluster[node_id] = None

    # --- CUSTOM COLOR FUNCTION ---
    def color_func(node_id):
        cl = node_cluster[node_id]
        if cl is None:
            return "#000000"   # black for mixed branches
        return to_hex(cluster_colors[cl - 1])
    
    # --- TOP: Dendrogram ---
    ax_dendro = plt.subplot(gs[0])
    dendrogram(
        Z,
        orientation="top",
        labels=names_ord,
        leaf_font_size=8,
        color_threshold=0,
        link_color_func=color_func,
        ax=ax_dendro
    )
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])

    # Remove the black border (all spines)
    for spine in ax_dendro.spines.values():
        spine.set_visible(False)

    # Shrink dendrogram width: make it 80% as wide as the heatmap
    pos = ax_dendro.get_position()
    ax_dendro.set_position([
        pos.x0,   # shift right
        pos.y0,          
        pos.width * 0.8, # shrink width
        pos.height
    ])

    # --- BOTTOM: Heatmap ---
    ax_heat = plt.subplot(gs[1])
    #ax_heat.set_aspect("equal")
    df = pd.DataFrame(C_ord, index=names_ord, columns=names_ord)

    hm = sns.heatmap(
        df,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        ax=ax_heat,
        cbar_kws={'label': 'Co-association'}
    )

    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=90)
    ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0)

    for tick_label, cl in zip(ax_heat.get_xticklabels(), clusters_ord):
        tick_label.set_color(to_hex(cluster_colors[cl-1]))

    for tick_label, cl in zip(ax_heat.get_yticklabels(), clusters_ord):
        tick_label.set_color(to_hex(cluster_colors[cl-1]))
    
    # find boundaries where cluster label changes
    cluser_boundaries = np.where(np.diff(clusters_ord) != 0)[0] + 1

    for b in cluser_boundaries:
        ax_heat.hlines(b, 0, len(C_ord), colors='white', linewidth=2, alpha=0.5)
        ax_heat.vlines(b, 0, len(C_ord), colors='white', linewidth=2, alpha=0.5)

    plt.tight_layout(pad=0.2)

    if do_save[0] and do_save[1] != None:
        plt.savefig(f"../results/plots/clustering/dendrogram_coass_{k_clusters}_{do_save[1]}.png")

    plt.show()


def plot_dendrogram_top_and_heatmap2(
        Z, 
        C, 
        station_names, 
        consensus_labels, 
        do_save=[False, None]
        ):

    # Derive leaf order

    order = leaves_list(Z)
    C_ord = C[order][:, order]
    names_ord = [station_names[i] for i in order]

    clusters = np.array(consensus_labels)
    cluster_colors = sns.color_palette("tab10", len(np.unique(clusters)))
    clusters_ord = clusters[order]
    n_clusters = len(np.unique(clusters))

    # Create figure with two rows
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 6], hspace=0.05)

    # --- BUILD NODE → CLUSTER MAPPING ---
    n_leaves = len(clusters)
    node_cluster = {}

    # Leaves: node_id is the index
    for i in range(n_leaves):
        node_cluster[i] = clusters[i]

    # Internal nodes (SciPy IDs: n, n+1, ..., n+Z.shape[0]-1)
    for i in range(Z.shape[0]):
        node_id = n_leaves + i
        left = int(Z[i, 0])
        right = int(Z[i, 1])

        left_cluster = node_cluster[left]
        right_cluster = node_cluster[right]

        # If children belong to same cluster → internal node is that cluster
        if left_cluster == right_cluster:
            node_cluster[node_id] = left_cluster
        else:
            # Mixed cluster: assign None or a fallback color
            node_cluster[node_id] = None

    # --- CUSTOM COLOR FUNCTION ---
    def color_func(node_id):
        cl = node_cluster[node_id]
        if cl is None:
            return "#000000"   # black for mixed branches
        return to_hex(cluster_colors[cl - 1])
    
    # --- TOP: Dendrogram ---
    ax_dendro = plt.subplot(gs[0])
    dendrogram(
        Z,
        orientation="top",
        labels=names_ord,
        leaf_font_size=8,
        color_threshold=0,
        link_color_func=color_func,
        ax=ax_dendro
    )
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])

    # Remove the black border (all spines)
    for spine in ax_dendro.spines.values():
        spine.set_visible(False)

    # Shrink dendrogram width: make it 80% as wide as the heatmap
    pos = ax_dendro.get_position()
    ax_dendro.set_position([
        pos.x0,   # shift right
        pos.y0,          
        pos.width * 0.8, # shrink width
        pos.height
    ])

    # --- BOTTOM: Heatmap ---
    ax_heat = plt.subplot(gs[1])
    #ax_heat.set_aspect("equal")
    df = pd.DataFrame(C_ord, index=names_ord, columns=names_ord)

    hm = sns.heatmap(
        df,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        ax=ax_heat,
        cbar_kws={'label': 'Co-association'}
    )

    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=90)
    ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0)

    for tick_label, cl in zip(ax_heat.get_xticklabels(), clusters_ord):
        tick_label.set_color(to_hex(cluster_colors[cl-1]))

    for tick_label, cl in zip(ax_heat.get_yticklabels(), clusters_ord):
        tick_label.set_color(to_hex(cluster_colors[cl-1]))
    
    # find boundaries where cluster label changes
    cluser_boundaries = np.where(np.diff(clusters_ord) != 0)[0] + 1

    for b in cluser_boundaries:
        ax_heat.hlines(b, 0, len(C_ord), colors='white', linewidth=2, alpha=0.5)
        ax_heat.vlines(b, 0, len(C_ord), colors='white', linewidth=2, alpha=0.5)

    #plt.tight_layout()
    #plt.tight_layout(rect=[0, 0, 0, 0])  # leave space at bottom for legend
    plt.tight_layout(pad=0.2)

    if do_save[0] and do_save[1] != None:
        plt.savefig(f"../results/plots/clustering/dendrogram_coass_{n_clusters}_{do_save[1]}.png")

    plt.show()


def plot_evaluation_metrics(metrics_df, 
                            metrics_col,
                            save_prefix="",
                            do_save=False):
    
    n_metrics = len(metrics_col)
    n_rows = int(np.ceil(n_metrics / 2))
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_col):
        ax = axes[idx]
        
        mean_col = "avg_" + metric
        std_col  = "std_" + metric
        
        k = metrics_df["k"]
        mean_vals = metrics_df[mean_col]
        
        ax.plot(k, mean_vals, marker="o", label=f"Mean {metric}")
        
        # If std metric exists, plot uncertainty band + error bars
        if std_col in metrics_df.columns:
            std_vals = metrics_df[std_col]
            
            ax.fill_between(
                k,
                mean_vals - std_vals,
                mean_vals + std_vals,
                alpha=0.2,
                label="±1 SD"
            )
            
            ax.errorbar(
                k, mean_vals, yerr=std_vals, 
                fmt='o-', capsize=4
            )
        
        ax.set_title(metric)
        ax.set_xlabel("k")
        ax.set_ylabel("Value")
        ax.grid(axis="y")
        ax.legend()
    
    # Hide unused subplots
    for j in range(idx+1, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()

    if do_save:
        plt.savefig(f"../results/clustering/evaluation_metrics{save_prefix}.png",
                    dpi=300)
    
    plt.show()


def plot_evaluation_metrics_2(metrics_df, 
                              metrics_col,
                              do_save=False,
                              save_prefix=""):
    
    n_metrics = len(metrics_col)
    n_rows = int(np.ceil(n_metrics / 2))
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_col):
        ax = axes[idx]
        
        k = metrics_df["k"]

        ax.plot(k, 
                metrics_df[metric], 
                marker="o", 
                label=metric)
        
        ax.set_title(metric)
        ax.set_xlabel("k")
        ax.set_ylabel("Value")
        ax.grid(axis="y")
        ax.legend()
    
    # Hide unused subplots
    for j in range(idx+1, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()

    if do_save[0] and do_save[1] is not None:
        plt.savefig(f"../results/clustering/evaluation_metrics{save_prefix}.png",
                    dpi=300)
    
    plt.show()


def plot_3d_clusters(X, consensus_df, chosen_k):
    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        c=consensus_df[f'k_{chosen_k}'],
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
        consensus_df[f'k_{chosen_k}'].unique().tolist(),
        loc="upper right",
        title="Classes",
    )
    ax.add_artist(legend1)

    plt.show()