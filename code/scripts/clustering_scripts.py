import numpy as np
import pandas as pd
import itertools
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform
from scripts.clustering_utils import build_coassoc_matrix, station_stability_scores, build_configs
from sklearn.mixture import GaussianMixture

def run_base_clustering(X, 
                   name_order, 
                   K_VALUES=range(2,11), 
                   N_RUNS=50,
                   base_clustering='KMeans',
                   params = None,
                   do_save=False,
                   save_prefix=""
                   ):
    # STORING LABELS
    all_labels = {}
    labels_df = pd.DataFrame()
    labels_df['name'] = name_order

    # To store ensemble metrics
    avg_silhouette, std_silhouette, all_silhouette = {}, {}, {}

    avg_dbi, std_dbi, all_dbi = {}, {}, {}

    internal_met = pd.DataFrame(K_VALUES, columns = ['k'])

    # ============================================
    # RUN BASE ENSEMBLE FOR EACH k
    # ============================================
    print("Starting Clustering...")
    for k in K_VALUES:
        labels_k = []
        sil_k = []
        dbi_k = []

        model_dic = {}
        
        # Run N_RUNS of KMeans for this k
        for r in range(N_RUNS):

            if base_clustering == 'KMeans':
                km = KMeans(
                    n_clusters=k,
                    init='k-means++',
                    n_init=1,                 # IMPORTANT: one init per run; randomness comes from seed
                    random_state=r            # different seed → different solution
                )
                
                z = km.fit_predict(X)
                labels_k.append(z)

                model_dic['cluster_centers'] = km.cluster_centers_
                model_dic['inertia'] = km.inertia_
            
            elif base_clustering == 'Spectral':
                sc = SpectralClustering(
                    n_clusters=k,
                    assign_labels='discretize', # try 'kmeans'
                    n_init=1,
                    random_state=r
                )
            
                z = sc.fit_predict(X)
                labels_k.append(z)
                model_dic['affinity_matrix'] = sc.affinity_matrix_
            
            elif base_clustering=="GMM":
                gm = GaussianMixture(
                    n_components=k, 
                    random_state=r
                )
                z = gm.fit_predict(X)
                labels_k.append(z)

            elif base_clustering=="Agglomerative":
                agg_m = AgglomerativeClustering(
                    n_clusters=k,
                    linkage=params["linkage"],
                    metric=params['metric']
                )
                z = agg_m.fit_predict(X)
                labels_k.append(z)

            else:
                raise ValueError("Unsupported base_clustering method. Choose 'KMeans' or 'Spectral'.")
                
            # Silhouette score for this run
            sil_k.append(silhouette_score(X, z))
            dbi_k.append(davies_bouldin_score(X, z))
        

        # Store results for this k
        all_labels[k] = labels_k

        avg_silhouette[k] = np.mean(sil_k)
        std_silhouette[k] = np.std(sil_k)
        all_silhouette[k] = sil_k

        avg_dbi[k] = np.mean(dbi_k)
        std_dbi[k] = np.std(dbi_k)
        all_dbi[k] = dbi_k

    print("Clustering done... Storing Internal metrics")

    internal_met['avg_silhouette'] = avg_silhouette.values()
    internal_met['std_silhouette'] = std_silhouette.values()
    internal_met['avg_dbi'] = avg_dbi.values()
    internal_met['std_dbi'] = std_dbi.values()

    if do_save:
        print(f"Saving all cluster results and internal metrics")
        internal_met.to_csv(f"../results/clustering/{save_prefix}metrics_internal.csv", index=False)

    print("\n")
    return all_labels, internal_met


def run_base_clustering_multi(
    X, 
    K_VALUES=range(2, 11), 
    N_RUNS=10,
    base_methods=('KMeans', 'GMM', 'Spectral', 'Agglomerative'),
    runs_per_method=None,
    params=None,
    do_save=False,
    save_prefix=""
    ):
    if params is None:
        params = {}

    if runs_per_method is None:
        runs_per_method = {
            "KMeans": N_RUNS,
            "GMM": N_RUNS,
            "Spectral": N_RUNS,
            "Agglomerative": 1
        }

    # Expand each method + param grid into a list of configs
    configs = build_configs(base_methods, params, runs_per_method, N_RUNS)
    print(f"Total base configurations: {len(configs)}")

    # STORING LABELS (for consensus later)
    all_labels = {}  # dict: k -> list of label arrays

    # store labels by method
    all_labels_by_method = {
        m: {k: [] for k in K_VALUES}
        for m in base_methods
    }

    # Internal metrics aggregated across ALL configs & runs
    avg_silhouette, std_silhouette, all_silhouette = {}, {}, {}
    avg_dbi, std_dbi, all_dbi = {}, {}, {}
    internal_met = pd.DataFrame(K_VALUES, columns=['k'])

    total_runs = 0

    print("Starting multi-method base clustering...")

    labels_info = []   # list of dicts containing {method, config, k, labels}

    for k in K_VALUES:
        labels_k = []
        sil_k = []
        dbi_k = []

        for cfg in configs:
            method = cfg["method"]
            cfg_name = cfg["name"]
            cfg_params = cfg["params"]
            n_runs = cfg["runs"]

            for r in range(n_runs):
                if method == 'KMeans':
                    km = KMeans(
                        n_clusters=k,
                        init='k-means++',
                        n_init=1,
                        random_state=r
                    )
                    z = km.fit_predict(X)

                elif method == 'Spectral':
                    sc = SpectralClustering(
                        n_clusters=k,
                        n_init=1,
                        random_state=r,
                        **cfg_params
                    )
                    z = sc.fit_predict(X)

                elif method == 'GMM':
                    gm = GaussianMixture(
                        n_components=k,
                        random_state=r,
                        **cfg_params
                    )
                    z = gm.fit_predict(X)

                elif method == 'Agglomerative':
                    agg_m = AgglomerativeClustering(
                        n_clusters=k,
                        **cfg_params
                    )
                    z = agg_m.fit_predict(X)

                else:
                    raise ValueError(f"Unsupported method: {method}")

                labels_k.append(z)
                sil = silhouette_score(X, z)
                dbi = davies_bouldin_score(X, z)
                sil_k.append(sil)
                dbi_k.append(dbi)
                
                # Store in global list for ensemble
                all_labels_by_method[cfg["method"]][k].append(z)
                #all_rows.append([method, cfg_name, k, r, sil_k, dbi_k])
                total_runs += 1

                labels_info.append({
                    "method": cfg["method"],
                    "config": cfg["name"],
                    "k": k,
                    "labels": z,
                    "sil": sil,
                    "dbi": dbi
                })


        all_labels[k] = labels_k
        avg_silhouette[k] = np.mean(sil_k)
        std_silhouette[k] = np.std(sil_k)
        all_silhouette[k] = sil_k

        avg_dbi[k] = np.mean(dbi_k)
        std_dbi[k] = np.std(dbi_k)
        all_dbi[k] = dbi_k

    print("Clustering done... Storing internal metrics")
    print("Total model fits:", total_runs)

    internal_met['avg_silhouette'] = list(avg_silhouette.values())
    internal_met['std_silhouette'] = list(std_silhouette.values())
    internal_met['avg_dbi'] = list(avg_dbi.values())
    internal_met['std_dbi'] = list(std_dbi.values())

    method_metric_df = {}
    for method in all_labels_by_method:
        rows = []
        for k in K_VALUES:
            sols = all_labels_by_method[method][k]
            #print(sols)
            sils = [silhouette_score(X, z) for z in sols]
            dbis = [davies_bouldin_score(X, z) for z in sols]
            
            rows.append({
                #"method": method,
                "k": k,
                "avg_silhouette": np.mean(sils),
                "std_silhouette": np.std(sils),
                "avg_dbi": np.mean(dbis),
                "std_dbi": np.std(dbis)
            })
        temp_df = pd.DataFrame(rows)
        method_metric_df[method] = temp_df

    for method in all_labels_by_method:
        rows = []
        for k in K_VALUES:
            sols = all_labels_by_method[method][k]
            #print(sols)
            sils = [silhouette_score(X, z) for z in sols]
            dbis = [davies_bouldin_score(X, z) for z in sols]
            
            rows.append({
                #"method": method,
                "k": k,
                "avg_silhouette": np.mean(sils),
                "std_silhouette": np.std(sils),
                "avg_dbi": np.mean(dbis),
                "std_dbi": np.std(dbis)
            })
        temp_df = pd.DataFrame(rows)
        method_metric_df[method] = temp_df

    if do_save:
        print(f"Saving all labels to ../results/clustering/{save_prefix}all_labels.npy")
        np.save(f'../results/clustering/{save_prefix}all_labels.npy', all_labels) 

        print(f"Saving internal metrics to ../results/clustering/{save_prefix}metrics_internal.csv")
        internal_met.to_csv(
            f"../results/clustering/{save_prefix}metrics_internal.csv",
            index=False
        )

    print("\n")
    return all_labels, all_labels_by_method, internal_met, method_metric_df, labels_info


def consensus_from_coassoc(C, 
                           n_clusters, 
                           consensus_clustering = 'Agglomerative', 
                           agg_linkage="average"):
    D = 1 - C
    
    # hierarchical ordering for heatmaps
    Z = linkage(squareform(D), method="average")
    order = leaves_list(Z)

    # CONSENSUS CLUSTERING -> GET CONSENSUS LABELS
    if consensus_clustering == "Agglomerative":
        agg = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage=agg_linkage # {‘average’, ‘single’}
        )
        consensus_labels = agg.fit_predict(D)

        consensus_labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    elif consensus_clustering == "Spectral":
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed'
        )
        consensus_labels = sc.fit_predict(C)

    else:
        raise ValueError('Unknown clustering for this consensus')

    #print("\n")
    return consensus_labels, D, Z, order


def external_metrics(all_labels,
                     do_save=False,
                     save_prefix=""):
    K_VALUES = list(all_labels.keys())
    external_met = pd.DataFrame(K_VALUES, columns = ['k'])

    ### ARI AND NMI FOR STABILITY
    print("Computing ARI and NMI scores...")
    stability_ari = {}
    stability_ari_std = {}
    stability_nmi = {}
    stability_nmi_std = {}
    for k, label_list in all_labels.items():
        ari_val = []
        nmi_val = []
        
        # all pairs of runs
        for z1, z2 in itertools.combinations(label_list, 2):
            ari_val.append(adjusted_rand_score(z1, z2))
            nmi_val.append(normalized_mutual_info_score(z1, z2))

        stability_ari[k] = np.mean(ari_val)
        stability_ari_std[k] = np.std(ari_val)
        stability_nmi[k] = np.mean(nmi_val)
        stability_nmi_std[k] = np.std(nmi_val)

    external_met['avg_ARI'] = stability_ari.values()
    external_met['std_ARI'] = stability_ari_std.values()
    external_met['avg_NMI'] = stability_nmi.values()
    external_met['std_NMI'] = stability_nmi_std.values()
    print("ARI and NMI scores done...")

    if do_save:
        print(f"Saving external metrics...")
        # SAVING RESULTS
        external_met.to_csv(f"../results/clustering/{save_prefix}metrics_external.csv", index=False)

    print("\n")
    return external_met


def cluster_coassoc_stability(Ck, labels_consensus):
    # Adjusted using ChatGPT
    labels_consensus = np.asarray(labels_consensus)
    clusters = np.unique(labels_consensus)

    per_cluster = {}
    pair_counts = {}
    for c in clusters:
        idx = np.where(labels_consensus == c)[0]
        nc = len(idx)
        if nc < 2:
            per_cluster[c] = np.nan
            pair_counts[c] = 0
            continue

        sub = Ck[np.ix_(idx, idx)]
        # take upper triangle without diagonal
        tri = sub[np.triu_indices(nc, k=1)]
        per_cluster[c] = float(tri.mean())
        pair_counts[c] = int(nc * (nc - 1) // 2)

    # aggregate
    vals = np.array([v for v in per_cluster.values() if np.isfinite(v)])
    mean_unweighted = float(vals.mean()) if len(vals) else np.nan

    weights = np.array([pair_counts[c] for c in clusters if np.isfinite(per_cluster[c])], dtype=float)
    weighted_vals = np.array([per_cluster[c] for c in clusters if np.isfinite(per_cluster[c])], dtype=float)
    mean_weighted = float(np.average(weighted_vals, weights=weights)) if weights.sum() > 0 else np.nan

    return per_cluster, mean_unweighted, mean_weighted


def consensus_clustering(X,
                         all_labels, 
                         name_order,
                         consensus_clustering = 'Agglomerative',
                         agg_linkage='average',
                         do_save=False,
                         save_prefix=""
                         ):
    
    consensus_labels = pd.DataFrame()
    consensus_labels['name'] = name_order

    K_VALUES = list(all_labels.keys())

    ### CONSTRUCTIONS PER K
    k_dic = {}

    ### STABILITY SCORES
    station_stabilities = pd.DataFrame(name_order, columns=['station'])
    avg_stability = {}
    std_stability = {}

    cons_silhouette = {}
    cons_dbi = {}

    cluster_stability = {}          # per-cluster dicts
    mean_cluster_stab = {}          # unweighted mean
    mean_cluster_stab_w = {}        # weighted mean
    min_cluster_stab = {}           # weakest cluster (very useful!)

    print("Loading constructions and computing stability scores...")
    for k in all_labels.keys():
        k_labels = all_labels[k]
        k_C = build_coassoc_matrix(k_labels)

        #if do_save:
        #    print("Saving co-association matrices")
        #    np.save(f"../results/clustering/similarity_matrix_k{k}.npy", k_C)

        k_consensus, k_D, k_Z, k_order = consensus_from_coassoc(k_C, 
                                                                k,
                                                                consensus_clustering=consensus_clustering,
                                                                agg_linkage=agg_linkage)
        per_cluster, mean_unw, mean_w = cluster_coassoc_stability(k_C, k_consensus) # integrated using ChatPGT


        k_stab = station_stability_scores(k_consensus, k_C)
        station_stabilities[k] = k_stab
        avg_stability[k] = np.mean(k_stab)
        std_stability[k] = np.std(k_stab)

        cons_silhouette[k] = silhouette_score(X, k_consensus)
        cons_dbi[k] = davies_bouldin_score(X, k_consensus)

        cluster_stability[k] = per_cluster
        mean_cluster_stab[k] = mean_unw
        mean_cluster_stab_w[k] = mean_w
        min_cluster_stab[k] = np.nanmin(list(per_cluster.values()))

        k_dic[k] = {
            "coassoc": k_C,
            "dist": k_D,
            "Z": k_Z,
            "order": k_order,
            "cons_labels": k_consensus
        }
        consensus_labels[f'k_{k}'] = k_consensus

    # Stack and average coassociation matrices across k
    C_list = [k_dic[k]["coassoc"] for k in K_VALUES]  # list of (n, n) arrays
    C_stack = np.stack(C_list, axis=0)   # shape: (n_k, n, n)
    C_avg   = C_stack.mean(axis=0)       # shape: (n, n)

    metrics = pd.DataFrame(K_VALUES, columns = ['k'])

    metrics['avg_stability'] = avg_stability.values()
    metrics['std_stability'] = std_stability.values()

    metrics['mean_cluster_stability'] = mean_cluster_stab.values()
    metrics['mean_cluster_stability_w'] = mean_cluster_stab_w.values()
    metrics['min_cluster_stability'] = min_cluster_stab.values()

    metrics['cons_silhouette'] = cons_silhouette.values()
    metrics['cons_dbi'] = cons_dbi.values()

    if do_save:
        print("Saving cluster results and metrics...")
        consensus_labels.to_csv(f"../results/clustering/{save_prefix}cons_labels.csv", index=False)
        metrics.to_csv(f"../results/clustering/{save_prefix}metrics_consensus.csv", index=False)
        np.save(f"../results/clustering/{save_prefix}run_results.npy", k_dic)
        np.save(f"../results/clustering/{save_prefix}similarity_matrix_avg_k{K_VALUES[-1]}.npy", C_avg) # NEW

    print("\n")
    return consensus_labels, metrics, k_dic, C_avg