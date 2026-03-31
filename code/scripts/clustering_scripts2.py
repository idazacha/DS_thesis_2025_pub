import numpy as np
import pandas as pd
import itertools
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform
from scripts.clustering_utils import build_coassoc_matrix, station_stability_scores

def run_base_clustering(X, 
                   domain_name,
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
        #inertia_k = []
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
        internal_met.to_csv(f"../results/clustering/{domain_name}/{save_prefix}_internal_metrics.csv", index=False)

    return all_labels, internal_met
