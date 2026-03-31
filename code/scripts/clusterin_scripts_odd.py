def run_base_clustering_agglomerative(X, 
                                      domain_name,
                                      name_order, 
                                      K_VALUES=range(2,11), 
                                      N_RUNS=10,
                                      do_save=False,
                                      save_prefix=""
                                      ):
    # STORING LABELS
    all_labels = {}
    labels_df = pd.DataFrame()
    labels_df['name'] = name_order

    # To store ensemble metrics
    avg_inertia = {}, std_inertia = {}, all_inertia = {}

    avg_silhouette = {}, std_silhouette = {}, all_silhouette = {}

    avg_dbi = {}, std_dbi = {}, all_dbi = {}

    internal_met = pd.DataFrame(K_VALUES, columns = ['k'])

    # ============================================
    # RUN BASE ENSEMBLE FOR EACH k
    # ============================================
    print("Starting Clustering...")
    for k in K_VALUES:
        labels_k = []
        inertia_k = []
        sil_k = []
        dbi_k = []
        
        # Run N_RUNS of KMeans for this k
        for r in range(N_RUNS):

            ## KMEANS

            km = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=1,                 # IMPORTANT: one init per run; randomness comes from seed
                random_state=r            # different seed → different solution
            )
                
            z = km.fit_predict(X)
            labels_k.append(z)
            inertia_k.append(km.inertia_)
            
            for lk in ['ward', 'complete', 'average', 'single']:
                ag = AgglomerativeClustering(
                    n_clusters=k,
                    linkage=lk
                )
                
                z = ag.fit_predict(X)
                labels_k.append(z)
                

                # Silhouette score for this run
                sil_k.append(silhouette_score(X, z))
                dbi_k.append(davies_bouldin_score(X, z))

            # Silhouette score for this run
            sil_k.append(silhouette_score(X, z))
            dbi_k.append(davies_bouldin_score(X, z))
        

        # Store results for this k
        all_labels[k] = labels_k

        # only store inertia if kMeans
        if base_clustering == 'KMeans':
            avg_inertia[k] = np.mean(inertia_k)
            std_inertia[k] = np.std(inertia_k)
            all_inertia[k] = inertia_k

        avg_silhouette[k] = np.mean(sil_k)
        std_silhouette[k] = np.std(sil_k)
        all_silhouette[k] = sil_k

        avg_dbi[k] = np.mean(dbi_k)
        std_dbi[k] = np.std(dbi_k)
        all_dbi[k] = dbi_k

    print("Clustering done... Storing Internal metrics")

    if base_clustering == 'KMeans':
        internal_met['avg_inertia'] = avg_inertia.values()
        internal_met['std_inertia'] = std_inertia.values()

    internal_met['avg_silhouette'] = avg_silhouette.values()
    internal_met['std_silhouette'] = std_silhouette.values()
    internal_met['avg_dbi'] = avg_dbi.values()
    internal_met['std_dbi'] = std_dbi.values()

    if do_save:
        print(f"Saving all cluster results and internal metrics")
        # save all labels too
        all_labels_df = pd.DataFrame()
        for k in K_VALUES:
            labels_k_df = pd.DataFrame(all_labels[k], columns=[f'run_{i+1}' for i in range(N_RUNS)])
            labels_k_df['name'] = name_order
            #labels_k_df.to_csv(f"../results/clustering/{domain_name}/base_{base_clustering}_labels_k{k}{save_prefix}.csv", index=False)
            all_labels_df = pd.concat([all_labels_df, labels_k_df], axis=0)
            all_labels_df.to_csv(f"../results/clustering/{domain_name}/base_{base_clustering}_labels_k{k}{save_prefix}.csv", index=False)

        internal_met.to_csv(f"../results/clustering/{domain_name}/km{save_prefix}_internal_metrics.csv", index=False)

    # and return all_labels_df?
    return all_labels, all_labels_df, internal_met
