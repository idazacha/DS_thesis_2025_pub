import numpy as np
from itertools import product

def build_configs(base_methods, params, runs_per_method, N_RUNS):
    configs = []

    for method in base_methods:
        if method == "KMeans":
            configs.append({
                "name": "KMeans",
                "method": "KMeans",
                "params": {},
                "runs": runs_per_method.get("KMeans", N_RUNS)
            })

        elif method == "GMM":
            gmm_spec = params.get("GMM", {})
            cov_types = gmm_spec.get("covariance_type", ["full"])
            for cov in cov_types:
                name = f"GMM_cov-{cov}"
                configs.append({
                    "name": name,
                    "method": "GMM",
                    "params": {"covariance_type": cov},
                    "runs": runs_per_method.get("GMM", N_RUNS)
                })

        elif method == "Spectral":
            spec_spec = params.get("Spectral", {})
            # assume all values are lists; use product
            keys = list(spec_spec.keys())
            values_lists = [spec_spec[k] for k in keys]
            for values in product(*values_lists):
                p = dict(zip(keys, values))
                name_parts = [f"{k}-{v}" for k, v in p.items()]
                name = "Spectral_" + "_".join(name_parts)
                configs.append({
                    "name": name,
                    "method": "Spectral",
                    "params": p,
                    "runs": runs_per_method.get("Spectral", N_RUNS)
                })

        elif method == "Agglomerative":
            agg_spec = params.get("Agglomerative", {})
            linkages = agg_spec.get("linkage", ["ward"])
            metric_map = agg_spec.get("metric", {})
            for link in linkages:
                for met in metric_map.get(link, ["euclidean"]):
                    name = f"Agg_{link}_{met}"
                    configs.append({
                        "name": name,
                        "method": "Agglomerative",
                        "params": {"linkage": link, "metric": met},
                        "runs": runs_per_method.get("Agglomerative", 1)
                    })

        else:
            raise ValueError(f"Unsupported method: {method}")

    return configs


def build_coassoc_matrix(labels_list):
    n = len(labels_list[0])
    C = np.zeros((n, n), dtype=float)

    for labels in labels_list:
        eq = (labels[:, None] == labels[None, :]).astype(int)
        C += eq

    return C / len(labels_list)

def station_stability_scores(consensus_labels, C):
    n = len(consensus_labels)
    R = np.zeros(n)
    
    for i in range(n):
        cluster = consensus_labels[i]
        same = np.where(consensus_labels == cluster)[0]
        same = same[same != i]  # exclude itself*
        
        if len(same) == 0:
            R[i] = 1.0
        else:
            R[i] = C[i, same].mean()
    
    return R