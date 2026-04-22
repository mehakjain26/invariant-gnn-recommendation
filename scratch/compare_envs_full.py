import json
import os

def get_metrics(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            recall = data['test']['recall@20']
            falling = data['subgroups']['falling']['recall@20']
            rising = data['subgroups']['rising']['recall@20']
            return {"Overall": round(recall, 4), "Falling": round(falling, 4), "Rising": round(rising, 4)}
    except Exception as e:
        return None

base_paths = {
    "E2": "cluster_results_E2/results",
    "E3": "results",
    "E5": "cluster_results_E5/results"
}

datasets = ["amazon-books", "yelp2018"]
methods = ["irm", "vrex"]
lambdas = ["1.0", "10.0", "100.0"]

def find_file(base, dataset, method, e_key, lam):
    patterns = [
        f"{dataset}_{method}_{e_key}_lam{lam}/results.json",
        f"{dataset}_ngcf_{method}_{e_key}_lam{lam}/results.json",
        f"{dataset}_ngcf_{method}_vanilla_{e_key}_lam{lam}/results.json",
    ]
    for p in patterns:
        full_path = os.path.join(base, p)
        if os.path.exists(full_path):
            return full_path
    return None

results = {}
for dataset in datasets:
    results[dataset] = {}
    for method in methods:
        results[dataset][method] = {}
        for lam in lambdas:
            results[dataset][method][lam] = {}
            for e_key in base_paths:
                path = find_file(base_paths[e_key], dataset, method, e_key, lam)
                if path:
                    results[dataset][method][lam][e_key] = get_metrics(path)
                else:
                    results[dataset][method][lam][e_key] = "N/A"

print(json.dumps(results, indent=2))
