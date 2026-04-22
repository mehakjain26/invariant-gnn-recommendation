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

def find_file(dataset, method, e_key, lam):
    if e_key == "E2":
        base = "cluster_results_E2/results"
        name = f"{dataset}_ngcf_{method}_{e_key}_lam{lam}.0/results.json"
        if dataset == "amazon-books" and e_key == "E2":
             # Try without ngcf for some
             pass
    elif e_key == "E3":
        base = f"results/layer_3/{'amazon' if 'amazon' in dataset else 'yelp'}"
        name = f"{method}_lam{lam if '.0' not in str(lam) else int(float(lam))}/results.json"
    elif e_key == "E5":
        base = "cluster_results_E5/results"
        name = f"{dataset}_ngcf_{method}_vanilla_{e_key}_lam{lam}.0/results.json"
    
    # Actually just use a more robust search
    if e_key == "E2":
        dirs = [f"{dataset}_ngcf_{method}_E2_lam{lam}.0", f"{dataset}_{method}_E2_lam{lam}.0"]
    elif e_key == "E3":
        dirs = [f"{method}_lam{int(float(lam))}", f"{method}_lam{lam}"]
    elif e_key == "E5":
        dirs = [f"{dataset}_ngcf_{method}_vanilla_E5_lam{lam}.0"]
    
    for d in dirs:
        full_path = os.path.join(base, d, "results.json")
        if os.path.exists(full_path):
            return full_path
    return None

results = {}
for dataset in ["amazon-books", "yelp2018"]:
    results[dataset] = {}
    for method in ["irm", "vrex"]:
        results[dataset][method] = {}
        for lam in ["1", "10", "100"]:
            results[dataset][method][lam] = {}
            for e_key in ["E2", "E3", "E5"]:
                path = find_file(dataset, method, e_key, lam)
                if path:
                    results[dataset][method][lam][e_key] = get_metrics(path)
                else:
                    results[dataset][method][lam][e_key] = "N/A"

print(json.dumps(results, indent=2))
