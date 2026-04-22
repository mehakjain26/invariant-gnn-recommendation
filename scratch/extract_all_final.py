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

results = {}

# E2
e2_base = "cluster_results_E2/results"
for d in os.listdir(e2_base):
    if not os.path.isdir(os.path.join(e2_base, d)): continue
    res_path = os.path.join(e2_base, d, "results.json")
    if os.path.exists(res_path):
        metrics = get_metrics(res_path)
        results[f"E2_{d}"] = metrics

# E5
e5_base = "cluster_results_E5/results"
for d in os.listdir(e5_base):
    if not os.path.isdir(os.path.join(e5_base, d)): continue
    res_path = os.path.join(e5_base, d, "results.json")
    if os.path.exists(res_path):
        metrics = get_metrics(res_path)
        results[f"E5_{d}"] = metrics

# E3
for ds in ["amazon", "yelp"]:
    e3_base = f"results/layer_3/{ds}"
    if not os.path.exists(e3_base): continue
    for d in os.listdir(e3_base):
        if not os.path.isdir(os.path.join(e3_base, d)): continue
        res_path = os.path.join(e3_base, d, "results.json")
        if os.path.exists(res_path):
            metrics = get_metrics(res_path)
            results[f"E3_{ds}_{d}"] = metrics

print(json.dumps(results, indent=2))
