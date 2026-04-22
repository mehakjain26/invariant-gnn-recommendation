import json
import os

def get_metrics(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            recall = data['test']['recall@20']
            falling = data['subgroups']['falling']['recall@20']
            rising = data['subgroups']['rising']['recall@20']
            return {"Overall": recall, "Falling": falling, "Rising": rising}
    except Exception as e:
        return None

base_paths = {
    "E2": "cluster_results_E2/results",
    "E3": "results",
    "E5": "cluster_results_E5/results"
}

scenarios = [
    ("IRM", "lam1.0"),
    ("IRM", "lam10.0"),
    ("V-REx", "lam1.0"),
    ("V-REx", "lam10.0")
]

# File mapping (names are slightly inconsistent)
def find_file(e_key, method, lam):
    base = base_paths[e_key]
    method_lower = method.lower()
    
    # Try different naming patterns
    patterns = [
        f"amazon-books_{method_lower}_{e_key}_{lam}/results.json",
        f"amazon-books_ngcf_{method_lower}_{e_key}_{lam}/results.json",
        f"amazon-books_ngcf_{method_lower}_vanilla_{e_key}_{lam}/results.json",
    ]
    
    for p in patterns:
        full_path = os.path.join(base, p)
        if os.path.exists(full_path):
            return full_path
    return None

results = {}
for e_key in base_paths:
    results[e_key] = {}
    for method, lam in scenarios:
        path = find_file(e_key, method, lam)
        if path:
            results[e_key][f"{method}_{lam}"] = get_metrics(path)
        else:
            results[e_key][f"{method}_{lam}"] = "N/A"

print(json.dumps(results, indent=2))
