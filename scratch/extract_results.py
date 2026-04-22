import json
import os

def get_metrics(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            recall = data['test']['recall@20']
            falling = data['subgroups']['falling']['recall@20']
            rising = data['subgroups']['rising']['recall@20']
            return f"{recall:.4f}, {falling:.4f}, {rising:.4f}"
    except Exception as e:
        return f"Error: {e}"

files = {
    "Table 2 (Amazon)": [
        ("Baseline", "results/layer_3/amazon/baseline_temporal/results.json"),
        ("V-REx 1.0", "results/layer_3/amazon/vrex_lam1/results.json"),
        ("V-REx 100.0", "results/layer_3/amazon/vrex_lam100/results.json"),
        ("IRM 1.0", "results/layer_3/amazon/irm_lam1/results.json"),
        ("IRM 100.0", "results/layer_3/amazon/irm_lam100/results.json"),
    ],
    "Table 3 (Yelp)": [
        ("Baseline", "results/layer_3/yelp/baseline_temporal/results.json"),
        ("V-REx 10.0", "results/layer_3/yelp/vrex_lam10/results.json"),
        ("V-REx 100.0", "results/layer_3/yelp/vrex_lam100/results.json"),
        ("IRM 10.0", "results/layer_3/yelp/irm_lam10/results.json"),
        ("IRM 100.0", "results/layer_3/yelp/irm_lam100/results.json"),
    ],
    "Table 4 (Layers Amazon)": [
        ("L1", "results/layer_1/amazon-books_ngcf_baseline_vanilla_E3_lam1.0_reg1e-4/results.json"),
        ("L2", "results/layer_2/amazon-books_ngcf_baseline_vanilla_E3_lam1.0_reg1e-4/results.json"),
        ("L3", "results/layer_3/amazon/baseline_temporal/results.json"),
        ("L4", "results/layer_4/amazon-books_ngcf_baseline_vanilla_E3_lam1.0_reg1e-4/results.json"),
    ],
    "Table 4 (Layers Yelp)": [
        ("L1", "results/layer_1/yelp2018_ngcf_baseline_vanilla_E3_lam1.0_reg1e-4/results.json"),
        ("L2", "results/layer_2/yelp2018_ngcf_baseline_vanilla_E3_lam1.0_reg1e-4/results.json"),
        ("L3", "results/layer_3/yelp/baseline_temporal/results.json"),
        ("L4", "results/layer_4/yelp2018_ngcf_baseline_vanilla_E3_lam1.0_reg1e-4/results.json"),
    ],
    "Table 5 (BPR Ablation)": [
        ("Yelp V-REx Vanilla", "results/layer_3/yelp/vrex_lam10/results.json"),
        ("Yelp V-REx Balanced", "results/layer_3_non_vanilla/yelp/vrex_lam10/results.json"),
        ("Yelp IRM Vanilla", "results/layer_3/yelp/irm_lam10/results.json"),
        ("Yelp IRM Balanced", "results/layer_3_non_vanilla/yelp/irm_lam10/results.json"),
        ("Amazon V-REx Vanilla", "results/layer_3/amazon/vrex_lam10/results.json"),
        ("Amazon V-REx Balanced", "results/layer_3_non_vanilla/amazon/vrex_lam10/results.json"),
        ("Amazon IRM Vanilla", "results/layer_3/amazon/irm_lam10/results.json"),
        ("Amazon IRM Balanced", "results/layer_3_non_vanilla/amazon/irm_lam10/results.json"),
    ]
}

for section, items in files.items():
    print(f"\n{section}:")
    for name, path in items:
        print(f"  {name}: {get_metrics(path)}")
