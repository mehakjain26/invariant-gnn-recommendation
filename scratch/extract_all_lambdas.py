import json
import os

def get_metrics(path):
    try:
        results_path = os.path.join(path, 'results.json')
        with open(results_path, 'r') as f:
            data = json.load(f)
            recall = data['test']['recall@20']
            falling = data['subgroups']['falling']['recall@20']
            rising = data['subgroups']['rising']['recall@20']
            return f"{recall:.4f}, {falling:.4f}, {rising:.4f}"
    except Exception as e:
        return f"Error: {e}"

lambdas = ["0.1", "1", "10", "100"]

print("Amazon IRM:")
for l in lambdas:
    print(f"  lam{l}: {get_metrics(f'results/layer_3/amazon/irm_lam{l}')}")

print("\nAmazon V-REx:")
for l in lambdas:
    print(f"  lam{l}: {get_metrics(f'results/layer_3/amazon/vrex_lam{l}')}")

print("\nYelp IRM:")
for l in lambdas:
    print(f"  lam{l}: {get_metrics(f'results/layer_3/yelp/irm_lam{l}')}")

print("\nYelp V-REx:")
for l in lambdas:
    print(f"  lam{l}: {get_metrics(f'results/layer_3/yelp/vrex_lam{l}')}")
