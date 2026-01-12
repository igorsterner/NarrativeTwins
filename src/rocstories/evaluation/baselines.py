import json
import random

import numpy as np

from src.rocstories.evaluation.metrics import auc, spearman


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_data(path):
    with open(path, "r") as f:
        return json.load(f)


def evaluate_once(dataset, scores_fn):
    sps, aucs = [], []
    for _, item in dataset.items():
        labels = item["most_important"]
        scores = scores_fn(item)
        sps.append(spearman(scores, labels))
        aucs.append(auc(scores, labels))
    return {
        "spearman": float(np.mean(sps)),
        "auc": float(np.mean(aucs)),
    }


def main():

    DATA_FILE = "data/rocstories/split/test.json"
    SEED = 42

    dataset = load_data(DATA_FILE)

    # Increasing
    inc_scores = lambda _: [1.0, 2.0, 3.0, 4.0, 5.0]
    inc_res = evaluate_once(dataset, inc_scores)
    print("Increasing baseline:")
    print(f"  spearman: {inc_res['spearman']:.4f}")
    print(f"  auc:      {inc_res['auc']:.4f}")

    # Decreasing
    dec_scores = lambda _: [5.0, 4.0, 3.0, 2.0, 1.0]
    dec_res = evaluate_once(dataset, dec_scores)
    print("Decreasing baseline:")
    print(f"  spearman: {dec_res['spearman']:.4f}")
    print(f"  auc:      {dec_res['auc']:.4f}")

    # Random
    rand_means = {"spearman": [], "auc": []}
    for t in range(100):
        set_seed(SEED + t)
        rand_scores = lambda _: np.random.random(5).tolist()
        res = evaluate_once(dataset, rand_scores)
        for k in rand_means:
            rand_means[k].append(res[k])

    print("Random baseline:")
    for k in ["spearman", "auc"]:
        mean_k = float(np.mean(rand_means[k]))
        std_k = float(np.std(rand_means[k], ddof=1))
        print(f"  {k}: mean={mean_k:.4f} std={std_k:.4f}")


if __name__ == "__main__":
    main()
