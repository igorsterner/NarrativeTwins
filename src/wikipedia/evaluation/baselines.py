import json
import os
import random

import numpy as np
import yaml

from src.wikipedia.data_manager import data_utils


def auc(scores, idx):
    n = len(scores)

    order = np.argsort(scores)[::-1]
    r = np.where(order == idx)[0][0] + 1

    return (n - r) / (n - 1)


def load_tripod_path():
    base_dir = os.path.dirname(__file__)
    cfg_path = os.path.join(base_dir, "..", "configs", "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["paths"]["tripod_json"]


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def eval_mode_per_tp(dataset, mode, rng_seed=0, num_runs=100):
    num_tps = 5

    if mode == "random":
        all_run_tps = [[] for _ in range(num_tps)]
        all_run_avgs = []
        for run in range(num_runs):
            rng = random.Random(rng_seed + run)
            aucs_per_tp = [[] for _ in range(num_tps)]
            aucs_all = []
            for _, rec in dataset.items():
                sents = rec["sentences"]
                n = len(sents)
                wins = data_utils.compute_windows_even(n)
                gold = [rec["tp1"], rec["tp2"], rec["tp3"], rec["tp4"], rec["tp5"]]
                for w_id, (lo, hi) in enumerate(wins, 1):
                    tp = gold[w_id - 1]
                    if not (lo <= tp <= hi):
                        continue
                    rel = tp - lo
                    scores = [rng.random() for _ in range(lo, hi + 1)]
                    auc_val = auc(scores, rel)
                    aucs_per_tp[w_id - 1].append(auc_val)
                    aucs_all.append(auc_val)
            for i in range(num_tps):
                all_run_tps[i].append(
                    np.mean(aucs_per_tp[i]) if aucs_per_tp[i] else float("nan")
                )
            all_run_avgs.append(np.mean(aucs_all) if aucs_all else float("nan"))
        means = [np.nanmean(xs) for xs in all_run_tps]
        stds = [np.nanstd(xs) for xs in all_run_tps]
        mean_avg = np.nanmean(all_run_avgs)
        std_avg = np.nanstd(all_run_avgs)
        return means + [mean_avg], stds + [std_avg]

    aucs_per_tp = [[] for _ in range(num_tps)]
    aucs_all = []
    for _, rec in dataset.items():
        sents = rec["sentences"]
        n = len(sents)
        wins = data_utils.compute_windows_even(n)
        gold = [rec["tp1"], rec["tp2"], rec["tp3"], rec["tp4"], rec["tp5"]]
        for w_id, (lo, hi) in enumerate(wins, 1):
            tp = gold[w_id - 1]
            if not (lo <= tp <= hi):
                continue
            rel = tp - lo
            if mode == "increasing":
                scores = [i - lo for i in range(lo, hi + 1)]
            elif mode == "decreasing":
                scores = [hi - i for i in range(lo, hi + 1)]
            else:
                raise ValueError(mode)
            auc_val = auc(scores, rel)
            aucs_per_tp[w_id - 1].append(auc_val)
            aucs_all.append(auc_val)
    means = [np.mean(xs) if xs else float("nan") for xs in aucs_per_tp]
    mean_avg = np.mean(aucs_all) if aucs_all else float("nan")
    return means + [mean_avg], [0.0] * 6


def format_latex_row(name, means, stds):
    def f(m, s):
        return f"{m:.2f}$_{{\\pm{s:.2f}}}$" if s > 1e-5 else f"{m:.2f}"

    row = (
        f"\\textsc{{{name}}} & "
        + " & ".join(f(m, s) for m, s in zip(means, stds))
        + " \\\\"
    )
    return row


def main():
    path = load_tripod_path()
    dataset = load_dataset(path)
    labels = ["TP1", "TP2", "TP3", "TP4", "TP5", "avg."]

    modes = [
        ("increasing", "Increasing"),
        ("decreasing", "Decreasing"),
        ("random", "Random"),
    ]

    rows = []
    for mode, disp in modes:
        means, stds = eval_mode_per_tp(dataset, mode, rng_seed=0, num_runs=100)
        rows.append(format_latex_row(disp, means, stds))

    print("\\begin{tabular}{lrrrrrr}")
    print("\\toprule")
    print(" & " + " & ".join(labels) + " \\\\ \\midrule")
    for row in rows:
        print(row)
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
