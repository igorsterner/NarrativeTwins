import json
import os

import numpy as np
import torch
import tqdm
import transformers
from omegaconf import OmegaConf

from src.wikipedia import operations
from src.wikipedia.data_manager import data_utils
from src.wikipedia.training.model import SiameseModernBERT


class Evaluator:
    def __init__(self, cfg, model, tokenizer, device):

        with open(cfg.paths.tripod_json, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.ops = {
            "deletion": operations.deletion.Deletion(model, tokenizer, device),
            "summarisation": operations.summarisation.Summarisation(
                model, tokenizer, device
            ),
            "shifting": operations.shifting.Shifting(model, tokenizer, device),
            "disruption": operations.disruption.Disruption(model, tokenizer, device),
        }

    def auc(self, scores, idx):
        n = len(scores)

        order = np.argsort(scores)[::-1]
        r = np.where(order == idx)[0][0] + 1

        return (n - r) / (n - 1)

    def evaluate(self):
        results = {k: {i: [] for i in range(5)} for k in self.ops}
        saliency_scores_per_op = {name: {} for name in self.ops}  # <--- NEW

        for story_id, rec in tqdm.tqdm(self.data.items()):

            sents = rec["sentences"]
            n = len(sents)
            wins = data_utils.compute_windows_even(n)
            gold = [rec["tp1"], rec["tp2"], rec["tp3"], rec["tp4"], rec["tp5"]]
            op_scores = {name: op.scores(sents) for name, op in self.ops.items()}

            for op_name in self.ops:
                saliency_scores_per_op[op_name][story_id] = op_scores[op_name]

            for w_id, (lo, hi) in enumerate(wins, start=1):
                tp = gold[w_id - 1]
                if not (lo <= tp <= hi):
                    continue
                rel = tp - lo
                for name in self.ops:
                    scores_w = op_scores[name][w_id - 1]
                    auc = self.auc(scores_w, rel)
                    results[name][w_id - 1].append(auc)

        mean_results = {}
        for op in results.keys():
            auc_values = []
            for i in range(5):
                mean_auc = float(np.mean(results[op][i]))
                mean_results[f"auc_{i+1}/{op}"] = mean_auc
                auc_values.append(mean_auc)

            mean_results[f"auc_avg/{op}"] = float(np.mean(auc_values))

        return mean_results, saliency_scores_per_op  # <--- NEW


if __name__ == "__main__":

    METHODS = [
        # ("simcse", "Dropout Twins"),
        ("nt", "Narrative Twins"),
        # ("mlm", "Masked LM"),
    ]

    SEEDS = [43]  # , 127, 263]
    TP_LABELS = ["TP1", "TP2", "TP3", "TP4", "TP5", "avg."]

    OP_DISPLAY = {
        "deletion": "Deletion",
        "shifting": "Shifting",
        "disruption": "Disruption",
        "summarisation": "Summarization",
    }

    OP_ORDER = ["deletion", "shifting", "disruption", "summarisation"]

    results_collect = {op: {m[0]: [] for m in METHODS} for op in OP_ORDER}

    cfg_path = "src/wikipedia/configs/config.yaml"
    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.model.base_model, use_fast=True
    )

    os.makedirs("data/wikipedia/results", exist_ok=True)

    for method, m_disp in METHODS:
        model = SiameseModernBERT(cfg.model.base_model, cfg.model.pooling, method).to(
            device
        )
        for seed in SEEDS:

            ckpt_path = f"outputs_wikipedia/model.pt"

            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
            ev = Evaluator(cfg, model, tokenizer, device)
            res, saliency_scores_per_op = ev.evaluate()

            for op in OP_ORDER:
                fname = f"data/wikipedia/results/{method}-{seed}-{op}.json"
                with open(fname, "w") as f:
                    json.dump(saliency_scores_per_op[op], f, indent=4)

            for op in OP_ORDER:
                entry = []
                for i in range(5):
                    v = res.get(f"auc_{i+1}/{op}", float("nan"))
                    entry.append(v)
                entry.append(res.get(f"auc_avg/{op}", float("nan")))
                results_collect[op][method].append(entry)

    def format(mean, std):
        return f"{mean:.2f}$_{{\\pm{std:.2f}}}$"

    out = []
    out.append("\\begin{table*}[t]")
    out.append("\\centering")
    out.append("\\caption{TODO}")
    out.append("\\label{tab:tripod-results}")
    out.append("\\begin{small}")
    out.append("\\begin{tabular}{lrrrrrr}")
    out.append("\\toprule")
    out.append(
        " & "
        + " & ".join([f"\\multicolumn{{1}}{{c}}{{{tp}}}" for tp in TP_LABELS])
        + " \\\\ \\midrule"
    )

    for op in OP_ORDER:
        out.append(
            f"\\multicolumn{{7}}{{c}}{{\\textsc{{{OP_DISPLAY[op]} Operation}}}} \\\\ \\midrule"
        )
        for method, m_disp in METHODS[
            ::-1
        ]:  # To match \textsc{Dropout Twins}, \textsc{Narrative Twins}
            vals = results_collect[op][method]  # List of 3 [TP1, TP2, ..., avg.]
            vals_np = np.array(vals)  # shape: (3,6)
            means = np.nanmean(vals_np, axis=0)  # (6,)
            stds = np.nanstd(vals_np, axis=0)
            line = (
                f"\\textsc{{{m_disp}}} & "
                + " & ".join(format(m, s) for m, s in zip(means, stds))
                + " \\\\"
            )
            out.append(line)
        out.append("\\midrule")
    out[-1] = "\\bottomrule"  # Fix last midrule

    out += [
        "\\end{tabular}",
        "\\end{small}",
        "",
        "\\end{table*}",
    ]
    print("\n".join(out))
