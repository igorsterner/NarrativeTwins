import json
from pathlib import Path

import numpy as np
import torch
import transformers
from omegaconf import OmegaConf
from tqdm import tqdm

from src.rocstories.evaluation import inference, metrics
from src.rocstories.operations import (deletion, disruption, shifting,
                                       summarisation)
from src.rocstories.training import model as training_model


class Evaluator:
    def __init__(self, cfg, model, tokenizer, device):
        self.cfg = cfg
        self.model = model
        self.tok = tokenizer
        self.device = device

        with open(cfg.paths.evaluation_data, "r") as f:
            print(f"Loading evaluation data from {cfg.paths.evaluation_data}")
            self.val = json.load(f)

        self.encoder = inference.Encoder(self.model, self.tok, self.device)

        self.operations = {
            "deletion": deletion.Deletion(self.encoder),
            "disruption": disruption.Disruption(self.encoder),
            "shifting": shifting.Shifting(self.encoder),
            "summarisation": summarisation.Summarisation(self.encoder),
        }

        self.metric_names = cfg.evaluation.metrics

    def evaluate(self):
        results = {
            operation: {metric: [] for metric in self.cfg.evaluation.metrics}
            for operation in self.operations
        }

        saliency_scores_per_op = {op: {} for op in self.operations}

        for story_id, item in tqdm(self.val.items()):
            sents = item["story"]
            labels = [int(x) for x in item["most_important"]]

            for operation in self.operations:
                saliency_scores = self.operations[operation].scores(sents)
                saliency_scores_per_op[operation][story_id] = list(
                    map(float, saliency_scores)
                )

                for metric in self.cfg.evaluation.metrics:
                    results[operation][metric].append(
                        getattr(metrics, metric)(saliency_scores, labels)
                    )

        mean_results = {}
        for operation, metric_dict in results.items():
            for metric, values in metric_dict.items():
                mean_results[f"{metric}/{operation}"] = float(np.mean(values))

        # Return both mean_results and the collected saliency scores
        return mean_results, saliency_scores_per_op


if __name__ == "__main__":

    cfg = OmegaConf.load("src/rocstories/configs/config.yaml")
    cfg.paths.evaluation_data = "data/rocstories/split/test.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.model.base_model, use_fast=True
    )

    seeds = [43]  # , 127, 263
    loss_types = [
        # ("mlm", "Masked LM"),
        # ("simcse", "Dropout Twins"),
        ("nt", "Narrative Twins"),
    ]
    ops_order = ["deletion", "shifting", "disruption", "summarisation"]

    def fmt(mean, std):
        return f"{mean:.2f}$_{{\\pm{std:.2f}}}$"

    Path("data/rocstories/results/").mkdir(parents=True, exist_ok=True)

    for loss_type, label in loss_types:
        agg = {op: {"spearman": [], "auc": []} for op in ops_order}

        for seed in seeds:
            cfg.loss.type = loss_type
            cfg.training.seed = seed

            model = training_model.ROCStoryBERT(
                cfg.model.base_model, cfg.model.pooling
            ).to(device)

            ckpt_path = Path(cfg.paths.output_dir) / f"model.pt"

            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()

            evaluator = Evaluator(cfg, model, tokenizer, device)
            res, saliency_scores_per_op = evaluator.evaluate()  # Updated to get scores

            for op in ops_order:
                fname = (
                    f"data/rocstories/results/base-mean-{cfg.training.seed}-{op}.json"
                )
                with open(fname, "w") as f:
                    json.dump(saliency_scores_per_op[op], f, indent=2)

            for op in ops_order:
                agg[op]["spearman"].append(res[f"spearman/{op}"])
                agg[op]["auc"].append(res[f"auc/{op}"])

        cells = []
        for op in ops_order:
            sp = np.array(agg[op]["spearman"])
            au = np.array(agg[op]["auc"])
            cells.append(fmt(np.mean(sp), np.std(sp)))
            cells.append(fmt(np.mean(au), np.std(au)))

        print("\\textsc{" + label + "} & " + " & ".join(cells) + r" \\")
