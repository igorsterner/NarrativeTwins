from pathlib import Path

import hydra
import torch
import transformers
import wandb
from tqdm import tqdm

from src.rocstories import evaluation, training


@hydra.main(version_base=None, config_path="../configs/", config_name="config")
def main(cfg):

    training.utils.set_seed(cfg.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project=cfg.paths.wandb_project, config=dict(cfg))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.model.base_model, use_fast=True
    )

    dataset = training.data.build_train_dataset(cfg)
    dataloader = training.data.build_train_dataloader(cfg, tokenizer, dataset, device)

    epochs = cfg.training.epochs
    total_steps = epochs * len(dataloader)

    model = training.model.ROCStoryBERT(cfg.model.base_model, cfg.model.pooling).to(
        device
    )

    model.tokenizer = tokenizer

    tr = training.trainer.Trainer(cfg, model, tokenizer, dataset, total_steps, device)
    ev = evaluation.evaluator.Evaluator(cfg, model, tokenizer, device)

    step = 0
    accumulated_metrics = {}
    accumulated_steps = 0

    for epoch in range(epochs):
        pbar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:

            if step % cfg.training.eval_every == 0:
                val_metrics, _ = ev.evaluate()
                wandb.log(val_metrics, step=step)

            metrics = tr.train_step(batch)
            step += 1

            for k, v in metrics.items():
                accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v
            accumulated_steps += 1

            wandb.log({"lr": tr.optim.param_groups[0]["lr"]}, step=step)

            if step % cfg.training.log_every == 0:
                avg_metrics = {
                    f"train/{k}": v / accumulated_steps
                    for k, v in accumulated_metrics.items()
                }
                wandb.log(avg_metrics, step=step)
                accumulated_metrics = {}
                accumulated_steps = 0

    final_metrics, _ = ev.evaluate()
    wandb.log(final_metrics, step=step)

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg.model.base_model.split("/")[-1]

    output_file = output_dir / f"model.pt"

    torch.save(model.state_dict(), output_file)
    wandb.finish()


if __name__ == "__main__":
    main()
