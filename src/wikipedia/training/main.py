import pathlib

import hydra
import torch
import tqdm
import transformers
import wandb

from src.wikipedia import training


@hydra.main(version_base=None, config_path="../configs/", config_name="config")
def main(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training.utils.set_seed(cfg.training.seed)

    wandb.init(project=cfg.paths.wandb_project, config=dict(cfg))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.model.base_model, use_fast=True
    )
    dataloader = training.data.build_dataloader(cfg, tokenizer, device)

    model = training.model.SiameseModernBERT(
        cfg.model.base_model, cfg.model.pooling, cfg.loss.type
    ).to(device)
    model.tokenizer = tokenizer

    total_steps = cfg.training.epochs * len(dataloader)

    tr = training.trainer.Trainer(
        cfg, model, tokenizer, dataloader, total_steps, device
    )

    train_accum = training.utils.MetricsAccumulator(prefix="train/")

    tr.model.train()
    pbar = tqdm.tqdm(total=total_steps, desc="Training")
    step = 0

    for epoch in range(cfg.training.epochs):
        for batch in tr.dl:

            batch_metrics = tr.train_step(batch)
            train_accum.update(batch_metrics)

            if step % cfg.training.log_every == 0:
                avg_metrics = train_accum.average()
                wandb.log(avg_metrics, step=step)

            wandb.log({"lr": tr.optim.param_groups[0]["lr"]}, step=step)

            pbar.update(1)
            step += 1

    output_dir = pathlib.Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"model.pt"

    torch.save(model.state_dict(), output_file)

    wandb.finish()


if __name__ == "__main__":
    main()
