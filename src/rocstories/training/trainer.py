import torch
import transformers

from src.rocstories.training import utils


class Trainer:
    def __init__(self, cfg, model, tokenizer, dataset, total_steps, device):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device
        self.optim = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        self.sched = torch.optim.lr_scheduler.LinearLR(
            self.optim, start_factor=1.0, end_factor=0.0, total_iters=total_steps
        )
        if cfg.loss.type == "mlm":
            self.mlm_collator = transformers.DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
            )

    def apply_mlm(self, input_ids):
        masked, labels = self.mlm_collator.torch_mask_tokens(
            input_ids.detach().clone().cpu()
        )
        masked = masked.to(input_ids.device)
        labels = labels.to(input_ids.device)
        return masked, labels

    def train_step(self, batch):

        self.model.train()

        if self.cfg.loss.type == "mlm":
            batch["story_input_ids"], batch["story_labels"] = self.apply_mlm(
                batch["story_input_ids"]
            )
            batch["story_negative_input_ids"], batch["story_negative_labels"] = (
                self.apply_mlm(batch["story_negative_input_ids"])
            )
            batch["twin_input_ids"], batch["twin_labels"] = self.apply_mlm(
                batch["twin_input_ids"]
            )
            batch["twin_negative_input_ids"], batch["twin_negative_labels"] = (
                self.apply_mlm(batch["twin_negative_input_ids"])
            )

        out = self.model(batch, loss_type=self.cfg.loss.type)

        if self.cfg.loss.type == "mlm":
            loss = out["loss"]
        else:
            story_emb = out["story_emb"]
            twin_emb = out["twin_emb"]
            story_negative_emb = out["story_negative_emb"]
            twin_negative_emb = out["twin_negative_emb"]

            if self.cfg.training.hard_negatives:
                a = torch.cat([story_emb, story_negative_emb], dim=0)  # shape: [2N, D]
                b = torch.cat([twin_emb, twin_negative_emb], dim=0)  # shape: [2N, D]
                loss = utils.infonce(a, b, self.cfg.model.temperature)
            else:
                loss1 = utils.infonce(story_emb, twin_emb, self.cfg.model.temperature)
                loss2 = utils.infonce(
                    story_negative_emb, twin_negative_emb, self.cfg.model.temperature
                )
                loss = 0.5 * (loss1 + loss2)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.sched.step()

        out_metrics = {
            "loss": loss.item(),
        }

        return out_metrics
