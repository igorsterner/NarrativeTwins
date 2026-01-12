import torch
import transformers

from src.wikipedia.training import utils


class Trainer:
    def __init__(self, cfg, model, tokenizer, dataloader, total_steps, device):

        self.cfg = cfg
        self.model = model
        self.tok = tokenizer
        self.dl = dataloader
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
                tokenizer=self.tok, mlm=True, mlm_probability=0.15
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
            batch["a_input_ids"], batch["a_labels"] = self.apply_mlm(
                batch["a_input_ids"]
            )
            batch["b_input_ids"], batch["b_labels"] = self.apply_mlm(
                batch["b_input_ids"]
            )
            batch["hn_input_ids"], batch["hn_labels"] = self.apply_mlm(
                batch["hn_input_ids"]
            )

        out = self.model(batch, self.cfg.loss.type)

        if self.cfg.loss.type == "mlm":
            loss = out["loss"]
        else:
            if self.cfg.training.hard_negatives == "gpt":
                loss = utils.hard_infonce(
                    out["win_a"],
                    out["win_b"],
                    out["win_hn"],
                    self.cfg.model.temperature,
                )
            elif self.cfg.training.hard_negatives == "windows":
                loss = utils.infonce(
                    out["win_a"], out["win_b"], self.cfg.model.temperature
                )
            elif self.cfg.training.hard_negatives == "no_windows":
                loss = utils.infonce(
                    out["win_a"],
                    out["win_b"],
                    self.cfg.model.temperature,
                    no_cross_chapters=True,
                )
            else:
                raise Exception

        self.optim.zero_grad()

        loss.backward()

        self.optim.step()
        self.sched.step()

        return {"loss": loss.item()}
