import random

import torch
import transformers


class ROCStoryBERT(torch.nn.Module):
    def __init__(self, model_name, pooling):
        super().__init__()
        self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(model_name)
        self.pooling = pooling

    def encode(self, input_ids, attention_mask, labels=None):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            labels=labels,
        )
        h = out.hidden_states[-1]

        if self.pooling == "mean":
            h = (h * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(
                dim=1, keepdim=True
            )
        elif self.pooling == "cls":
            h = h[:, 0, :]
        else:
            raise NotImplementedError

        return h, out.loss if hasattr(out, "loss") else None

    def forward(self, batch, loss_type):

        if loss_type == "nt":
            story_emb, _ = self.encode(
                batch["story_input_ids"], batch["story_attention_mask"]
            )
            story_negative_emb, _ = self.encode(
                batch["story_negative_input_ids"],
                batch["story_negative_attention_mask"],
            )
            twin_emb, _ = self.encode(
                batch["twin_input_ids"], batch["twin_attention_mask"]
            )
            twin_negative_emb, _ = self.encode(
                batch["twin_negative_input_ids"], batch["twin_negative_attention_mask"]
            )

        elif loss_type == "simcse":
            if random.random() < 0.5:
                story_emb, _ = self.encode(
                    batch["story_input_ids"], batch["story_attention_mask"]
                )
                story_negative_emb, _ = self.encode(
                    batch["story_negative_input_ids"],
                    batch["story_negative_attention_mask"],
                )

                twin_emb, _ = self.encode(
                    batch["story_input_ids"], batch["story_attention_mask"]
                )
                twin_negative_emb, _ = self.encode(
                    batch["story_negative_input_ids"],
                    batch["story_negative_attention_mask"],
                )

            else:
                story_emb, _ = self.encode(
                    batch["twin_input_ids"], batch["twin_attention_mask"]
                )
                story_negative_emb, _ = self.encode(
                    batch["twin_negative_input_ids"],
                    batch["twin_negative_attention_mask"],
                )

                twin_emb, _ = self.encode(
                    batch["twin_input_ids"], batch["twin_attention_mask"]
                )
                twin_negative_emb, _ = self.encode(
                    batch["twin_negative_input_ids"],
                    batch["twin_negative_attention_mask"],
                )

        elif loss_type == "mlm":

            story_emb, story_loss = self.encode(
                batch["story_input_ids"],
                batch["story_attention_mask"],
                batch["story_labels"],
            )
            story_negative_emb, story_negative_loss = self.encode(
                batch["story_negative_input_ids"],
                batch["story_negative_attention_mask"],
                batch["story_negative_labels"],
            )
            twin_emb, twin_loss = self.encode(
                batch["twin_input_ids"],
                batch["twin_attention_mask"],
                batch["twin_labels"],
            )
            _, twin_negative_loss = self.encode(
                batch["twin_negative_input_ids"],
                batch["twin_negative_attention_mask"],
                batch["twin_negative_labels"],
            )

            total_loss = (
                story_loss + story_negative_loss + twin_loss + twin_negative_loss
            )
            return {"loss": total_loss}

        return {
            "story_emb": story_emb,
            "twin_emb": twin_emb,
            "story_negative_emb": story_negative_emb,
            "twin_negative_emb": twin_negative_emb,
        }
