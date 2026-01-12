import random

import torch
import transformers


class SiameseModernBERT(torch.nn.Module):
    def __init__(self, model_id, pooling, loss_type):
        super().__init__()

        if loss_type == "mlm":
            self.mlm_backbone = transformers.AutoModelForMaskedLM.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                attention_dropout=0.1,
                attn_implementation="flash_attention_2",
            )

        self.backbone = transformers.AutoModel.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            attention_dropout=0.1,
            attn_implementation="flash_attention_2",
        )
        self.pooling = pooling

    def encode_inputs(self, input_ids, attention_mask, labels=None):

        if labels is not None:
            out1 = self.mlm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                labels=labels,
            )

            with torch.no_grad():
                out2 = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )

            return out2.last_hidden_state, out1.loss
        else:

            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

            return out.last_hidden_state

    def _mean_pool_windows(self, h, token2window, attention_mask):
        win_ids = torch.arange(1, 6, device=h.device).view(1, 1, 5)
        window_mask = (
            token2window.unsqueeze(-1) == win_ids
        ) & attention_mask.bool().unsqueeze(-1)

        return (h.unsqueeze(2) * window_mask.unsqueeze(-1)).sum(
            dim=1
        ) / window_mask.float().sum(dim=1).clamp(min=1.0).unsqueeze(-1)

    def pool_windows(self, h, token2window, attention_mask):
        if self.pooling == "mean":
            return self._mean_pool_windows(h, token2window, attention_mask)
        else:
            raise NotImplementedError

    def forward(self, batch, loss_type):

        a_input_ids = batch["a_input_ids"]
        a_attention_mask = batch["a_attention_mask"]
        a_token2window = batch["a_token2window"]
        b_input_ids = batch["b_input_ids"]
        b_attention_mask = batch["b_attention_mask"]
        b_token2window = batch["b_token2window"]
        hn_input_ids = batch["hn_input_ids"]
        hn_attention_mask = batch["hn_attention_mask"]
        hn_token2window = batch["hn_token2window"]

        if loss_type == "mlm":

            ha, loss1 = self.encode_inputs(
                a_input_ids, a_attention_mask, batch["a_labels"]
            )
            hb, loss2 = self.encode_inputs(
                b_input_ids, b_attention_mask, batch["b_labels"]
            )

            return {
                "loss": loss1 + loss2,
            }

        hhn = self.encode_inputs(hn_input_ids, hn_attention_mask)
        whn = self.pool_windows(hhn, hn_token2window, hn_attention_mask)
        win_hn = whn.flatten(0, 1)

        if loss_type == "nt":

            ha = self.encode_inputs(a_input_ids, a_attention_mask)
            wa = self.pool_windows(ha, a_token2window, a_attention_mask)
            win_a = wa.flatten(0, 1)

            hb = self.encode_inputs(b_input_ids, b_attention_mask)
            wb = self.pool_windows(hb, b_token2window, b_attention_mask)
            win_b = wb.flatten(0, 1)

        elif loss_type == "simcse":

            if random.random() < 0.5:

                ha = self.encode_inputs(a_input_ids, a_attention_mask)
                wa = self.pool_windows(ha, a_token2window, a_attention_mask)
                win_a = wa.flatten(0, 1)

                hb = self.encode_inputs(a_input_ids, a_attention_mask)
                wb = self.pool_windows(ha, a_token2window, a_attention_mask)
                win_b = wb.flatten(0, 1)

            else:

                ha = self.encode_inputs(b_input_ids, b_attention_mask)
                wa = self.pool_windows(ha, b_token2window, b_attention_mask)
                win_a = wa.flatten(0, 1)

                hb = self.encode_inputs(b_input_ids, b_attention_mask)
                wb = self.pool_windows(hb, b_token2window, b_attention_mask)
                win_b = wb.flatten(0, 1)

        else:
            raise NotImplementedError

        return {
            "win_a": win_a,
            "win_b": win_b,
            "win_hn": win_hn,
        }
