import torch


class Encoder:
    def __init__(self, model, tokenizer, device):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def encode(self, texts):
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        emb, _ = self.model.encode(enc["input_ids"], enc["attention_mask"])
        return emb
