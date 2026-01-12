import json

import torch



class TwinPairsDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        items = []
        for sid, rec in raw.items():

            story_text = " ".join(rec["story"])
            twin_text = " ".join(rec["narrative_twin"])

            story_negative_text = " ".join(rec["negative"])

            twin_negative_text = " ".join(rec["negative_narrative_twin"])

            items.append(
                {
                    "id": sid,
                    "story_text": story_text,
                    "twin_text": twin_text,
                    "story_negative_text": story_negative_text,
                    "twin_negative_text": twin_negative_text,
                }
            )

        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            "id": it["id"],
            "story_text": it["story_text"],
            "twin_text": it["twin_text"],
            "story_negative_text": it["story_negative_text"],
            "twin_negative_text": it["twin_negative_text"],
        }


def make_collate_fn(tokenizer, device):
    def collate(batch):
        story_texts = [it["story_text"] for it in batch]
        twin_texts = [it["twin_text"] for it in batch]
        story_negative_texts = [it["story_negative_text"] for it in batch]
        twin_negative_texts = [it["twin_negative_text"] for it in batch]

        enc_story = tokenizer(
            story_texts, return_tensors="pt", padding=True, truncation=True
        )
        enc_twin = tokenizer(
            twin_texts, return_tensors="pt", padding=True, truncation=True
        )
        enc_story_negative = tokenizer(
            story_negative_texts, return_tensors="pt", padding=True, truncation=True
        )
        enc_twin_negative = tokenizer(
            twin_negative_texts, return_tensors="pt", padding=True, truncation=True
        )

        return {
            "story_input_ids": enc_story["input_ids"].to(device),
            "story_attention_mask": enc_story["attention_mask"].to(device),
            "twin_input_ids": enc_twin["input_ids"].to(device),
            "twin_attention_mask": enc_twin["attention_mask"].to(device),
            "story_negative_input_ids": enc_story_negative["input_ids"].to(device),
            "story_negative_attention_mask": enc_story_negative["attention_mask"].to(
                device
            ),
            "twin_negative_input_ids": enc_twin_negative["input_ids"].to(device),
            "twin_negative_attention_mask": enc_twin_negative["attention_mask"].to(
                device
            ),
        }

    return collate


def build_train_dataset(cfg):
    ds = TwinPairsDataset(cfg.paths.training_data)
    return ds


def build_train_dataloader(cfg, tokenizer, dataset, device):
    collate = make_collate_fn(tokenizer, device)
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
    )
    return dl
