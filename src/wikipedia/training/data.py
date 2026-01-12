import json
import random
from itertools import groupby

import torch

from src.wikipedia.data_manager import data_utils
from src.wikipedia.training import utils


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer):
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        print(f"Found {len(raw_data)} items")

        self.items = self.preprocess_raw_data(raw_data)

        print(f"Filtered to {len(self.items)} items")
        self.tokenizer = tokenizer

    def preprocess_raw_data(self, raw_data):

        items = []
        for story_id, story_data in raw_data.items():
            a_sents = story_data["english_narrative"]
            b_sents = story_data["translated_narrative_twin"]
            align = story_data["alignment"]

            a_text, a_spans = utils.join_text_and_spans(a_sents)
            b_text, b_spans = utils.join_text_and_spans(b_sents)

            a_wins = data_utils.compute_windows_even(len(a_sents))

            a_sent2win = utils.sentence2window(a_wins)

            b_sent2win = utils.build_b_sent2win(
                align, len(a_sents), len(b_sents), a_sent2win
            )

            # only keep when alignment means every window in b
            # has at least 3 sentences
            cnt_a = [a_sent2win.count(w) for w in range(1, 6)]
            cnt_b = [b_sent2win.count(w) for w in range(1, 6)]
            if min(cnt_a) < 3 or min(cnt_b) < 3:
                continue

            hn_sents, hn_bounds = utils.collapse_sections(
                story_data["negative_narrative"]
            )

            if hn_bounds is None:
                continue
            hn_text, hn_spans = utils.join_text_and_spans(hn_sents)
            hn_sent2win = utils.sentence2window(hn_bounds)

            items.append(
                {
                    "a_text": a_text,
                    "b_text": b_text,
                    "a_spans": a_spans,
                    "b_spans": b_spans,
                    "a_sent2win": a_sent2win,
                    "b_sent2win": b_sent2win,
                    "hn_text": hn_text,
                    "hn_spans": hn_spans,
                    "hn_sent2win": hn_sent2win,
                    "id": story_id,
                }
            )

        return items

    def __getitem__(self, idx):
        it = self.items[idx]

        a_text = it["a_text"]
        b_text = it["b_text"]
        a_spans = it["a_spans"]
        b_spans = it["b_spans"]
        a_sent2win = it["a_sent2win"]
        b_sent2win = it["b_sent2win"]
        hn_text = it["hn_text"]
        hn_spans = it["hn_spans"]
        hn_sent2win = it["hn_sent2win"]

        a_enc = self.tokenizer(
            a_text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
        )

        b_enc = self.tokenizer(
            b_text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
        )
        hn_enc = self.tokenizer(
            hn_text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
        )

        a_off = a_enc["offset_mapping"]
        b_off = b_enc["offset_mapping"]
        hn_off = hn_enc["offset_mapping"]

        a_token2window = utils.token2window(a_off, a_spans, a_sent2win)
        b_token2window = utils.token2window(b_off, b_spans, b_sent2win)
        hn_token2window = utils.token2window(hn_off, hn_spans, hn_sent2win)

        return {
            "a_input_ids": a_enc["input_ids"],
            "a_attention_mask": a_enc["attention_mask"],
            "a_token2window": a_token2window,
            "b_input_ids": b_enc["input_ids"],
            "b_attention_mask": b_enc["attention_mask"],
            "b_token2window": b_token2window,
            "hn_input_ids": hn_enc["input_ids"],
            "hn_attention_mask": hn_enc["attention_mask"],
            "hn_token2window": hn_token2window,
        }

    def __len__(self):
        return len(self.items)


def make_collate_fn(tokenizer, device):
    pad_id = tokenizer.pad_token_id

    def collate(batch):
        a_ids = [torch.tensor(it["a_input_ids"]) for it in batch]
        a_am = [torch.tensor(it["a_attention_mask"]) for it in batch]
        a_t2w = [torch.tensor(it["a_token2window"]) for it in batch]
        b_ids = [torch.tensor(it["b_input_ids"]) for it in batch]
        b_am = [torch.tensor(it["b_attention_mask"]) for it in batch]
        b_t2w = [torch.tensor(it["b_token2window"]) for it in batch]
        hn_ids = [torch.tensor(it["hn_input_ids"]) for it in batch]
        hn_am = [torch.tensor(it["hn_attention_mask"]) for it in batch]
        hn_t2w = [torch.tensor(it["hn_token2window"]) for it in batch]

        def check_constraints(t2w_list, am_list):
            for t2w, am in zip(t2w_list, am_list):
                valid = t2w[am.bool()]  # ignore padding

                # ensure all 5 windows exist
                for w in range(1, 6):
                    cnt = (valid == w).sum().item()
                    assert cnt >= 10, f"Window {w} has only {cnt} tokens"

                w_ids = valid.tolist()

                seq = [k for k, _ in groupby(w_ids) if k != -100]
                expected = [1, 2, 3, 4, 5]
                assert (
                    seq == expected
                ), f"Non-contiguous windows order: got {seq}, expected {expected}"

        check_constraints(a_t2w, a_am)
        check_constraints(b_t2w, b_am)
        check_constraints(hn_t2w, hn_am)

        a_ids = torch.nn.utils.rnn.pad_sequence(
            a_ids, batch_first=True, padding_value=pad_id
        ).to(device)
        a_am = torch.nn.utils.rnn.pad_sequence(
            a_am, batch_first=True, padding_value=0
        ).to(device)
        a_t2w = torch.nn.utils.rnn.pad_sequence(
            a_t2w, batch_first=True, padding_value=-100
        ).to(device)
        b_ids = torch.nn.utils.rnn.pad_sequence(
            b_ids, batch_first=True, padding_value=pad_id
        ).to(device)
        b_am = torch.nn.utils.rnn.pad_sequence(
            b_am, batch_first=True, padding_value=0
        ).to(device)
        b_t2w = torch.nn.utils.rnn.pad_sequence(
            b_t2w, batch_first=True, padding_value=-100
        ).to(device)
        hn_ids = torch.nn.utils.rnn.pad_sequence(
            hn_ids, batch_first=True, padding_value=pad_id
        ).to(device)
        hn_am = torch.nn.utils.rnn.pad_sequence(
            hn_am, batch_first=True, padding_value=0
        ).to(device)
        hn_t2w = torch.nn.utils.rnn.pad_sequence(
            hn_t2w, batch_first=True, padding_value=-100
        ).to(device)

        return {
            "a_input_ids": a_ids,
            "a_attention_mask": a_am,
            "a_token2window": a_t2w,
            "b_input_ids": b_ids,
            "b_attention_mask": b_am,
            "b_token2window": b_t2w,
            "hn_input_ids": hn_ids,
            "hn_attention_mask": hn_am,
            "hn_token2window": hn_t2w,
        }

    return collate


def split_dataset(ds, split_ratio, seed):

    idx = list(range(len(ds)))

    rng = random.Random(seed)
    rng.shuffle(idx)

    split = int(split_ratio * len(idx))
    train_idx = idx[:split]
    val_idx = idx[split:]

    ds_train = torch.utils.data.Subset(ds, train_idx)
    ds_val = torch.utils.data.Subset(ds, val_idx)

    return ds_train, ds_val


def build_dataloader(cfg, tokenizer, device):
    ds = PairDataset(cfg.paths.dataset_json, tokenizer)

    collate = make_collate_fn(tokenizer, device)

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
    )

    return dl
