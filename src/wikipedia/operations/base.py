import torch

from src.wikipedia.training import utils


class Operation:
    def __init__(self, model, tokenizer, device):

        self.tokenizer = tokenizer
        self.device = device

        self.model = model

    def tokenize_and_map(self, texts, spans_list, sent2win_list):

        ids_list = []
        mask_list = []
        t2w_list = []

        for text, spans, s2w in zip(texts, spans_list, sent2win_list):
            enc = self.tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=True,
                truncation=True,
            )
            ids_list.append(torch.tensor(enc["input_ids"]))
            mask_list.append(torch.tensor(enc["attention_mask"]))
            t2w_list.append(
                torch.tensor(utils.token2window(enc["offset_mapping"], spans, s2w))
            )

        pad_id = self.tokenizer.pad_token_id

        input_ids = torch.nn.utils.rnn.pad_sequence(
            ids_list, batch_first=True, padding_value=pad_id
        ).to(self.device)

        attention_mask = torch.nn.utils.rnn.pad_sequence(
            mask_list, batch_first=True, padding_value=0
        ).to(self.device)

        token2window = torch.nn.utils.rnn.pad_sequence(
            t2w_list, batch_first=True, padding_value=-100
        ).to(self.device)

        return input_ids, attention_mask, token2window
