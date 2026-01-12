import torch

from src.wikipedia.data_manager import data_utils
from src.wikipedia.operations import base
from src.wikipedia.training import utils


def update_bounds_after_delete(bounds, deleted_idx, new_len):

    assert new_len == bounds[-1][-1], f"{new_len} {bounds}"
    new_len = bounds[-1][-1]

    out = []

    for lo, hi in bounds:

        lo2 = lo - 1 if lo > deleted_idx else lo
        hi2 = hi - 1 if hi >= deleted_idx else hi

        if hi2 < lo2:
            raise Exception
            lo2, hi2 = hi2, lo2

        lo2 = max(0, min(lo2, new_len - 1))
        hi2 = max(0, min(hi2, new_len - 1))
        out.append((lo2, hi2))

    return out


class Deletion(base.Operation):

    @staticmethod
    def variants(sentences):

        n = len(sentences)

        # base story
        windows = data_utils.compute_windows_even(n)
        base_text, base_spans = utils.join_text_and_spans(sentences)
        base_sent2win = utils.sentence2window(windows)
        texts = [base_text]
        spans_list = [base_spans]
        sent2win_list = [base_sent2win]

        # variants
        variant_row = {}
        row = 1
        for wid, (lo, hi) in enumerate(windows, start=1):
            for s in range(lo, hi + 1):
                mod_sentences = sentences[:s] + sentences[s + 1 :]

                mod_text, mod_spans = utils.join_text_and_spans(mod_sentences)
                mod_bounds = update_bounds_after_delete(windows, s, n - 1)
                mod_sent2win = utils.sentence2window(mod_bounds)

                texts.append(mod_text)
                spans_list.append(mod_spans)
                sent2win_list.append(mod_sent2win)

                variant_row[(wid, s)] = row

                row += 1

        return texts, spans_list, sent2win_list, windows, 0, variant_row

    @torch.no_grad()
    def scores(self, sentences):
        texts, spans_list, sent2win_list, windows, base_row, variant_row = (
            self.variants(sentences)
        )

        input_ids, attention_mask, token2window = self.tokenize_and_map(
            texts, spans_list, sent2win_list
        )

        h = self.model.encode_inputs(input_ids, attention_mask)

        pooled = self.model.pool_windows(h, token2window, attention_mask)

        base_windows = pooled[base_row]

        results = []
        for wid, (lo, hi) in enumerate(windows, start=1):

            base_vec = base_windows[wid - 1]

            window_scores = []
            for s in range(lo, hi + 1):
                row = variant_row[(wid, s)]

                mod_vec = pooled[row, wid - 1]
                sim = torch.nn.functional.cosine_similarity(base_vec, mod_vec, dim=-1)

                window_scores.append(1.0 - sim.item())

            results.append(window_scores)

        return results
