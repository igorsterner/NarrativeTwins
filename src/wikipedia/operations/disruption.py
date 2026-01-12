import torch

from src.wikipedia.data_manager import data_utils
from src.wikipedia.operations import base
from src.wikipedia.training import utils


def clip_bounds(bounds, k):

    # bit messy
    out = []
    for lo, hi in bounds:
        out.append((lo, min(hi, k - 1)))

    return out


class Disruption(base.Operation):

    @staticmethod
    def variants(sentences):

        n = len(sentences)

        windows = data_utils.compute_windows_even(n)

        texts = []
        spans_list = []
        sent2win_list = []
        k2row = {}
        row = 0

        for k in range(1, n + 1):
            text_k, spans_k = utils.join_text_and_spans(sentences[:k])

            bounds_k = clip_bounds(windows, k)
            s2w_k = utils.sentence2window(bounds_k)

            texts.append(text_k)
            spans_list.append(spans_k)
            sent2win_list.append(s2w_k)

            k2row[k] = row
            row += 1

        return texts, spans_list, sent2win_list, windows, k2row

    @torch.no_grad()
    def scores(self, sentences):

        texts, spans_list, sent2win_list, windows, k2row = self.variants(sentences)

        input_ids, attention_mask, token2window = self.tokenize_and_map(
            texts, spans_list, sent2win_list
        )

        h = self.model.encode_inputs(input_ids, attention_mask, labels=None)
        pooled = self.model.pool_windows(h, token2window, attention_mask)

        results = []
        for wid, (lo, hi) in enumerate(windows, 1):

            window_scores = []
            for i in range(lo, hi + 1):

                if i == 0:
                    window_scores.append(0.0)
                    continue

                # need to check this logic
                row_inc = k2row[i + 1]
                row_exc = k2row[i]

                v_inc = pooled[row_inc, wid - 1]
                v_exc = pooled[row_exc, wid - 1]

                sim = torch.nn.functional.cosine_similarity(v_inc, v_exc, dim=-1)

                window_scores.append(1.0 - sim.item())

            results.append(window_scores)

        return results
