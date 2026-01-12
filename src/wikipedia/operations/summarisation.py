import torch

from src.wikipedia.data_manager import data_utils
from src.wikipedia.operations import base
from src.wikipedia.training import utils


class Summarisation(base.Operation):

    @torch.no_grad()
    def scores(self, sentences):

        n = len(sentences)
        windows = data_utils.compute_windows_even(n)
        full_text, full_spans = utils.join_text_and_spans(sentences)
        full_sent2win = utils.sentence2window(windows)

        ids_full, mask_full, t2w_full = self.tokenize_and_map(
            [full_text], [full_spans], [full_sent2win]
        )

        h_full = self.model.encode_inputs(ids_full, mask_full, labels=None)
        win_emb = self.model.pool_windows(h_full, t2w_full, mask_full)[0]

        sent_texts = []
        sent_spans_list = []
        sent_s2w_list = []
        one_win_bounds = [(0, 0)]
        one_win_s2w = utils.sentence2window(one_win_bounds)

        for s in sentences:
            t, sp = utils.join_text_and_spans([s])
            sent_texts.append(t)
            sent_spans_list.append(sp)
            sent_s2w_list.append(one_win_s2w)

        ids_s, mask_s, t2w_s = self.tokenize_and_map(
            sent_texts, sent_spans_list, sent_s2w_list
        )
        h_s = self.model.encode_inputs(ids_s, mask_s, labels=None)

        sent_embs = self.model.pool_windows(h_s, t2w_s, mask_s)[:, 0]

        results = []
        for wid, (lo, hi) in enumerate(windows, start=1):

            sims = torch.nn.functional.cosine_similarity(
                sent_embs[lo : hi + 1], win_emb[wid - 1], dim=-1
            )
            results.append(sims.tolist())
        return results
