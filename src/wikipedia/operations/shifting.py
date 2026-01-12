import torch

from src.wikipedia.data_manager import data_utils
from src.wikipedia.operations import base
from src.wikipedia.training import utils


class Shifting(base.Operation):

    @staticmethod
    def variants_one_sentence(sentences, windows, window_id, s):

        lo, hi = windows[window_id - 1]

        base_sent2win = utils.sentence2window(windows)

        window_sentences = sentences[lo : hi + 1]

        rel_source = s - lo

        source_sentence = window_sentences[rel_source]
        base_seq = window_sentences[:rel_source] + window_sentences[rel_source + 1 :]
        texts = []
        spans_list = []
        sent2win_list = []

        W = hi - lo + 1

        for rel_target in range(W):

            if rel_target == rel_source:
                continue

            reordered = (
                base_seq[:rel_target] + [source_sentence] + base_seq[rel_target:]
            )
            modified_full = sentences[:lo] + reordered + sentences[hi + 1 :]
            text, spans = utils.join_text_and_spans(modified_full)

            texts.append(text)
            spans_list.append(spans)
            sent2win_list.append(base_sent2win)

        return texts, spans_list, sent2win_list

    @torch.no_grad()
    def scores(self, sentences):

        n = len(sentences)
        windows = data_utils.compute_windows_even(n)
        base_text, base_spans = utils.join_text_and_spans(sentences)
        base_sent2win = utils.sentence2window(windows)
        ids_base, mask_base, t2w_base = self.tokenize_and_map(
            [base_text], [base_spans], [base_sent2win]
        )
        h_base = self.model.encode_inputs(ids_base, mask_base, labels=None)
        base_window_embeddings = self.model.pool_windows(h_base, t2w_base, mask_base)[0]

        results = []
        for window_id, (lo, hi) in enumerate(windows, start=1):

            base_emd = base_window_embeddings[window_id - 1].unsqueeze(0)
            window_scores = []
            for s in range(lo, hi + 1):

                texts, spans_list, sent2win_list = self.variants_one_sentence(
                    sentences, windows, window_id, s
                )

                ids, mask, t2w = self.tokenize_and_map(texts, spans_list, sent2win_list)
                h = self.model.encode_inputs(ids, mask, labels=None)
                pooled = self.model.pool_windows(h, t2w, mask)

                variant_embds = pooled[:, window_id - 1]

                sims = torch.nn.functional.cosine_similarity(
                    base_emd, variant_embds, dim=-1
                )

                window_scores.append(1.0 - sims.mean().item())

            results.append(window_scores)

        return results
