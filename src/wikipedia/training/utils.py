import collections
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def join_text_and_spans(sentences):

    text = " ".join(sentences)

    spans = []
    cur = 0
    for s in sentences:
        spans.append((cur, cur + len(s) + 1))
        cur += len(s) + 1

    return text, spans


def sentence2window(win_bounds):

    out = [-100] * (win_bounds[-1][-1] + 1)
    for w_id, (lo, hi) in enumerate(win_bounds, start=1):
        for si in range(lo, hi + 1):
            out[si] = w_id
    return out


def build_char2win(sent_spans, sent2win):

    max_pos = max(b for _, b in sent_spans)
    assert max_pos == sent_spans[-1][-1]

    max_pos = sent_spans[-1][-1]

    char2win = [-100] * (max_pos + 1)
    for si, (a, b) in enumerate(sent_spans):
        w = sent2win[si]
        for pos in range(a, b):
            char2win[pos] = w

    return char2win


def token2window(offsets, sent_spans, sent2win):

    char2win = build_char2win(sent_spans, sent2win)

    out = []
    for a, b in offsets:

        wins = [char2win[pos] for pos in range(a, b)]

        if not wins:
            out.append(-100)
            continue

        cnt = collections.Counter(wins)
        win = cnt.most_common(1)[0][0]

        out.append(win)

    return out


def build_b_sent2win(align, na, nb, a_sent2win):

    cand = {j: [] for j in range(nb)}
    for a_idx, b_idx in align:
        w = a_sent2win[a_idx]
        cand[b_idx].append(w)

    b_sent2win = [0] * nb
    for j in range(nb):
        c = cand[j]

        assert len(c) > 0

        if len(c) == 1:
            b_sent2win[j] = c[0]
        else:
            b_sent2win[j] = random.choice(c)

    return b_sent2win


def collapse_sections(sections):

    sents = []
    bounds = []
    start = 0

    for i in range(1, 6):

        sec = sections[f"section_{i}"]

        if not len(sec) >= 3:
            return None, None

        sents.extend(sec)

        end = start + len(sec) - 1

        bounds.append((start, end))

        start = end + 1

    return sents, bounds


def retrieval(a, b):
    batch_size = a.size(0)
    feats = torch.cat([a.float(), b.float()], dim=0)
    feats = torch.nn.functional.normalize(feats, dim=1)
    sim = feats @ feats.t()
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=feats.device)
    sim = sim.masked_fill(mask, float("-inf"))
    preds = sim[:batch_size].argmax(dim=1)
    labels = torch.arange(batch_size, device=feats.device) + batch_size
    acc = (preds == labels).float().mean()
    return acc


def infonce(a, b, temperature, no_cross_chapters=False, num_chapters=5):

    batch_chapter = a.size(0)
    assert (
        batch_chapter % num_chapters == 0
    ), f"Input size {batch_chapter} is not divisible by num_chapters {num_chapters}."
    batch_size = batch_chapter // num_chapters

    feats = torch.cat([a.float(), b.float()], dim=0)
    feats = torch.nn.functional.normalize(feats, dim=1)

    sim = feats @ feats.t() / float(temperature)
    mask = torch.eye(2 * batch_chapter, dtype=torch.bool, device=feats.device)

    if no_cross_chapters:
        book_ids = torch.arange(batch_chapter, device=feats.device) // num_chapters
        all_book_ids = torch.cat([book_ids, book_ids], dim=0)  # [2*batch_chapter]
        same_book = all_book_ids.unsqueeze(0) == all_book_ids.unsqueeze(1)

        pos_idx = torch.arange(batch_chapter, device=feats.device)
        same_book[pos_idx, pos_idx + batch_chapter] = False
        same_book[pos_idx + batch_chapter, pos_idx] = False

        total_mask = mask | same_book
        sim = sim.masked_fill(total_mask, float("-inf"))
    else:
        sim = sim.masked_fill(mask, float("-inf"))

    labels = torch.arange(batch_chapter, device=feats.device)
    labels = torch.cat([labels + batch_chapter, labels], dim=0)

    return torch.nn.functional.cross_entropy(sim, labels)


def hard_infonce(a, b, c, temperature):

    B = a.size(0)

    a_n = torch.nn.functional.normalize(a.float(), dim=1)
    b_n = torch.nn.functional.normalize(b.float(), dim=1)
    c_n = torch.nn.functional.normalize(c.float(), dim=1)

    Q = torch.cat([a_n, b_n], dim=0)
    K = torch.cat([a_n, b_n, c_n], dim=0)

    logits = (Q @ K.t()) / float(temperature)

    mask = torch.zeros((2 * B, 3 * B), dtype=torch.bool, device=a.device)
    mask[:B, :B] = torch.eye(B, dtype=torch.bool, device=a.device)
    mask[B:, B : B + B] = torch.eye(B, dtype=torch.bool, device=a.device)
    logits = logits.masked_fill(mask, float("-inf"))

    labels = torch.arange(B, device=a.device)
    labels = torch.cat([labels + B, labels], dim=0)

    return torch.nn.functional.cross_entropy(logits, labels)


class MetricsAccumulator:
    def __init__(self, prefix=""):
        self.prefix = prefix
        self.sums = {}
        self.count = 0

    def update(self, metrics: dict):
        for k, v in metrics.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)

        self.count += 1

    def average(self):

        avg = {f"{self.prefix}{k}": v / self.count for k, v in self.sums.items()}

        self.sums = {}
        self.count = 0

        return avg
