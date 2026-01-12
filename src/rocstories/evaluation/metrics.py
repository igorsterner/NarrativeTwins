

def counts_from_labels(labels):
    counts = [0, 0, 0, 0, 0]
    for x in labels:
        i = int(x) - 1
        counts[i] += 1
    return counts


def rank_desc(values):
    n = len(values)
    pairs = [(-values[i], i) for i in range(n)]
    pairs.sort()
    ranks = [0.0] * n
    pos = 1
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        start = pos
        end = pos + (j - i)
        mid = (start + end) / 2.0
        for k in range(i, j + 1):
            idx = pairs[k][1]
            ranks[idx] = mid
        pos = end + 1
        i = j + 1
    return ranks


def pearson_corr(a, b):

    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    num = 0.0
    da = 0.0
    db = 0.0

    for i in range(n):
        xa = a[i] - ma
        xb = b[i] - mb
        num += xa * xb
        da += xa * xa
        db += xb * xb

    den = (da**0.5) * (db**0.5)

    if den == 0:
        return None
    return num / den


def spearman(scores, labels):

    gold_counts = counts_from_labels(labels)

    r_gold = rank_desc(gold_counts)
    r_pred = rank_desc(scores)

    return pearson_corr(r_gold, r_pred)


def auc(scores, labels):

    rel = counts_from_labels(labels)

    pos = [i for i in range(5) if rel[i] > 0]
    neg = [i for i in range(5) if rel[i] == 0]

    total = len(pos) * len(neg)

    good = 0.0
    for i in pos:
        for j in neg:

            if scores[i] > scores[j]:
                good += 1.0
            elif scores[i] == scores[j]:
                # raise Exception("Wow, somehow saliency was identical")
                good += 0.5

    return good / total
