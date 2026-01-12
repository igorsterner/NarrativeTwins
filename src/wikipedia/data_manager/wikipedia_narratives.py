import json
import random
from pathlib import Path

import nltk
import numpy as np
import torch
import tqdm
import transformers

from src.wikipedia.data_manager import stringutils

nltk.download("punkt_tab")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSLATION_SCORES = {
    "fr": 68.1,
    "de": 67.4,
    "it": 61.2,
    "es": 59.1,
}

DATA_DIR = Path("data/wikipedia/tell_me_again_v1/summaries")
OUTPUT_FILE = Path("data/wikipedia/wikipedia_narratives.json")
MIN_EN_SENTENCES = 20

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "princeton-nlp/sup-simcse-bert-base-uncased"
)
model = transformers.AutoModel.from_pretrained(
    "princeton-nlp/sup-simcse-bert-base-uncased"
).to(device)


def list_summary_files(root: Path):
    return sorted(str(p) for p in root.rglob("*.json"))


def sentence_segment(text: str):
    return [s.strip() for s in nltk.tokenize.sent_tokenize(text) if s.strip()]


def load_story_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_en_similarity(similarity_block: dict):

    labels = similarity_block["indexes"]
    matrix = similarity_block["similarities"]

    en_idx = labels.index("en")

    en_row = matrix[en_idx]

    return {lang: sim * 100 for lang, sim in zip(labels, en_row) if lang != "en"}


def collect_stories(files):
    collected = {}
    for idx, fp in tqdm.tqdm(enumerate(files), desc="Reading summaries"):

        data = load_story_json(fp)

        wikidata_id = data["wikidata_id"]
        assert wikidata_id[1:] == Path(fp).stem

        title = data["title"]

        summaries = data["summaries"]

        if "en" not in summaries or "en_translated_summaries" not in data:
            continue
        else:
            en_text = summaries["en"]

        translated = data["en_translated_summaries"]

        en_sentences = sentence_segment(en_text)

        if len(en_sentences) < MIN_EN_SENTENCES:
            continue

        anonymized = data["anonymized"]
        sim_with_en = extract_en_similarity(data["similarity"])

        translated_texts = {lang: block["text"] for lang, block in translated.items()}
        translated_sentences = {
            lang: sentence_segment(text)
            for lang, text in translated_texts.items()
            if lang in anonymized
        }
        anonymized_sentences = {
            lang: sentence_segment(text)
            for lang, text in anonymized.items()
            if lang in translated_sentences
        }

        en_sentences = [stringutils.clean_sentence(s) for s in en_sentences]
        translated_sentences = {
            lang: [stringutils.clean_sentence(s) for s in sents]
            for lang, sents in translated_sentences.items()
        }
        anonymized_sentences = {
            lang: [stringutils.clean_sentence(s) for s in sents]
            for lang, sents in anonymized_sentences.items()
        }

        collected[wikidata_id] = {
            "title": title,
            "en_sentences": en_sentences,
            "translated_sentences": translated_sentences,
            "anonymized_sentences": anonymized_sentences,
            "similarity_with_en": sim_with_en,
        }
        # if idx > 10:
        #     print("BREAKING")
        #     break

    return collected


def compute_en_sentence_stats(collected):
    lengths = [len(v["en_sentences"]) for v in collected.values()]

    mean = sum(lengths) / len(lengths)
    var = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    std = var**0.5
    return mean, std


def floor(n):
    return int(n // 1)


def ceil(n):
    return int(-1 * n // 1 * -1)


def choose_target_language(entry, mean_en, std_en):

    en_len = len(entry["en_sentences"])
    lower = max(en_len - 1 * std_en, 20)
    upper = ceil(en_len + 1 * std_en)

    available = set(entry["translated_sentences"].keys())

    # print(f"Available: {available}")

    filtered = []
    for lang in available:
        sim = entry["similarity_with_en"][lang]

        if sim > TRANSLATION_SCORES[lang]:
            continue

        lang_len = len(entry["translated_sentences"][lang])
        if not (lower <= lang_len <= upper):
            continue

        filtered.append(lang)

    # print(f"Selecting from {filtered}")
    if not filtered:
        return None
    elif len(filtered) == 1:
        return filtered[0]
    else:
        return random.choice(filtered)


def encode_sentences(sentences):

    inputs = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        emb = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    return emb.cpu().numpy()


def cosine_similarity_matrix(a_emb, b_emb):
    a_norm = a_emb / np.linalg.norm(a_emb, axis=1, keepdims=True)
    b_norm = b_emb / np.linalg.norm(b_emb, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


def dtw_path(similarity: np.ndarray):
    n, m = similarity.shape
    cost = -similarity
    dp = np.zeros((n, m))
    dp[0, 0] = cost[0, 0]

    for i in range(1, n):
        dp[i, 0] = cost[i, 0] + dp[i - 1, 0]
    for j in range(1, m):
        dp[0, j] = cost[0, j] + dp[0, j - 1]

    for i in range(1, n):
        for j in range(1, m):
            dp[i, j] = cost[i, j] + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    i, j = n - 1, m - 1
    path = [(i, j)]
    moves = [(-1, -1), (-1, 0), (0, -1)]

    while (i, j) != (0, 0):
        i, j = min(
            ((i + di, j + dj) for di, dj in moves if i + di >= 0 and j + dj >= 0),
            key=lambda x: dp[x],
        )
        path.append([i, j])

    # print(f"path: {path}")
    # print(f"path_r: {path.reverse()}")

    return path[::-1]


def compute_windows_a(n):
    return [(i * n // 5, (i + 1) * n // 5 - 1) for i in range(5)]


def compute_windows_b(align, na, nb, wins_a):
    amap = {}

    for a, b in align:
        amap.setdefault(a, []).append(b)

    wins = []
    for lo, hi in wins_a:
        bs = []
        for ai in range(lo, hi + 1):
            bs.extend(amap[ai])

        if not bs:
            return None
            wins.append((-1, -1))
            continue

        lo_b = max(0, min(bs))
        hi_b = min(nb - 1, max(bs))

        if (hi_b - lo_b) < 3:  # cut window twins less than 3 sentences
            return None
        else:
            wins.append((lo_b, hi_b))

    return wins


def build_output(collected, mean_en, std_en):
    output = {}
    for wikidata_id, entry in tqdm.tqdm(collected.items()):
        story_a = entry["en_sentences"]

        lang = choose_target_language(entry, mean_en, std_en)

        if not lang:
            continue

        story_b = entry["translated_sentences"][lang]
        # story_b_anon = entry["anonymized_sentences"][lang]

        emb_a = encode_sentences(story_a)
        emb_b = encode_sentences(story_b)
        sim_matrix = cosine_similarity_matrix(emb_a, emb_b)
        path = dtw_path(sim_matrix)

        wins_a = compute_windows_a(len(story_a))
        wins_b = compute_windows_b(path, len(story_a), len(story_b), wins_a)

        if wins_b is None:
            continue

        output[wikidata_id] = {
            "title": entry["title"],
            "english_narrative": story_a,
            "translated_narrative_twin": story_b,
            "alignment": path,
        }

    return output


def main():

    files = list_summary_files(DATA_DIR)

    print(f"Found {len(files)} files")

    collected = collect_stories(files)

    mean_en, std_en = compute_en_sentence_stats(collected)

    print(f"Mean English sentence count: {mean_en:.2f}")
    print(f"Std English sentence count: {std_en:.2f}")

    final = build_output(collected, mean_en, std_en)

    print(f"Left with {len(final)} narrative twins")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    random.seed(2025)
    main()
