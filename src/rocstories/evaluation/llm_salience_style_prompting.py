import asyncio
import json

import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

DATA_PATH = "data/rocstories/split/test.json"
PROMPT_PATH = "data/rocstories/prompts/llm_salience_style_prompt.txt"

MODEL_NAME = "gpt-4.1"
MODEL_PRICING = {
    "gpt-4.1": (3.0, 12.0),
    "gpt-4.1-mini": (0.80, 3.2),
}


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
                good += 0.5
    return good / total


def load_api_key(file_path):
    with open(file_path, "r") as f:
        return f.read().strip()


def load_rocstories_dataset_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(story, template):
    numbered = [f"{i+1}. {s}" for i, s in enumerate(story)]
    story_str = "\n".join(numbered)
    return template.replace("INSERT_STORY", story_str)


class ROCSalienceOutput(BaseModel):
    sentence_1: float = Field(gt=0.0, lt=1.0)
    sentence_2: float = Field(gt=0.0, lt=1.0)
    sentence_3: float = Field(gt=0.0, lt=1.0)
    sentence_4: float = Field(gt=0.0, lt=1.0)
    sentence_5: float = Field(gt=0.0, lt=1.0)


def evaluate_saliency_predictions(dataset, saliences_by_story):
    metrics_store = {
        "spearman": [],
        "auc": [],
    }

    for sid, item in dataset.items():
        gold_labels = item["most_important"]
        pred_scores = saliences_by_story[sid]
        metrics_store["spearman"].append(spearman(pred_scores, gold_labels))
        metrics_store["auc"].append(auc(pred_scores, gold_labels))

    mean_metrics = {k: float(np.mean(v)) for k, v in metrics_store.items()}
    return mean_metrics


async def process_item(client, story_id, template, item, model, total_cost):
    prompt = build_prompt(item["story"], template)
    response = await client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        temperature=1.0,
        text_format=ROCSalienceOutput,
    )
    parsed = response.output_parsed
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    total_tokens = response.usage.total_tokens
    input_price, output_price = MODEL_PRICING[model]
    cost = (input_tokens * input_price / 1_000_000) + (
        output_tokens * output_price / 1_000_000
    )
    total_cost[0] += cost
    print(
        f"[{item.get('storytitle','')[:28]:28s}] Tokens: {total_tokens:5d} | Cost: ${cost:.6f} | Cumulative: ${total_cost[0]:.6f}"
    )
    parsed_dict = parsed.dict()
    return story_id, parsed_dict, total_tokens, cost


async def run_once(client, dataset, template):
    saliences_by_story = {}
    costs = []
    tokenss = []
    total_cost = [0.0]
    tasks = []
    for story_id, item in dataset.items():
        tasks.append(
            process_item(client, story_id, template, item, MODEL_NAME, total_cost)
        )
    timeout = 60
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="One pass"):
        try:
            story_id, parsed, tokens, cost = await asyncio.wait_for(
                coro, timeout=timeout
            )
            saliences_by_story[story_id] = [
                parsed["sentence_1"],
                parsed["sentence_2"],
                parsed["sentence_3"],
                parsed["sentence_4"],
                parsed["sentence_5"],
            ]
            costs.append(cost)
            tokenss.append(tokens)
        except asyncio.TimeoutError:
            print(f"Task timed out: {coro}")
    print(
        f"\nParsed OK: {len(costs)}/{len(dataset)} | Total cost: ${total_cost[0]:.6f} for {len(costs)} prompts"
    )
    mean_metrics = evaluate_saliency_predictions(dataset, saliences_by_story)
    print("\nLLM Saliency Metrics (mean over stories)")
    print(f"spearman: {mean_metrics['spearman']:.4f}")
    print(f"auc:      {mean_metrics['auc']:.4f}")
    return mean_metrics, saliences_by_story


async def main():
    dataset = load_rocstories_dataset_json(DATA_PATH)
    api_key = load_api_key("data/keys/openai")
    client = AsyncOpenAI(api_key=api_key)
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        template = f.read()
    all_spearman = []
    all_auc = []
    for run_id in range(3):
        print(f"\nRUN {run_id+1}/3")
        metrics, saliences_by_story = await run_once(client, dataset, template)
        all_spearman.append(metrics["spearman"])
        all_auc.append(metrics["auc"])
        save_json = {}
        for sid, item in dataset.items():
            newitem = dict(item)
            newitem["salience_scores"] = saliences_by_story[sid]
            save_json[sid] = newitem
        with open(
            f"data/{MODEL_NAME}-salience-{run_id}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(save_json, f, indent=2, ensure_ascii=False)
    print("\n Final mean/std over 3 runs")
    print(
        f"spearman: mean {np.mean(all_spearman):.4f} | std {np.std(all_spearman):.4f}"
    )
    print(f"auc:      mean {np.mean(all_auc):.4f} | std {np.std(all_auc):.4f}")


if __name__ == "__main__":
    asyncio.run(main())
