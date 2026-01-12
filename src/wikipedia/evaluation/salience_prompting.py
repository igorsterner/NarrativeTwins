import asyncio
import json

import numpy as np
from openai import AsyncOpenAI
from pydantic import Field, create_model
from tqdm import tqdm

DATA_PATH = "data/wikipedia/tripod.json"
PROMPT_PATH = "data/wikipedia/prompts/salience_prompt.txt"
MODEL_NAME = "gpt-5"

MODEL_PRICING = {
    "gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-4.1": (3.0, 12.0),
    "gpt-4.1-mini": (0.80, 3.2),
}


def build_salience_output_model(num_sents):
    fields = {
        f"sentence_{i+1}": (float, Field(gt=0.0, lt=1.0)) for i in range(num_sents)
    }
    return create_model("ROCSalienceOutput", **fields)


def auc(scores, idx):
    n = len(scores)
    order = np.argsort(scores)[::-1]
    r = np.where(order == idx)[0][0] + 1
    return (n - r) / (n - 1)


def load_api_key(file_path):
    with open(file_path, "r") as f:
        return f.read().strip()


def load_rocstories_dataset_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(story, template):
    numbered = [f"{i + 1}. {s}" for i, s in enumerate(story)]
    story_str = "\n".join(numbered)
    return template.replace("INSERT_STORY", story_str).replace(
        "NUM_SENTS", str(len(story))
    )


def compute_windows_even(n):
    return [(i * n // 5, (i + 1) * n // 5 - 1) for i in range(5)]


def evaluate_saliency_predictions(dataset, saliences_by_story):
    results = {"llm": {i: [] for i in range(5)}}
    for sid, item in tqdm(dataset.items()):
        sents = item["sentences"]
        n = len(sents)
        wins = compute_windows_even(n)
        gold = [item["tp1"], item["tp2"], item["tp3"], item["tp4"], item["tp5"]]
        pred_scores = saliences_by_story[sid]

        for w_id, (lo, hi) in enumerate(wins, start=1):
            tp = gold[w_id - 1]
            if not (lo <= tp <= hi):
                continue
            rel = tp - lo
            scores_w = pred_scores[lo : hi + 1]
            auc_val = auc(scores_w, rel)
            results["llm"][w_id - 1].append(auc_val)

    mean_results = {}
    auc_values = []
    for i in range(5):
        vs = results["llm"][i]
        mean_auc = float(np.mean(vs)) if vs else float("nan")
        mean_results[f"auc_{i + 1}/llm"] = mean_auc
        auc_values.append(mean_auc)

    mean_results[f"auc_avg/llm"] = float(np.nanmean(auc_values))

    return mean_results


async def process_item(client, story_id, template, item, model, total_cost):

    prompt = build_prompt(item["sentences"], template)
    num_sents = len(item["sentences"])

    SalienceOutputModel = build_salience_output_model(num_sents)

    response = await client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        temperature=1.0,
        text_format=SalienceOutputModel,
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
        f"[{item.get('storytitle', '')[:28]:28s}] Tokens: {total_tokens:5d} | Cost: ${cost:.6f} | Cumulative: ${total_cost[0]:.6f}"
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

    timeout = 300
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="One pass"):
        try:
            story_id, parsed, tokens, cost = await asyncio.wait_for(
                coro, timeout=timeout
            )
            saliences_by_story[story_id] = [
                parsed[f"sentence_{i + 1}"] for i in range(len(parsed))
            ]
            costs.append(cost)
            tokenss.append(tokens)
        except asyncio.TimeoutError:
            print(f"Task timed out: {coro}")

    print(
        f"\nParsed OK: {len(costs)}/{len(dataset)} | Total cost: ${total_cost[0]:.6f} for {len(costs)} prompts"
    )

    mean_metrics = evaluate_saliency_predictions(dataset, saliences_by_story)

    return mean_metrics, saliences_by_story


async def main():
    dataset = load_rocstories_dataset_json(DATA_PATH)
    api_key = load_api_key("data/keys/openai")
    client = AsyncOpenAI(api_key=api_key)
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        template = f.read()

    all_metrics = []
    for run_id in range(3):
        metrics, saliences_by_story = await run_once(client, dataset, template)
        all_metrics.append(metrics)
        save_json = {}
        for sid, item in dataset.items():
            newitem = dict(item)
            newitem["salience_scores"] = saliences_by_story[sid]
            save_json[sid] = newitem

        with open(
            f"data/wikipedia/results/tripod-{MODEL_NAME}-salience-{run_id}.json", "w"
        ) as f:
            json.dump(save_json, f)

    keys = list(all_metrics[0].keys())
    arr = {k: np.array([run[k] for run in all_metrics]) for k in keys}

    for k in keys:
        vals = arr[k]
        print(f"{k:20s}: mean {np.mean(vals):.4f} | std {np.std(vals):.4f}")


if __name__ == "__main__":
    asyncio.run(main())
