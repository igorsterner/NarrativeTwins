import asyncio
import json
import random

import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.rocstories.evaluation.metrics import auc, spearman

DATA_PATH = "data/rocstories/split/test.json"
PROMPT_PATH = "data/rocstories/prompts/annotation_style_prompt.txt"

MODEL_NAME = "gpt-4.1"

MODEL_PRICING = {
    "gpt-4.1": (3.0, 12.0),
    "gpt-4.1-mini": (0.80, 3.2),
}
N_REPEATS_PER_STORY = 10


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


class ROCStructuredOutput(BaseModel):
    summary: str
    most_important: int = Field(ge=1, le=5)


def evaluate_saliency_predictions(dataset, counts_by_story):
    metrics_store = {
        "spearman": [],
        "auc": [],
    }
    for sid, item in dataset.items():
        gold_labels = item["most_important"]
        pred_scores = counts_by_story[sid]
        metrics_store["spearman"].append(spearman(pred_scores, gold_labels))
        metrics_store["auc"].append(auc(pred_scores, gold_labels))
    mean_metrics = {k: float(np.mean(v)) for k, v in metrics_store.items()}
    return mean_metrics


async def process_item(client, story_id, template, item, model, total_cost):
    was_ok = True
    prompt = build_prompt(item["story"], template)
    response = await client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        temperature=1.0,
        text_format=ROCStructuredOutput,
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
    return story_id, parsed_dict, total_tokens, cost, was_ok


async def run_once(client, dataset, template, n_repeats=5):
    counts_by_story = {sid: [0, 0, 0, 0, 0] for sid in dataset.keys()}
    costs = []
    tokenss = []
    oks = []
    total_cost = [0.0]
    tasks = []
    for story_id, item in dataset.items():
        for _ in range(n_repeats):
            tasks.append(
                process_item(client, story_id, template, item, MODEL_NAME, total_cost)
            )

    timeout = 10
    for coro in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Processing prompts"
    ):
        try:
            story_id, parsed, tokens, cost, was_ok = await asyncio.wait_for(
                coro, timeout=timeout
            )
            idx = parsed.get("most_important", None)
            counts_by_story[story_id][random.randint(0, 4)] += 1
            costs.append(cost)
            tokenss.append(tokens)
            oks.append(was_ok)
        except asyncio.TimeoutError:
            print(f"Task timed out: {coro}")

    ok_count = sum(oks)
    print(
        f"\nParsed OK: {ok_count}/{len(oks)} | Total cost: ${total_cost[0]:.6f} for {len(costs)} prompts"
    )
    mean_metrics = evaluate_saliency_predictions(dataset, counts_by_story)
    print("\n------- LLM Saliency Metrics (mean over stories) -------")
    print(f"spearman: {mean_metrics['spearman']:.4f}")
    print(f"auc:      {mean_metrics['auc']:.4f}")
    return mean_metrics


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
        metrics = await run_once(
            client, dataset, template, n_repeats=N_REPEATS_PER_STORY
        )
        all_spearman.append(metrics["spearman"])
        all_auc.append(metrics["auc"])
    print("\nFinal mean/std over 3 runs")
    print(
        f"spearman: mean {np.mean(all_spearman):.4f} | std {np.std(all_spearman):.4f}"
    )
    print(f"auc:      mean {np.mean(all_auc):.4f} | std {np.std(all_auc):.4f}")


if __name__ == "__main__":
    asyncio.run(main())
