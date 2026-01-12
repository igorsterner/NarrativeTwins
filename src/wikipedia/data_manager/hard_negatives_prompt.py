import asyncio
import json
from pathlib import Path

import nltk
import openai
import pydantic
import tqdm

from src.wikipedia.data_manager import data_utils
from src.wikipedia.training import utils


class StorySections(pydantic.BaseModel):
    section_1: str
    section_2: str
    section_3: str
    section_4: str
    section_5: str


def load_data(file_paths):
    all_data = {}
    for file_path in file_paths:
        with open(file_path, "r") as file:
            data = json.load(file)
            all_data.update(data)
    return all_data


def load_prompt(file_path):
    with open(file_path, "r") as file:
        return file.read()


def load_api_key(file_path):
    with open(file_path, "r") as file:
        return file.read().strip()


def load_cache(path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_cache(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=4)
    tmp.replace(path)


def filter_dataset(raw_data):

    filtered = {}
    for story_id, story_data in raw_data.items():
        a_sents = story_data["english_narrative"]
        b_sents = story_data["translated_narrative_twin"]
        align = story_data["alignment"]

        a_wins = data_utils.compute_windows_even(len(a_sents))
        a_sent2win = utils.sentence2window(a_wins)
        b_sent2win = utils.build_b_sent2win(
            align, len(a_sents), len(b_sents), a_sent2win
        )

        cnt_a = [a_sent2win.count(w) for w in range(1, 6)]
        cnt_b = [b_sent2win.count(w) for w in range(1, 6)]
        if min(cnt_a) < 3 or min(cnt_b) < 3:
            continue

        filtered[story_id] = story_data

    return filtered


def split_story_into_sections(sentences):

    windows = data_utils.compute_windows_even(len(sentences))
    sections = []

    for i, (lo, hi) in enumerate(windows, 1):
        section_text = " ".join(sentences[lo : hi + 1])
        sections.append(f"section_{i}:\n\n{section_text.strip()}")

    return "\n\n".join(sections)


async def process_step(
    client, model_name, story_id, story_details, continuation_prompt, pricing
):

    story_text = split_story_into_sections(story_details["english_narrative"])
    prompt = continuation_prompt.replace("INSERT_STORY", story_text)

    response = await client.responses.parse(
        model=model_name,
        input=[{"role": "user", "content": prompt}],
        text_format=StorySections,
    )

    price = (response.usage.input_tokens * pricing[0] / 1000000) + (
        response.usage.output_tokens * pricing[1] / 1000000
    )
    tokens_used = response.usage.total_tokens

    parsed = response.output_parsed
    sections = {
        "section_1": nltk.sent_tokenize(parsed.section_1),
        "section_2": nltk.sent_tokenize(parsed.section_2),
        "section_3": nltk.sent_tokenize(parsed.section_3),
        "section_4": nltk.sent_tokenize(parsed.section_4),
        "section_5": nltk.sent_tokenize(parsed.section_5),
    }

    return story_id, model_name, prompt, sections, tokens_used, price


async def run_experiment(test_data, continuation_prompt, cache_path):

    client = openai.AsyncOpenAI(api_key=load_api_key("data/keys/openai"))
    cache = load_cache(cache_path)
    predictions = dict(cache)
    model_names = ["gpt-5-mini"]

    ALL_TOKENS_USED = 0
    TOTAL_PRICE = 0
    model_pricing = {
        "gpt-5-mini": (0.25, 2.00),
    }

    filtered_data = filter_dataset(test_data)

    print(f"Filtered to {len(filtered_data)} items")

    story_items = list(filtered_data.items())
    N = len(story_items)
    batch_size = 3000
    batch_idx = 1

    while batch_size * (batch_idx - 1) < N:
        start = batch_size * (batch_idx - 1)
        end = min(batch_size * batch_idx, N)
        batch = story_items[start:end]

        tasks = []
        for model_name in model_names:
            for story_id, story_details in batch:
                if story_id in predictions:
                    continue

                tasks.append(
                    process_step(
                        client,
                        model_name,
                        story_id,
                        story_details,
                        continuation_prompt,
                        model_pricing[model_name],
                    )
                )
        for task in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks)):

            try:
                story_id, model_name, prompt, sections, tokens_used, price = (
                    await asyncio.wait_for(task, timeout=300)
                )
                ALL_TOKENS_USED += tokens_used
                TOTAL_PRICE += price
                print(f"Tokens used: {ALL_TOKENS_USED}")
                print(f"Total Price: ${TOTAL_PRICE:.4f}")
                filtered_data[story_id]["negative_narrative"] = sections
                predictions[story_id] = filtered_data[story_id]

            except asyncio.TimeoutError:
                print(f"Timeout error on story {story_id}")
            except Exception as e:
                print(f"Error on story {story_id}: {e}")

            print(
                f"Processed {len([k for k in predictions if k in filtered_data])} out of {N}"
            )

        batch_idx += 1
        save_cache(cache_path, predictions)

    return predictions


def save_predictions(predictions, file_path):
    with open(file_path, "w") as file:
        json.dump(predictions, file, indent=4)


async def main():

    utils.set_seed(127)

    hard_negative_prompt_file = Path("data/wikipedia/prompts/hard_negative_prompt.txt")

    data_file = "data/wikipedia/wikipedia_narratives.json"

    all_data = load_data([data_file])

    hard_negative_prompt = load_prompt(hard_negative_prompt_file)

    cache_path = Path("data/wikipedia/cache.json")

    predictions_prompt = await run_experiment(
        all_data, hard_negative_prompt, cache_path
    )

    out_file = "data/train.json.json"
    save_predictions(predictions_prompt, out_file)


if __name__ == "__main__":
    asyncio.run(main())
