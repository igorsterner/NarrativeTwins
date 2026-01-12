import asyncio
import json
from pathlib import Path

import nltk
from openai import AsyncOpenAI
from tqdm import tqdm
from unidecode import unidecode


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


async def process_item(
    client, model_name, story_id, story_details, prepared_prompt, pricing, max_retries=3
):
    retries = 0
    while retries < max_retries:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prepared_prompt}],
                max_completion_tokens=500,
            )
            if response.choices[0].finish_reason != "stop":
                retries += 1
                continue

            response_text = response.choices[0].message.content
            response_text = unidecode(response_text)

            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            price = (prompt_tokens * pricing[0] / 1000000) + (
                completion_tokens * pricing[1] / 1000000
            )

            tokens_used = response.usage.total_tokens
            return (
                story_id,
                model_name,
                prepared_prompt,
                response_text,
                tokens_used,
                price,
            )
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
    print(f"Failed to process prompt after {max_retries} retries: {prepared_prompt}")
    return story_id, model_name, prepared_prompt, "", 0, 0


async def process_two_step(
    client,
    model_name,
    story_id,
    story_details,
    negative_prompt,
    pricing,
    max_retries=3,
):
    story_text = " ".join(story_details["story"])

    prompt = negative_prompt.replace("INSERT_STORY", story_text)

    result = await process_item(
        client, model_name, story_id, story_details, prompt, pricing, max_retries
    )

    continued_story = result[3]

    tokens_used = result[4]
    price = result[5]

    negative_sentences = nltk.sent_tokenize(continued_story)

    return (
        story_id,
        model_name,
        prompt,
        {
            "negative": continued_story,
            "continued_story_sentences": negative_sentences,
        },
        tokens_used,
        price,
    )


async def run_experiment(test_data, negative_prompt, cache_path):
    client = AsyncOpenAI(api_key=load_api_key("data/keys/openai"))

    cache = load_cache(cache_path)

    predictions = dict(cache)

    model_names = ["gpt-4.1-mini"]

    batch_size = 1000
    batch_idx = 1
    ALL_TOKENS_USED = 0
    TOTAL_PRICE = 0

    model_pricing = {
        "gpt-4.1-mini": (0.80, 3.20),
    }

    while batch_size * batch_idx <= 20000:
        print(f"Treating {batch_size*batch_idx-batch_size}-{batch_size*batch_idx}")
        start = batch_size * (batch_idx - 1)
        end = batch_size * batch_idx
        tasks = []

        for model_name in model_names:
            for story_id, story_details in list(test_data.items())[start:end]:
                if story_id in predictions:
                    continue
                task = process_two_step(
                    client,
                    model_name,
                    story_id,
                    story_details,
                    negative_prompt,
                    model_pricing[model_name],
                )
                tasks.append(task)

        timeout = 10
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            try:
                story_id, model_name, prompt, response_dict, tokens_used, price = (
                    await asyncio.wait_for(task, timeout=timeout)
                )
                ALL_TOKENS_USED += tokens_used
                print(f"Tokens used: {ALL_TOKENS_USED}")
                TOTAL_PRICE += price
                print(f"Total Price: ${TOTAL_PRICE:.4f}")
                test_data[story_id]["negative"] = response_dict[
                    "continued_story_sentences"
                ]

                predictions[story_id] = test_data[story_id]

            except asyncio.TimeoutError:
                print(f"Task timed out: {task}")

        batch_idx += 1
        print(len(predictions))
        print("saving cache!")
        save_cache(cache_path, predictions)
        print("done saving cache")

    print(f"TOTAL: {len(predictions)}")
    return predictions


def save_predictions(predictions, file_path):
    with open(file_path, "w") as file:
        json.dump(predictions, file, indent=4)


async def main():

    prompts_folder = Path("data/rocstories/prompts")

    negative_prompt_file = prompts_folder / "negative_prompt.txt"

    data_file = "data/rocstories/split/train_narrativetwins.json"
    all_data = load_data([data_file])

    negative_prompt = load_prompt(negative_prompt_file)

    cache_path = Path("data/rocstories/split/cache.json")

    predictions_prompt = await run_experiment(all_data, negative_prompt, cache_path)
    out_file = "data/rocstories/split/train_full_narrative_twins_negatives.json"

    save_predictions(predictions_prompt, out_file)


if __name__ == "__main__":
    asyncio.run(main())
