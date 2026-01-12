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


def load_prompts(extended_path, retelling_path):
    extended_prompt = load_prompt(extended_path)
    retelling_prompt = load_prompt(retelling_path)
    return extended_prompt, retelling_prompt


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
    extended_prompt,
    retelling_prompt,
    pricing,
    max_retries=3,
):
    story_text = " ".join(story_details["negative"])

    prompt1 = extended_prompt.replace("INSERT_STORY", story_text)

    result1 = await process_item(
        client, model_name, story_id, story_details, prompt1, pricing, max_retries
    )
    extended_story = result1[3]
    prompt2 = retelling_prompt.replace("INSERT_STORY", extended_story)

    result2 = await process_item(
        client, model_name, story_id, story_details, prompt2, pricing, max_retries
    )
    retelling = result2[3]
    tokens_used = result1[4] + result2[4]
    price = result1[5] + result2[5]

    extended_story_sentences = nltk.sent_tokenize(extended_story)
    retelling_sentences = nltk.sent_tokenize(retelling)

    return (
        story_id,
        model_name,
        prompt1,
        {
            "extended_story": extended_story,
            "extended_story_sentences": extended_story_sentences,
            "narrative_twin": retelling,
            "retelling_sentences": retelling_sentences,
        },
        tokens_used,
        price,
    )


async def run_experiment(test_data, extended_prompt, retelling_prompt, cache_path):
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
                    extended_prompt,
                    retelling_prompt,
                    model_pricing[model_name],
                )
                tasks.append(task)

        timeout = 10
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            try:
                story_id, model_name, prompt1, response_dict, tokens_used, price = (
                    await asyncio.wait_for(task, timeout=timeout)
                )
                ALL_TOKENS_USED += tokens_used
                print(f"Tokens used: {ALL_TOKENS_USED}")
                TOTAL_PRICE += price
                print(f"Total Price: ${TOTAL_PRICE:.4f}")
                test_data[story_id]["extended_negative"] = response_dict[
                    "extended_story_sentences"
                ]
                test_data[story_id]["negative_narrative_twin"] = response_dict[
                    "retelling_sentences"
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

    extended_prompt_path = prompts_folder / "extended_version_prompt.txt"
    retelling_prompt_path = prompts_folder / "narrativetwin_prompt.txt"

    data_file = "data/rocstories/split/train_full_narrative_twins_negatives.json"
    all_data = load_data([data_file])

    extended_prompt, retelling_prompt = load_prompts(
        extended_prompt_path, retelling_prompt_path
    )

    cache_path = Path("data/rocstories/split/cache-climcon-ret.json")

    predictions_prompt = await run_experiment(
        all_data, extended_prompt, retelling_prompt, cache_path
    )
    out_file = "data/rocstories/split/train.json"

    save_predictions(predictions_prompt, out_file)


if __name__ == "__main__":
    asyncio.run(main())
