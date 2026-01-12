import json
from pathlib import Path

import nltk
import spacy
import torch
from coref import add_coref, request_coref
from fastcoref import FCoref
from tqdm import tqdm

from src.rocstories.data_manager.anonymization import (anonymize, names)

BABY_NAMES_PATH = "data/rocstories/baby-names.csv"
name_db = names.NameDB(BABY_NAMES_PATH)

nlp = spacy.load("en_core_web_lg", disable=["ner"])

nlp.add_pipe("flair_ner")

coref_model = FCoref(device="cuda:0" if torch.cuda.is_available() else "cpu")


def anonymize_text(text):
    doc = nlp(text)
    coref_preds = request_coref(coref_model, doc)
    doc = add_coref(doc, coref_preds)
    return anonymize.coref_replace(doc, name_db)


def main():

    data_dir = Path("data/rocstories/split/")
    input_path = data_dir / "train.json"
    output_path = data_dir / "train_ner_coref.json"

    with open(input_path, "r") as f:
        data = json.load(f)

    for story_id in tqdm(data.keys()):

        # 1
        sentences = data[story_id]["narrative_twin"]
        story = " ".join(sentences)
        anonymized_story = anonymize_text(story)
        anonymized_sentences = nltk.sent_tokenize(anonymized_story)
        data[story_id]["narrative_twin"] = anonymized_sentences

        # 2
        sentences = data[story_id]["negative"]
        story = " ".join(sentences)
        anonymized_story = anonymize_text(story)
        anonymized_sentences = nltk.sent_tokenize(anonymized_story)
        data[story_id]["negative"] = anonymized_sentences

    with open(output_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved anonymized retellings to {output_path}")


if __name__ == "__main__":
    main()
