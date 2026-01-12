import csv
import json

from src.wikipedia.data_manager import data_utils

INPUT_FILES = [
    "data/wikipedia/TRIPOD_synopses_test.csv",
    "data/wikipedia/TRIPOD_synopses_train.csv",
]
OUTPUT_FILE = "data/wikipedia/tripod.json"


def parse_sentences(text):

    start_tag = "[STR_SENT]"
    end_tag = "[END_SENT]"
    sentences = []
    i = 0
    while True:
        start = text.find(start_tag, i)
        if start == -1:
            break

        start += len(start_tag)
        end = text.find(end_tag, start)

        if end == -1:
            break

        seg = text[start:end].replace("\n", " ").strip()

        if seg:
            sentences.append(seg)

        i = end + len(end_tag)

    return sentences


def row_to_record(row):

    sentences = parse_sentences(row["synopsis_segmented"])
    sentences = [data_utils.clean_sentence(s) for s in sentences]
    return row["movie_name"], {
        "sentences": sentences,
        "tp1": int(row["tp1"]),
        "tp2": int(row["tp2"]),
        "tp3": int(row["tp3"]),
        "tp4": int(row["tp4"]),
        "tp5": int(row["tp5"]),
    }


def read_data(csv_paths):

    data = {}

    for csv_path in csv_paths:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                title, d = row_to_record(row)
                if title.endswith("_1") or title.endswith("_2"):
                    continue
                else:
                    data[title] = d

    return data


def main():
    data = read_data(INPUT_FILES)

    print(len(data))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
