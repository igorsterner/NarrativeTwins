import csv
import json
import random
from pathlib import Path


def set_seed(seed):
    random.seed(seed)


class ROCStoriesProcessor:
    def __init__(self):
        set_seed(seed=2025)

    def read_json(self, json_path):
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def write_split_json(self, data, json_path, subset_size):
        data_split = {}

        if subset_size is not None:
            split_keys = random.sample(list(data.keys()), subset_size)
            for key in split_keys:
                data_split[key] = data.pop(key)
        else:
            data_split = data
            data = {}

        print(f"Writing {len(data_split)} stories to {str(json_path).split('/')[-1]}")
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data_split, f, indent=4)

        return data

    def story_preprocessing(self, text):
        return text.replace("  ", " ").strip()

    def load_rocstories(self, roc_files):
        stories = {}

        for file in roc_files:
            with file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    story_id = row["storyid"]
                    stories[story_id] = {
                        "storytitle": row["storytitle"],
                        "story": [
                            self.story_preprocessing(row[f"sentence{i}"])
                            for i in range(1, 6)
                        ],
                    }

        stories = list(stories.items())
        random.shuffle(stories)
        return dict(stories)

    def data_split(self):
        rocstories_path = Path("data/rocstories/")
        save_path = Path("data/rocstories/split/")

        roc_files = [
            rocstories_path / "ROCStories_winter2017 - ROCStories_winter2017.csv",
            rocstories_path / "ROCStories__spring2016 - ROCStories_spring2016.csv",
        ]

        rocstories = self.load_rocstories(roc_files)

        test_set_size = 250
        val_set_size = 250
        train_set_size = 20000

        rocstories = self.write_split_json(
            rocstories, save_path / "val.json", subset_size=val_set_size
        )
        rocstories = self.write_split_json(
            rocstories, save_path / "test.json", subset_size=test_set_size
        )
        # rocstories = self.write_split_json(rocstories, save_path / f"qualification-{qual_num}.json", subset_size=qualification_set_size)
        rocstories = self.write_split_json(
            rocstories, save_path / "train.json", subset_size=train_set_size
        )


if __name__ == "__main__":
    processor = ROCStoriesProcessor()
    processor.data_split()
