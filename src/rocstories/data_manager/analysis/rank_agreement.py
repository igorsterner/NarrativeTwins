import json

from src.rocstories.evaluation import metrics

with open("data/rocstories/split/test.json", "r") as f:
    data1 = json.load(f)
with open("data/rocstories/split/test_double.json", "r") as f:
    data2 = json.load(f)

spearman_scores = []
auc_scores = []
count = 0
for story_id in data1:

    if story_id not in data2:
        continue

    count += 1
    labels = data1[story_id]["most_important"]
    selections = data2[story_id]["most_important"]

    saliency_scores = metrics.counts_from_labels(selections)

    spearman = metrics.spearman(saliency_scores, labels)
    if spearman is None:
        continue

    spearman_scores.append(spearman)

    auc = metrics.auc(saliency_scores, labels)
    auc_scores.append(auc)

average_spearman = sum(spearman_scores) / len(spearman_scores) if spearman_scores else 0
average_auc = sum(auc_scores) / len(auc_scores) if auc_scores else 0

print("Average Spearman:", average_spearman)
print("Average AUC:", average_auc)
