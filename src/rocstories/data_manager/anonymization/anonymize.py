import json
import os
import random
import string
from collections import Counter


def texts_to_entities(texts):
    from flair.data import Sentence
    from flair.nn import Classifier

    tagger = Classifier.load("ner")
    all_entities = []
    for sentences in texts:
        sents = [Sentence(s) for s in sentences]
        tagger.predict(sents)
        entities = []
        for s in sents:
            for label in s.get_labels():
                for token in label.data_point.tokens:
                    entities.append(token.text.lower())
        all_entities.append(entities)
    return all_entities


def add_renamed_texts():
    from spacy.tokens import Doc
    from spacy.vocab import Vocab

    ds = dataset.SummaryDataset("data")
    name_db = NameDB("data/baby-names.csv")
    for id_ in ds.stories.keys():
        lang_ids, translations = ds[id_].get_all_summaries_en()
        renamed = {}
        for lang_id, text in zip(lang_ids, translations):
            spacy_path = f"data/spacy/{id_[:2]}/{id_}_{lang_id}.spacy"
            coref_path = f"data/coref/{id_[:2]}/{id_}_{lang_id}.json"
            doc = Doc(Vocab()).from_disk(spacy_path)
            if len(doc) == 0:
                continue
            coref_info = json.load(open(coref_path))
            add_coref(doc, coref_info)
            renamed[lang_id] = coref_replace(doc, name_db)
        summary_path = f"data/summaries/{id_[:2]}/{id_}.json"
        data = json.load(open(summary_path))
        data["anonymized"] = renamed
        temp_file_path = summary_path + "_temp"
        json.dump(data, open(temp_file_path, "w"))
        os.replace(temp_file_path, summary_path)


def guess_text_span_gender(text, name_db):
    sexes = []
    for name in text.split(" "):
        if sex := name_db.get_sex_for_name(name):
            sexes.append(sex)
    try:
        sex = Counter(sexes).most_common(1)[0][0]
    except IndexError:
        sex = None
    return sex


def get_cluster_name(cluster, used_names, name_db):
    resp = None
    first_time = True
    counter = 0
    while (first_time or (resp in used_names)) and counter < 10:
        first_time = False
        counter += 1
        if cluster.ner_label == "PER":
            sexes = []
            for span in cluster.spans:
                for name in span.text.split(" "):
                    if sex := name_db.get_sex_for_name(name):
                        sexes.append(sex)
            try:
                sex = Counter(sexes).most_common(1)[0][0]
            except IndexError:
                sex = None
            resp = name_db.random_name_with_sex(sex)
        elif cluster.ner_label == "LOC":
            resp = f"Location {string.ascii_uppercase[cluster.id % 26]}"
        elif cluster.ner_label == "ORG":
            resp = f"Organization {string.ascii_uppercase[cluster.id % 26]}"
        elif cluster.ner_label == "MISC":
            resp = f"Entity {string.ascii_uppercase[cluster.id % 26]}"
        else:
            resp = None
    return resp


def get_replacement_text(tag, text, name_db, used_names, performed_replacements):
    out = None
    counter = 0
    if already_replaced := performed_replacements.get((tag, text)):
        return already_replaced, performed_replacements
    while out is None or out in used_names and counter < 100:
        counter += 1
        if tag == "PER":
            sex = guess_text_span_gender(text, name_db)
            out = name_db.random_name_with_sex(sex)
        elif tag == "LOC":
            out = f"Location {random.choice(string.ascii_uppercase)}"
        elif tag == "ORG":
            out = f"Organization {random.choice(string.ascii_uppercase)}"
        elif tag == "MISC":
            out = f"Entity {random.choice(string.ascii_uppercase)}"
    if counter == 100:
        out = text
    performed_replacements.update({(tag, text): out})
    return out, performed_replacements


def coref_replace(doc, name_db):
    posessives = set(["my", "our", "your", "his", "her", "its", "their", "whose"])
    pronouns = (
        set(
            [
                "I",
                "you",
                "he",
                "she",
                "it",
                "we",
                "you",
                "they",
                "me",
                "you",
                "him",
                "her",
                "it",
                "us",
                "you",
                "them",
            ]
        )
        | posessives
    )
    replacements = []
    used_names = set()
    performed_singleton_replacements = {}
    for cluster in doc._.coref_clusters:
        cluster_name = get_cluster_name(cluster, used_names, name_db)
        if cluster_name is not None:
            used_names.add(cluster_name)
        previous_span = None
        for span in cluster.spans:
            # If it's very recent we continue using the possesssives
            if (
                (previous_span is not None)
                and (previous_span.start + 8 >= span.start)
                and (span.text.lower().strip() in pronouns)
            ):
                replacement_text = span.text
            elif (
                span.text.endswith("'s") or span.text.lower() in posessives
            ) and cluster_name is not None:
                replacement_text = cluster_name + "'s"
            else:
                replacement_text = cluster_name
            previous_span = span
            replacements.append((span, replacement_text))
    for span in doc.ents:
        if span._.has_coref:
            continue
        # We can assume it to be a singleton.
        replace, performed_singleton_replacements = get_replacement_text(
            span.label_,
            span.text,
            name_db,
            used_names,
            performed_singleton_replacements,
        )
        replacements.append((span, replace))

    sorted_replacements = sorted(replacements, key=lambda rep: rep[0].start_char)
    texts = []
    current_pos = 0
    for span, replacement_text in sorted_replacements:
        if replacement_text is None:
            texts.append(doc.text[current_pos : span.end_char])
            current_pos = span.end_char
            continue
        if span.start_char < current_pos:
            continue
        texts.append(doc.text[current_pos : span.start_char])
        texts.append(replacement_text)
        current_pos = span.end_char
    texts.append(doc.text[current_pos:])
    return "".join(texts)
