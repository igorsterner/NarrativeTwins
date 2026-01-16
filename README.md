<div align="center">

<h1 style="text-align: center;">Contrastive Learning with Narrative Twins for Modeling Story Salience</h1>

<div>
  <a href='https://igorsterner.github.io/' target='_blank'><b>Igor Sterner</b></a>&emsp;
  <a href='https://homepages.inf.ed.ac.uk/alex/' target='_blank'><b>Alex Lascarides</b></a>&emsp;
  <a href='https://homepages.inf.ed.ac.uk/keller/' target='_blank'><b>Frank Keller</b></a>
</div>

<div>
School of Informatics<br>
University of Edinburgh<br>
United Kingdom
</div>

</div>

<h4>

</h4>

<div align="center">

[![arxiv](https://img.shields.io/badge/arXiv-2601.07765-b31b1b.svg)](https://arxiv.org/abs/2601.07765)
[![Dataset: ROCStory Salience Annotation](https://img.shields.io/badge/Dataset-ROCStory_Salience-FFD21E.svg)](https://huggingface.co/datasets/igorsterner/rocstory-salience)
[![Dataset: ROCStory Summaries](https://img.shields.io/badge/Dataset-ROCStory_Summaries-FFD21E.svg)](https://huggingface.co/datasets/igorsterner/rocstory-summaries)
[![Dataset: ROCStory Narrative Twins](https://img.shields.io/badge/Dataset-ROCStory_Narrative_Twins-FFD21E.svg)](https://huggingface.co/datasets/igorsterner/rocstory-narrative-twins)
[![Dataset: Wikipedia Narrative Twins](https://img.shields.io/badge/Dataset-Wikipedia_Narrative_Twins-FFD21E.svg)](https://huggingface.co/datasets/igorsterner/wikipedia-narrative-twins)

</div>

<div align="left">

## Abstract

Understanding narratives requires identifying which events are most salient for a story's progression. We present a contrastive learning framework for modeling narrative salience that learns story embeddings from narrative twins: stories that share the same plot but differ in surface form. Our model is trained to distinguish a story from both its narrative twin and a distractor with similar surface features but different plot. Using the resulting embeddings, we evaluate four narratologically motivated operations for inferring salience (deletion, shifting, disruption, and summarization). Experiments on short narratives from the ROCStories corpus and longer Wikipedia plot summaries show that contrastively learned story embeddings outperform a masked-language-model baseline, and that summarization is the most reliable operation for identifying salient sentences. If narrative twins are not available, random dropout can be used to generate the twins from a single story. Effective distractors can be obtained either by prompting LLMs or, in long-form narratives, by using different parts of the same story.

## Installation

#### Coding Environment
```bash
conda create -n narrativetwins python=3.13
conda activate narrativetwins
```

Clone this repository

```bash
git clone https://github.com/igorsterner/NarrativeTwins
cd NarrativeTwins
```

Then, install the dependencies.

```bash
pip install -r requirements.txt
```

And make sure you have your python path set appropriately.

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## ROCStories

#### Data

Our salience annotations are in a pre-prepared format in `data/rocstories/split` as `val.json ` and `test.json`. The human-written summaries are also available in this directory, `val_summaries.json` and `test_summaries.json`. The double annoated subset of the test set is available as `test_double.json`.

For the narrative twins, you have two options. Either you can re-create the dataset or you use the provided `train.json` in the same folder.

(optional) If you want to re-create the training dataset:

```bash
# Request ROCStories data (https://cs.rochester.edu/nlp/rocstories/)
# Download data to "data/rocstories/ROCStories_winter2017 - ROCStories_winter2017.csv" and "data/rocstories/ROCStories__spring2016 - ROCStories_spring2016.csv"

# Split data into train/val/test. Note that if you wish to use our salience annotations, you will need to ensure your val/test sets are identical to ours on huggingface and non-overalpping with the train split.
python src/rocstories/data_manager/split/split_rocstories.py

# Prompt an LLM for narrative twins and distractor negatives
python src/rocstories/data_manager/narrative_twins/retellings_prompt.py
python src/rocstories/data_manager/narrative_twins/negatives_prompt.py
python src/rocstories/data_manager/narrative_twins/negatives_retellings_prompt.py
```

#### Training, evaluation and baselines

Run a training run using the provided configuration file, evaluate the model on the test set, get the baselines and run significance tests. The following commands should work with the data provided in this repo.

```bash
python src/rocstories/training/main.py
python src/rocstories/evaluation/evaluator.py

python src/rocstories/evaluation/baselines.py

python src/rocstories/evaluation/llm_annotation_style_prompting.py # OpenAI key required in data/keys/openai
python src/rocstories/evaluation/llm_salience_style_prompting.py # OpenAI key required in data/keys/openai

python src/rocstories/data_manager/analysis/rank_agreement.py
```

## Wikipedia Narratives


#### Materials

Load the evaluation data as follows

```
cd data/wikipedia
wget https://github.com/ppapalampidi/TRIPOD/raw/refs/heads/master/Synopses_and_annotations/TRIPOD_synopses_test.csv
wget https://raw.githubusercontent.com/ppapalampidi/TRIPOD/refs/heads/master/Synopses_and_annotations/TRIPOD_synopses_train.csv

cd ../..
python src/wikipedia/data_manager/tripod.py
```

Training narrative twins are provided as required in `data/wikipedia/train.json`.

(optional) If you wish to re-create this data, follow these steps.

```
cd data/wikipedia/
wget https://ltdata1.informatik.uni-hamburg.de/tell_me_again_v1.zip

cd ../..
python src/wikipedia/data_manager/wikipedia_narratives.py
python src/wikipedia/data_manager/hard_negatives_prompt.py
```

#### Training, evaluation and baselines

Run a training run using the provided configuration file, evaluate the model on the test set, get the baselines and run significance tests. The following commands should work with the data provided in this repo.

```
python src/wikipedia/training/main.py
python src/wikipedia/evaluation/evaluator.py

python src/wikipedia/evaluation/baselines.py

python src/wikipedia/evaluation/salience_prompting.py # OpenAI key required in data/keys/openai
```

## Ablations

Replacing entities according to Hatzel et al. uses the scripts in `src/rocstories/data_manager/anonymization/` for ROCStories and the data provided by Hatzel et al. for Wikipedia narratives (load and align `story_b_anon` in `src/wikipedia/data_manager/plot_summaries.py`). Training run variants can be performed by changing the `config.yaml` files accordingly or via command line arguments (we use Hydra, for instance you can run the `main.py` training scripts with `loss.type=simcse` or `loss.type=mlm`).

## Citation

If our code our paper has been useful to you, please consider citing our paper:

```
@inproceedings{sterner-2026-narrativetwins,
    title = "Contrastive Learning with Narrative Twins for Modeling Story Salience",
    author = "Sterner, Igor and Lascarides, Alex  and Keller, Frank ",
    booktitle = "Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
}
```
