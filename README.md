# DisentQA: Disentangling Parametric and Contextual Knowledge with Counterfactual Question Answering

This code accompanies the paper [DisentQA: Disentangling Parametric and Contextual Knowledge with Counterfactual
Question Answering](https://arxiv.org/pdf/2211.05655.pdf). Our approach proposes to disentangle two sources of knowledge available to QA models:
(1) parametric knowledge – the factual knowledge encoded in the model weights, and (2) contextual knowledge – external
knowledge (e.g., a Wikipedia passage) given to the model to generate a grounded answer. Using counterfactual data
augmentation, we introduce a model that predicts two answers for a given question, one per each source of knowledge.

<!-- TOC -->

* [Datasets](#datasets)
* [Prerequisites](#prerequisites)
* [Usage](#usage)
* [Citation](#citation)
<!-- TOC -->

## Datasets
The datasets produced by the `prepare_data.py` script are available in
this [document](https://docs.google.com/document/d/1Z4vA7ifMQTk5YBF3BEYCFSnIvXCYaznLP_7VBcXPEeU/edit?usp=sharing), which
contains links to the data files and their description, according to Table 3 in the paper:

![](/Users/ella/Documents/hujigoog/table_3.png)

The Evaluation Set section contains one link to all 4 types of data in the multi answer format. One can compare only the
contextual answers from it, in order to evaluate the answers from single answer format.

The datasets are stored in csv files. The key columns are:

- `'question'` - the original question from NQ.
- `'context'` - original or altered context given to the model (according to the `type` column.)
- `'parametric_answer'` - an answer that suits the original question from NQ.
- `'contextual_answer'` - an answer that suits the question according to the given context.
- `'answerable'` - boolean, is the question answerable using its context or not.
- `'type'` - factual, counterfactual, closed_book, or random_context.
    - if closed_book the context is ignored and the model gets an empty context.
- `'input'` - a concatenated string: `question: <question>\n context: <context>`
- `'output'` - a concatenated string: `comtextual: <contextual_answer>\n parametric: <parametric_answer>`

Our datasets are based on a subset of
the [Natural Questions](https://ai.google.com/research/NaturalQuestions/download) (NQ)
dataset. Specifically, the
[simplified version of the training set](https://storage.cloud.google.com/natural_questions/v1.0-simplified/simplified-nq-train.jsonl.gz)
for training and validation, and
the [development set](https://storage.cloud.google.com/natural_questions/v1.0-simplified/nq-dev-all.jsonl.gz) as a
held-out set for evaluation. For the factual dataset we always use the “gold” passage as the context, assuming an oracle
retrieval system. We use the examples that have both a gold passage and a short answer (35% of the data). For the
counterfactual dataset we use [Longpe et al. (2021)](https://github.com/apple/ml-knowledge-conflicts)'s framework with a
few additions that we show [here](ml-knowledge-conflicts/src/).

## Prerequisites

```
blis==0.4.1
catalogue==1.0.0
certifi==2021.10.8
charset-normalizer==2.0.12
cymem==2.0.6
distlib==0.3.4
filelock==3.7.1
idna==3.3
murmurhash==1.0.7
numpy==1.22.2
pandas==1.4.1
pbr==5.9.0
Pillow==9.0.1
plac==1.1.3
platformdirs==2.5.2
preshed==3.0.6
python-dateutil==2.8.2
pytz==2021.3
requests==2.27.1
spacy==2.2.4
srsly==1.0.5
stevedore==3.5.0
thinc==7.4.0
torch==1.10.2
torchaudio==0.10.2
torchvision==0.11.3
tqdm==4.64.0
typing_extensions==4.1.1
urllib3==1.26.9
virtualenv==20.14.1
virtualenv-clone==0.5.7
virtualenvwrapper==4.8.4
wasabi==0.9.1
```

## Usage

This is the basic flow of the project:

- [prepare_data.py](prepare_data.py) parses the NQ original dataset from the paths specified in the `config.json` file
  and splits it into training, validation and evaluation sets with this command:

  ```commandline
  python prepare_data.py --mode split 
  ```
  At this stage, the factual dataset has been created. Next, we assume
  the [ml-knowledge-conflicts](ml-knowledge-conflicts)
  scripts were updated in
  the [original repository clone directory](https://github.com/apple/ml-knowledge-conflicts.git).
  Here, we assume the directory sits in the same directory as the project's directory (sibling directories).
  We run the code to produce counterfactual examples.

  ```commandline
  python load_dataset.py -d NaturalQuestionsTrainTrain -w wikidata/entity_info.json.gz
  python load_dataset.py -d NaturalQuestionsTrainVal -w wikidata/entity_info.json.gz
  python load_dataset.py -d NaturalQuestionsDevAny -w wikidata/entity_info.json.gz
  ```

  ```commandline
  python generate_substitutions.py --inpath datasets/normalized/NaturalQuestionsTrainTrain.jsonl.gz --outpath datasets/substitution-sets/NaturalQuestionsTrainTrain_corpus-substitution.jsonl corpus-substitution -n 1
  python generate_substitutions.py --inpath datasets/normalized/NaturalQuestionsTrainVal.jsonl.gz --outpath datasets/substitution-sets/NaturalQuestionsTrainVal_corpus-substitution.jsonl corpus-substitution -n 1
  python generate_substitutions.py --inpath datasets/normalized/NaturalQuestionsDevAny.jsonl.gz --outpath datasets/substitution-sets/NaturalQuestionsDevAny_corpus-substitution.jsonl corpus-substitution -n 1
  ```
  Finally, the rest of the data preparation code can be executed, to create all kinds of training/validation/evaluation
  sets for each baseline/model for comparison.
  ```commandline
  python prepare_data.py --mode enrich 
  ```

  The output of this stage are the [datasets](#datasets) described above.

- [run_nq_fine_tuning.py](run_nq_fine_tuning.py) fine-tunes a T5 model based on the implementation
  from [this notebook](https://colab.research.google.com/drive/1WXLtGQmYyrMi484ox9R5ZkJe_4Vm3fny). Saves the checkpoints
  to the `checkpoints_dirpath` specified in the [config.json](config.json) file, under the `fine_tune` key.
  ```commandline
  python run_nq_fine_tuning.py --path data/<path prefix for train/val data>
  ```
- [query_model.py](query_model.py) - inference time, gets a path to a trained model `checkpoint_name` and loads it to
  evaluate the given data in `path` on a specific `answer_type` or all of them.
  ```commandline
  python query_model.py --answer_type <f/cf/rc/cb>  
                        --path data/simplified-nq-dev_any_from_cf_full_subset_no_dec.csv 
                        --checkpoint_name checkpoints/<checkpoint_name>.ckpt
  ```
- [evaluate.py](evaluate.py) - Gets inference outputs and calculates scores to evaluate the model.
  ```commandline
   python evaluate.py --path <path to query_model output file with all answer types>.csv
  ```
- [config.json](config.json) holds all the necessary configuration fields (per script), as shown below. Fill
  in paths and api keys.
  ```commandline
  {
    "prepare_data": {
      "train_path":
      "dev_path": 
      "counterfactual_path_pattern": 
    },

    "fine_tune": {
      "checkpoints_dirpath": "",
      "wandb_api_key": "",
      "model_name": "t5-small",
      ...
    },

    "query_model": {
      "results_dir": "",
      "model_name": "t5-small",
      ...
    }
  } 
    ```

## Citation
If you make use of this code or data, please cite the following paper:
  ```
    @misc{https://doi.org/10.48550/arxiv.2211.05655,
    title = {DisentQA: Disentangling Parametric and Contextual Knowledge with Counterfactual Question Answering},
    author = {Neeman, Ella and
    Aharoni, Roee and
    Honovich, Or and
    Choshen, Leshem and
    Szpektor, Idan and
    Abend, Omri},
    keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer
    and information sciences, FOS: Computer and information sciences},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International},
    doi = {10.48550/ARXIV.2211.05655},
    url = {https://arxiv.org/abs/2211.05655}
    }
  ```






