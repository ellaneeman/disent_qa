import argparse
import os
import pathlib

import numpy as np
import json
import pandas as pd
import gzip

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)


########################################################################################################################
# parse_and_split_data
########################################################################################################################
def json_to_pandas(path, gzipped=False, nrows=None):
    """
    helper function to covert json or json.gzip to a pandas dataframe
    """
    data = []
    if not gzipped:
        with open(path) as f:
            for i, line in enumerate(f):
                if i == nrows:
                    break
                data.append(json.loads(line))
    else:
        with gzip.open(path) as f:
            for i, line in enumerate(f):
                if i == nrows:
                    break
                jsonline = json.loads(line)
                data.append(jsonline)
    df = pd.DataFrame(data)
    return df


def save_split(df, path=None):
    df_answerable = df[df["type"] != "unanswerable"]
    train_df_answerable, dev_df_answerable = split_by_fraction(df_answerable)
    train_df_answerable.to_csv(path + "_train_split.csv")
    dev_df_answerable.to_csv(path + "_val_split.csv")
    return train_df_answerable, dev_df_answerable


def split_by_fraction(df, fraction=0.8):
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle rows inplace
    split_place = int(len(df) * fraction)
    train_df = df[:split_place]
    dev_df = df[split_place:]
    return train_df, dev_df


def split_answerabilty(df):
    df_answerable = df[df["answerable"]]
    df_unanswerable = df[~df["answerable"]]
    df_unanswerable["contextual_answer"] = "unanswerable"
    df_unanswerable["parametric_answer"] = ""
    df_answerable["parametric_answer"].fillna("", inplace=True)
    df_no_short_answer = df_answerable[df_answerable['answer'] == '']
    df_no_short_answer["contextual_answer"] = "unanswerable"
    df_no_short_answer["parametric_answer"] = ""
    df_no_short_answer["type"] = "unanswerable"
    df_no_short_answer["answerable"] = False
    df_answerable = df_answerable[df_answerable['answer'] != '']
    df_answerable["type"] = "factual"
    df_unanswerable["type"] = "unanswerable"
    return df_answerable, df_unanswerable, df_no_short_answer


def extract_context_from_annotations(df):
    """
    Gets a NQ df with long answers' annotations containing valid long answers or -1 for no answers.
    Collects contexts according to the annotations or dummy context for unanswerables.
    """
    df = df.assign(ls=df["annotations"].str[0].str['long_answer'].str['start_token'])
    df = df.assign(le=df["annotations"].str[0].str['long_answer'].str['end_token'])
    df = df.assign(ss=df["annotations"].str[0].str['short_answers'].str[0].str['start_token'].fillna(-1).astype(int))
    df = df.assign(se=df["annotations"].str[0].str['short_answers'].str[0].str['end_token'].fillna(-1).astype(int))
    df_answerable = df[df["annotations"].str[0].str['long_answer'].str['candidate_index'] != -1]
    df_answerable = df_answerable.assign(
        context=df_answerable.apply(lambda x: " ".join(x["document_text"].split()[x["ls"]:x["le"]]), 1))
    df_answerable = df_answerable.assign(
        answer=df_answerable.apply(lambda x: " ".join(x["document_text"].split()[x["ss"]:x["se"]]), 1))
    df_answerable["answerable"] = True
    # df_unanswerable = df[df["annotations"].str[0].str['long_answer'].str['candidate_index'] == -1]
    # df_unanswerable=df_unanswerable.assign(context=df_unanswerable.apply(extract_dummy_context_from_document,axis=1))
    # df_unanswerable["answerable"] = False
    # df = pd.concat([df_answerable, df_unanswerable], ignore_index=False)
    df_answerable.rename(columns={'question_text': 'question'}, inplace=True)
    return df_answerable.sort_index()  # TODO maybe unnecessary


def extract_context_from_annotations_for_filtered(df):
    """
    Gets a NQ df with long answers' annotations containing valid long answers or -1 for no answers.
    Collects contexts according to the annotations or dummy context for unanswerables.
    """
    df = df.assign(ls=df["first_valid_answer"].str['long_answer'].str['start_token'])
    df = df.assign(le=df["first_valid_answer"].str['long_answer'].str['end_token'])
    df = df.assign(ss=df["first_valid_answer"].str['short_answers'].str[0].str['start_token'].fillna(-1).astype(int))
    df = df.assign(se=df["first_valid_answer"].str['short_answers'].str[0].str['end_token'].fillna(-1).astype(int))
    df = df.assign(context=df.apply(lambda x: " ".join(x["document_text"].split()[x["ls"]:x["le"]]), 1))
    df = df.assign(answer=df.apply(lambda x: " ".join(x["document_text"].split()[x["ss"]:x["se"]]), 1))
    df.rename(columns={'question_text': 'question'}, inplace=True)
    df["answerable"] = True
    return df


def get_has_more_than_n_answer_df(df, n):
    """
    for majority - 2
    for any (at least 1) answer - 0
    for all - 4
    """
    filter_ = df["annotations"].apply(
        lambda x: sum(int(a['long_answer']['candidate_index'] != -1 and len(a['short_answers']) > 0) for a in x) > n)
    return df[filter_]


def get_first_valid_answer(df):
    exploded = df["annotations"].explode()
    filtered_exploded = exploded[
        (exploded.str["long_answer"].str["candidate_index"] != -1) & (exploded.str['short_answers'].str.len() > 0)]
    gb = filtered_exploded.groupby(filtered_exploded.index)
    return gb.first()


def parse_and_split_data(data_path):
    simplified_nq_train_df = json_to_pandas(data_path, gzipped=True)
    nq_with_context_df = extract_context_from_annotations(simplified_nq_train_df)  # factual, before split
    df_answerable, df_unanswerable, df_no_short_answer = split_answerabilty(nq_with_context_df)  # drop unanswerable
    save_split(df_answerable, path="".join(data_path.split(".")[:-2]))


def parse_and_evaluation_set(dev_path):
    df = json_to_pandas(dev_path, gzipped=True)
    df_any = get_has_more_than_n_answer_df(df, 0)
    df_any = df_any.assign(first_valid_answer=get_first_valid_answer(df_any))
    df_any = extract_context_from_annotations_for_filtered(df_any)
    df_any = create_factual(df_any)
    df_any.to_csv(dev_path.replace(".jsonl.gz", "_any.csv"))


########################################################################################################################
#  Explicit input/output format
########################################################################################################################

def disent_qa_explicit_answers(row, answerability_prediction=False):
    """
    format contextual_answer, parametric_answer, answerable into expected model output
    """
    if answerability_prediction:
        answerable = "answerable" if row["answerable"] else "unanswerable"
        if row["type"] != "unanswerable":
            formatted_answer = "{answerable}\ncontextual: {c}\nparametric: {p}".format(answerable=answerable,
                                                                                       c=row["contextual_answer"],
                                                                                       p=row["parametric_answer"])
        else:
            formatted_answer = "{answerable}\ncontextual: {c}\nparametric:".format(answerable=answerable,
                                                                                   c=row["contextual_answer"])
    else:
        if row["type"] != "unanswerable":
            formatted_answer = "contextual: {c}\nparametric: {p}".format(c=row["contextual_answer"],
                                                                         p=row["parametric_answer"])
        else:
            formatted_answer = "contextual: {c}\nparametric:".format(c=row["contextual_answer"])

    return formatted_answer


def disent_qa_explicit_input(row):
    """
    concatenate "question: {q}\ncontext: {c}".
    """
    question = row['question'].strip("?") + "?"
    if row["type"] == "closed_book":
        input = "question: {q}\ncontext: ".format(q=question)
    else:
        input = "question: {q}\ncontext: {c}".format(q=question, c=row['context'])
    return input


def disent_qa_explicit_answers_baseline(row, answer_type, answerability_declaration=False):
    """
    format contextual_answer, parametric_answer, answerable into expected model output
    """

    answer_col = answer_type + "_answer"
    if answerability_declaration:
        answerable = "answerable" if row["answerable"] else "unanswerable"
        formatted_answer = "{answerable}\n{answer_type}: {a}".format(answerable=answerable, answer_type=answer_type,
                                                                     a=row[answer_col])
    else:
        formatted_answer = "{answer_type}: {a}".format(answer_type=answer_type, a=row[answer_col])
    return formatted_answer


########################################################################################################################
#  data type dfs from df_answerable or from path
########################################################################################################################
def create_factual(df_answerable):
    df_factual = df_answerable.copy(deep=True)
    df_factual["contextual_answer"] = df_answerable["answer"]
    df_factual["parametric_answer"] = df_answerable["answer"]
    df_factual["type"] = "factual"
    df_factual["answerable"] = True
    return df_factual


def create_counterfactual_from_path(path_pattern, split_name="Train", sub_suffix="corpus_substitution", nrows=None):
    path = path_pattern.format(split_name, sub_suffix)
    df = json_to_pandas(path, nrows=nrows)
    df = df.iloc[1:, :]
    df['parametric_answer'] = df["original_example"].apply(lambda x: x['gold_answers'][0]['text'] if x == x else x)
    df['contextual_answer'] = df['gold_answers'].str[0].apply(lambda x: x['text'] if x == x else x)
    df['question'] = df['query']
    df = df.assign(original_context=df.original_example.str["context"])
    df["answerable"] = True
    df["type"] = "counterfactual"
    return df


def create_closed_book(df_answerable):
    df_closed_book = df_answerable.copy(deep=True)
    df_closed_book["type"] = "closed_book"
    df_closed_book["contextual_answer"] = "unanswerable"
    df_closed_book["answerable"] = False
    return df_closed_book


def create_random_context(df_answerable):
    df = df_answerable.copy(deep=True)
    df["random_context"] = np.roll(df["context"], 10)
    if len(df[df.apply(lambda x: x["parametric_answer"] in x["random_context"], axis=1)]) > 0:
        print("parametric_answer in random_context")
    df["type"] = "random_context"
    df["contextual_answer"] = "unanswerable"
    df["answerable"] = False
    df = df[["question", "random_context", "parametric_answer", "contextual_answer", "answerable", "type"]]
    df.rename(columns={"random_context": "context"}, inplace=True)
    return df


def create_factual_from_counterfactual(df_counterfactual):
    df = df_counterfactual[["question", "original_context", "parametric_answer", "contextual_answer", "answerable"]]
    df["contextual_answer"] = df["parametric_answer"]
    df = df.rename(columns={"original_context": "context"})
    df["type"] = "factual"
    return df


########################################################################################################################
# Turn NQ to Augmented Data
########################################################################################################################


def enrich_nq(path, counterfactual_path_pattern, split_name="Train", answerability_declaration=False, nrows=None):
    """
    Gets paths to files for factual +its counterfactual dfs. saves to a file the full (augmented) data.
    """
    path = path + f"_{split_name.lower()}_split.csv" if path[:-4] != ".csv" else path

    # get 4 answer types
    df_factual_answerable = pd.read_csv(path)
    df_counterfactual = create_counterfactual_from_path(path_pattern=counterfactual_path_pattern,
                                                        split_name=split_name,
                                                        nrows=nrows)
    df_closed_book = create_closed_book(df_factual_answerable)
    df_random_context = create_random_context(df_factual_answerable)

    # concatenate to one a full df
    df = pd.concat([df_factual_answerable, df_counterfactual, df_closed_book, df_random_context],
                   ignore_index=False)

    # create input/output cols based on the type
    df = df.assign(input=df.apply(disent_qa_explicit_input, axis=1))
    df = df[["question", "context", "parametric_answer", "contextual_answer", "answerable", "type", "input"]]
    df = df.assign(output=df.apply(disent_qa_explicit_answers, args=(answerability_declaration,), axis=1))

    path = path + "_full_explicit_{}_split.csv".format(
        split_name.lower()) if answerability_declaration else path + "_full_no_dec_{}_split.csv".format(
        split_name.lower())

    df = df.sort_values(by=['question', 'type'])
    df.to_csv(path)
    return df


def enrich_nq_from_counterfactuals(path, split_name="Train", suffix="", answerability_declaration=False, nrows=None):
    """
    Gets paths to files for factual +its counterfactual dfs. saves to a file the full (augmented) data.
    """
    df_counterfactual = create_counterfactual_from_path(split_name=split_name,
                                                        nrows=nrows)  # returns with extra columns
    df_factual = create_factual_from_counterfactual(df_counterfactual)
    df_counterfactual = df_counterfactual[["question", "context", "parametric_answer", "contextual_answer",
                                           "answerable", "type"]]  # , "input"]]
    df_closed_book = create_closed_book(df_factual)
    df_random_context = create_random_context(df_factual)
    # df_sub = create_counterfactual(nrows=nrows)
    df = pd.concat([df_factual, df_counterfactual, df_closed_book, df_random_context], ignore_index=False)
    df = df.assign(input=df.apply(disent_qa_explicit_input, axis=1))
    df = df[["question", "context", "parametric_answer", "contextual_answer", "answerable", "type", "input"]]
    # df = pd.concat([df, df_counterfactual], ignore_index=False)
    df = df.assign(output=df.apply(disent_qa_explicit_answers, args=(answerability_declaration,), axis=1))
    path = path[:-4] if path[-4:] == ".csv" else path
    path = path + "_from_cf"
    path = path + "_full_subset_explicit{}.csv".format(
        suffix) if answerability_declaration else path + "_full_subset_no_dec{}.csv".format(suffix)
    df = df.sort_values(by=['question', 'type'])
    df.to_csv(path)
    return df


########################################################################################################################
# Create baselines data from fully augmented data
########################################################################################################################

def create_closed_book_baseline_data(path, df_full, split_name, answerability_declaration=False):
    """
    (s) cb - single answer, closed book
    """
    df = df_full[df_full["type"] == "closed_book"]
    print(df.columns)
    df = df.assign(
        output=df.apply(lambda x: disent_qa_explicit_answers_baseline(x, "parametric", False), axis=1))
    df.to_csv(path + "_closed_book_baseline_{split_name}_split.csv".format(split_name=split_name))
    return df


def create_contextual_baseline_data(path, df_full, split_name, answerability_declaration=False):
    """
    (s) f - single answer, factual (vanilla)
    """
    df = df_full[df_full["type"] == "factual"]
    df = df.assign(
        output=df.apply(lambda x: disent_qa_explicit_answers_baseline(x, "contextual", False), axis=1))
    df.to_csv(path + "_contextual_baseline_{split_name}_split.csv".format(split_name=split_name))
    return df


def create_factual_counterfactual_contextual_baseline_data(path, df_full, split_name, answerability_declaration=False):
    """
    (s) f + cf - single answer, factual + counterfactual
    """
    df_f = df_full[df_full["type"] == "factual"]
    df_cf = df_full[df_full["type"] == "counterfactual"]
    df = pd.concat([df_f, df_cf])
    df = df.assign(
        output=df.apply(lambda x: disent_qa_explicit_answers_baseline(x, "contextual", answerability_declaration),
                        axis=1))
    df.to_csv(path + "_factual_counterfactual_contextual_baseline_{split_name}_split.csv".format(split_name=split_name))
    return df


def create_factual_counterfactual_answerabilty_contextual_baseline_data(path, df_full, split_name,
                                                                        answerability_declaration=False):
    """
    (s) f + cf + a - single answer,  factual + counterfactual + answerabilty
    """
    df_full = df_full.assign(
        output=df_full.apply(lambda x: disent_qa_explicit_answers_baseline(x, "contextual", answerability_declaration),
                             axis=1))
    df_full.to_csv(path + "_factual_counterfactual_answerabilty_contextual_baseline_{split_name}_split.csv".format(
        split_name=split_name))
    return df_full


def create_factual_counterfactual_disentangled_baseline_data(path, df_full, split_name,
                                                             answerability_declaration=False):
    """
    (m) f + cf - multi answer,  factual + counterfactual
    """
    df_f = df_full[df_full["type"] == "factual"]
    df_cf = df_full[df_full["type"] == "counterfactual"]
    df = pd.concat([df_f, df_cf])
    df = df.assign(
        output=df.apply(lambda x: disent_qa_explicit_answers(x, answerability_declaration), axis=1))
    df.to_csv(
        path + "_factual_counterfactual_disentangled_baseline_{split_name}_split.csv".format(split_name=split_name))
    return df


def create_factual_disentangelment_answerabilty(path, df_full, split_name, answerability_declaration=False):
    """
    (s) f + cf + a - single answer,  factual + counterfactual + answerabilty
    """
    df_f = df_full[df_full["type"] == "factual"]
    df_cb = df_full[df_full["type"] == "closed_book"]
    df_rc = df_full[df_full["type"] == "random_context"]
    df = pd.concat([df_f, df_cb, df_rc])
    df = df.assign(
        output=df.apply(lambda x: disent_qa_explicit_answers(x, answerability_declaration), axis=1))
    df.to_csv(
        path + "_factual_disentangled_answerability_baseline_{split_name}_split.csv".format(split_name=split_name))
    return df


def create_factual_answerabilty(path, df_full, split_name, answerability_declaration=False):
    df_f = df_full[df_full["type"] == "factual"]
    df_cb = df_full[df_full["type"] == "closed_book"]
    df_rc = df_full[df_full["type"] == "random_context"]
    df = pd.concat([df_f, df_cb, df_rc])
    df = df.assign(
        output=df.apply(lambda x: disent_qa_explicit_answers_baseline(x, "contextual", answerability_declaration),
                        axis=1))
    df.to_csv(
        path + "_factual_answerability_baseline_{split_name}_split.csv".format(split_name=split_name))
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=False, default="split",
                        help="split/enrich")
    args = parser.parse_args()

    f = open(os.path.dirname(__file__) + "/config.json")
    config = json.load(f)["prepare_data"]

    if args.mode == "split":
        parse_and_split_data(data_path=config["train_path"])
        parse_and_evaluation_set(config["dev_path"])
        # stop and run ml-knowledge-conflicts code
        exit(0)

    elif args.mode == "enrich":
        path_prefix = "".join(config["train_path"].split(".")[:-2])
        test_prefix = "".join(config["dev_path"].split(".")[:-2])
        cf_path_pattern = config["counterfactual_path_pattern"]
        fully_augmented_train_df = enrich_nq(path=path_prefix,
                                             counterfactual_path_pattern=cf_path_pattern,
                                             split_name="Train")
        fully_augmented_val_df = enrich_nq(path=path_prefix,
                                           counterfactual_path_pattern=cf_path_pattern,
                                           split_name="Val")
        fully_augmented_from_cf_test_df = enrich_nq_from_counterfactuals(test_prefix, split_name="DevAny")
        # fully_augmented_val_from_cf_df = enrich_nq_from_counterfactuals(path=path_prefix,
        #                                                                 suffix="_val_split",
        #                                                                 split_name="TrainVal")
        augmented_data_dir = "data/augmented/"
        pathlib.Path(augmented_data_dir).mkdir(parents=True, exist_ok=True)
        path_prefix = augmented_data_dir + path_prefix
        create_contextual_baseline_data(path_prefix, fully_augmented_train_df.copy(), "train")
        create_contextual_baseline_data(path_prefix, fully_augmented_val_df.copy(), "val")
        create_closed_book_baseline_data(path_prefix, fully_augmented_train_df.copy(), "train")
        create_closed_book_baseline_data(path_prefix, fully_augmented_val_df.copy(), "val")
        create_factual_counterfactual_disentangled_baseline_data(path_prefix, fully_augmented_train_df.copy(), "train")
        create_factual_counterfactual_disentangled_baseline_data(path_prefix, fully_augmented_val_df.copy(), "val")
        create_factual_counterfactual_contextual_baseline_data(path_prefix, fully_augmented_train_df.copy(), "train")
        create_factual_counterfactual_contextual_baseline_data(path_prefix, fully_augmented_val_df.copy(), "val")
        create_factual_counterfactual_answerabilty_contextual_baseline_data(path_prefix,
                                                                            fully_augmented_train_df.copy(), "train")
        create_factual_counterfactual_answerabilty_contextual_baseline_data(path_prefix, fully_augmented_val_df.copy(),
                                                                            "val")
        create_factual_disentangelment_answerabilty(path_prefix, fully_augmented_train_df.copy(), "train")
        create_factual_disentangelment_answerabilty(path_prefix, fully_augmented_val_df.copy(), "val")
        create_factual_answerabilty(path_prefix, fully_augmented_train_df.copy(), "train")
        create_factual_answerabilty(path_prefix, fully_augmented_val_df.copy(), "val")

    else:
        print("please select mode")
        exit(1)
