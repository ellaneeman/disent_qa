#!/usr/bin/python
# -*- coding: latin-1 -*-
import argparse
from pathlib import Path

import pandas as pd
import os
import evaluation_script_squad_v2 as ev

pd.set_option('display.max_columns', None)

NAMES = ["cc_em", "cc_f1", "pp_em", "pp_f1", "sep_cp_em", "sep_cp_f1", "cp_em", "cp_f1", "pc_em", "pc_f1", "uc_em",
         "up_em"]
factual = ["cc_em", "cc_f1", "pp_em", "pp_f1", "sep_cp_em", "sep_cp_f1"]
cf = ["cc_em", "cc_f1", "pp_em", "pp_f1", "sep_cp_em", "sep_cp_f1", "cp_em", "cp_f1", ]  # with p peeking at c gold
cb = ["cc_em", "cc_f1", "pp_em", "pp_f1", "sep_cp_em", "sep_cp_f1", "pc_em", "pc_f1"]  # with c peeking at p gold
type_to_cols = {"factual": factual, "counterfactual": cf, "closed_book": cb, "random_context": cb}  # same for cb,rc


def exact_match(row, answer_col, prediction_col):
    """
    uses squad exact match function, compute_exact, to compare between gold answer and prediction
    """
    pred = str(row[prediction_col]) if str(row[prediction_col]) == str(row[prediction_col]) else ""  # if not nan
    answer = str(row[answer_col]) if row[answer_col] == row[answer_col] else ""  # not nan
    return ev.compute_exact(answer, pred)


def exact_match_unanswerable(row, prediction_col):
    """
    uses squad exact match function, compute_exact, to check if the prediction is "unanswerable"
    """
    pred = str(row[prediction_col]) if str(row[prediction_col]) == str(row[prediction_col]) else ""  # if not nan
    return ev.compute_exact("unanswerable", pred)


def f1_score(row, answer_col, prediction_col):
    """
    uses squad f1 score function, compute_f1, to compare between gold answer and prediction
    """
    pred = row[prediction_col] if row[prediction_col] == row[prediction_col] else ""
    answer = row[answer_col] if row[answer_col] == row[answer_col] else ""
    return ev.compute_f1(answer, pred)


def run_eval(path, answer_type, tables=True):
    """
    Gets inference outputs and calculates some metrics for the model, by comparing predictions to gold
    on several dimensions and computing means.
    """
    print("evaluation for {}".format(path))
    df = pd.read_csv(path)
    df = omit_tables(df) if not tables else df
    df = df[df["type"] == answer_type] if answer_type else df
    if len(df) == 0:
        return None
    df = df.rename(columns={"prediction": "model_output"})
    df = extract_model_answer_cols_from_new_format(df)
    df = compare_prediction_to_gold_per_row(df)
    results_df = get_total_score_means(df)
    save_results_and_evaluations(results_df, df, path, answer_type, tables)
    return results_df


def save_results_and_evaluations(results_df, df, path, answer_type, tables=True):
    """
    saves results to `stats` dir and evaluations (scores per row with the original data) in `eval_files`
    """
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)[:-4]
    stat_path = f"{dirname}/stats/{basename}.stat"
    eval_path = f"{dirname}/eval_files/{basename}_{answer_type}_eval.csv"
    if not tables:
        stat_path = f"{dirname}/stats_no_tables/{basename}.stat"
        eval_path = f"{dirname}/eval_files_no_tables/{basename}_{answer_type}_eval.csv"
    Path(stat_path).parent.mkdir(parents=True, exist_ok=True)
    Path(eval_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(stat_path, index=True)
    df.to_csv(eval_path, index=True, index_label="idx")


def compare_prediction_to_gold_per_row(df):
    """
    Compares prediction to gold on several dimensions that will serve as metrics for the model.
    """
    df = eval_new_format(df, "contextual_answer", "model_contextual")  # performance (against gold)
    df = eval_new_format(df, "parametric_answer", "model_parametric")  # performance (against gold)
    df = eval_new_format(df, "model_contextual", "model_parametric")  # disentanglement separation
    df = eval_new_format(df, "contextual_answer", "model_parametric")  # disentanglement p peekiness
    df = eval_new_format(df, "parametric_answer", "model_contextual")  # disentanglement c peekiness
    df = eval_unanswerable(df, prediction="model_contextual")
    df = eval_unanswerable(df, prediction="model_parametric")
    df = df[['Unnamed: 0', "question", "context", "type", "input", "output", "model_output", "contextual_answer",
             "model_contextual", "parametric_answer", "model_parametric",
             "contextual_answer_model_contextual_em_score", "contextual_answer_model_contextual_f1_score",
             "parametric_answer_model_parametric_em_score", "parametric_answer_model_parametric_f1_score",

             "model_contextual_model_parametric_em_score", "model_contextual_model_parametric_f1_score",
             "contextual_answer_model_parametric_em_score", "contextual_answer_model_parametric_f1_score",
             "parametric_answer_model_contextual_em_score", "parametric_answer_model_contextual_f1_score",
             "unanswerable_model_contextual_em_score", "unanswerable_model_parametric_em_score"]]
    return df


def eval_unanswerable(df, prediction):
    """
    Applies exact_match_unanswerable pre each row.
    """
    df = df.assign(temp_em=df.apply(exact_match_unanswerable, args=(prediction,), axis=1))
    df.rename(columns={"temp_em": "unanswerable_" + prediction + "_em_score"}, inplace=True)
    df = df.reset_index()
    return df


def convert_results_to_df(results):
    """
    Converts the results dictionary into a dataframe.
    """
    df = pd.DataFrame(pd.DataFrame({"names": NAMES, "results": results}))
    df = df.transpose()
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    return df


def extract_model_answer_cols_from_new_format(df):
    """
    Gets a model output (prediction) and splits it into model_contextual and model_parametric columns
    """
    df = df.assign(tmp=df['model_output'].str.split("contextual:").str[-1])
    df = df.assign(model_contextual=df['tmp'].str.split("parametric").str[0].str.strip())
    df = df.assign(model_parametric=df['tmp'].str.split("parametric:").str[1])
    df = df.assign(model_answerable=df['model_output'].str.split("contextual").str[0])
    df = df.assign(model_answerable=df['model_answerable'].str.split("parametric").str[0].str.strip())
    return df


def eval_new_format(df, gold_answer, prediction):
    """
    Applies exact_match and f1_score pre each row.
    Gets a df after extract_model_answer_cols_from_new_format execution.
    """
    assert ("model_contextual" in df.columns and "model_parametric" in
            df.columns and "model_answerable" in df.columns)
    df = df.assign(temp_f1=df.apply(f1_score, args=(gold_answer, prediction,), axis=1))
    df = df.assign(temp_em=df.apply(exact_match, args=(gold_answer, prediction,), axis=1))
    df.rename(columns={"temp_f1": gold_answer + "_" + prediction + "_f1_score"}, inplace=True)
    df.rename(columns={"temp_em": gold_answer + "_" + prediction + "_em_score"}, inplace=True)
    return df


def get_total_score_means(df):
    """
    Aggregate row results into one mean df score per each wanted comparison.
    """
    cc_em = df["contextual_answer_model_contextual_em_score"].mean()
    cc_f1 = df["contextual_answer_model_contextual_f1_score"].mean()
    pp_em = df["parametric_answer_model_parametric_em_score"].mean()
    pp_f1 = df["parametric_answer_model_parametric_f1_score"].mean()

    sep_cp_em = df["model_contextual_model_parametric_em_score"].mean()
    sep_cp_f1 = df["model_contextual_model_parametric_f1_score"].mean()
    cp_em = df["contextual_answer_model_parametric_em_score"].mean()
    cp_f1 = df["contextual_answer_model_parametric_f1_score"].mean()
    pc_em = df["parametric_answer_model_contextual_em_score"].mean()
    pc_f1 = df["parametric_answer_model_contextual_f1_score"].mean()

    uc_em = df["unanswerable_model_contextual_em_score"].mean()
    up_em = df["unanswerable_model_parametric_em_score"].mean()
    return convert_results_to_df([cc_em, cc_f1, pp_em, pp_f1, sep_cp_em,
                                  sep_cp_f1, cp_em, cp_f1, pc_em, pc_f1, uc_em, up_em])


def merge_and_save_results(results_dfs, path, tables=True):
    """
    merges results dfs with different answer_types into one, and saves them to a file in `merge` directory.
    """
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)[:-4]
    m = results_dfs[0]
    for i in range(1, len(results_dfs)):
        m = pd.merge(m, results_dfs[i], on='index')
    stat_path = f"{dirname}/merge/{basename}.stat"
    if not tables:
        stat_path = f"{dirname}/merge_no_tables/{basename}.stat"
    Path(stat_path).parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(stat_path)


def omit_tables(df):
    """
    Drops rows that have a table in their context.
    """
    df = df[~df["context"].str.startswith("<Table>")]
    df = df[~df["context"].str.startswith("<Tr>")]
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False, help="csv file with inference data for evaluation")
    args = parser.parse_args()
    tables = True
    results_dfs = []
    for answer_type in ["factual", "counterfactual", "closed_book", "random_context"]:
        results_dfs.append(run_eval(args.path, answer_type, tables=tables).reset_index())
    merge_and_save_results(results_dfs, args.path, tables=tables)
