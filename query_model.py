import json
import time
import re
from run_nq_fine_tuning import NaturalQuestionsModel
import os
from pathlib import Path
import torch
from transformers import T5Tokenizer
import pandas as pd
import argparse

print("torch.cuda.is_available(): {}".format(torch.cuda.is_available()))
torch.cuda.empty_cache()

answer_types = {"f": "factual", "cf": "counterfactual", "cb": "closed_book", "rc": "random_context", "all": None}


def df_chunks_generator(lst, n, input_col):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        small_df = lst.iloc[i:i + n]
        input_items = small_df[input_col].to_list()
        yield input_items


def generate_answer(question_and_context, input_max_length, output_max_length, repetition_penalty, length_penalty,
                    num_beams):
    """
    pass a batch of question_and_context together (in the format question: <question>\ncontext: <context>) to the model
    to generate an answer batch
    """
    source_encoding = tokenizer(question_and_context, max_length=input_max_length,
                                padding="max_length", truncation=True, return_attention_mask=True,
                                add_special_tokens=True, return_tensors="pt")
    generated_ids = model.model.generate(input_ids=source_encoding["input_ids"],
                                         attention_mask=source_encoding["attention_mask"], num_beams=num_beams,
                                         max_length=output_max_length, repetition_penalty=repetition_penalty,
                                         length_penalty=length_penalty,
                                         early_stopping=True, use_cache=True)
    preds = [tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for
             generated_id in generated_ids]
    return preds


def query_model(path, checkpoint_name, config_dict, answer_type=""):
    """
    gets a path to an evaluation set and an answer type and queries the global loaded model,
    then saves the results to a csv file.
    """
    df = pd.read_csv(path)
    if answer_type:
        df = df[df["type"] == answer_type]
    vals = []
    for batch in df_chunks_generator(df, config_dict["batch_size"], "input"):
        model_output = generate_answer(batch,  # pass in batches
                                       input_max_length=config_dict['input_max_length'],
                                       output_max_length=config_dict['output_max_length'],
                                       repetition_penalty=config_dict['repetition_penalty'],
                                       length_penalty=config_dict['length_penalty'],
                                       num_beams=config_dict['num_beams'])
        vals.extend(model_output)
    # add the model answers as a col to the df
    df = df.assign(model_output=vals)
    save_results_to_file(df, answer_type, checkpoint_name, config_dict, path)


def save_results_to_file(df, answer_type, checkpoint_name, config_dict, original_path):
    """
    extracts information from the original_path, checkpoint_name etc. to save the results file.
    """
    checkpoint_name = re.sub(".*checkpoints/", "", checkpoint_name)
    vlen = len(df[df["type"] == "factual"] if not answer_type else df[df["type"] == answer_type]),
    test_set_name = original_path.split("/")[-1].split(".csv")[0].split("-")[-1]
    results_dir, model_name = config_dict['results_dir'], config_dict['model_name']
    file_name = f"{results_dir}/{model_name}/{checkpoint_name}_{test_set_name}_len{vlen}_{answer_type}_inference.csv"
    Path(os.path.dirname(file_name)).mkdir(parents=True, exist_ok=True)
    df.to_csv(file_name)


if __name__ == '__main__':
    global tokenizer, model
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False, help="Path to a csv file")
    parser.add_argument("--checkpoint_name", type=str, required=False, help="checkpoint_name - path with .ckpt")
    parser.add_argument("--answer_type", type=str, required=False, help="f, cf, rc, or cb.")
    args = parser.parse_args()

    query_answer_type = answer_types[args.answer_type] if args.answer_type else ""
    f = open(os.path.dirname(__file__) + "/config.json")
    config = json.load(f)["query_model"]
    tic = time.perf_counter()
    print("loading the {} model".format(args.checkpoint_name))
    tokenizer = T5Tokenizer.from_pretrained(config["model_name"], cache_dir="models/")
    # load checkpoint from a path in the form of "checkpoints/" + checkpoint_names[args.model_name] + ".ckpt")
    trained_model = NaturalQuestionsModel.load_from_checkpoint(args.checkpoint_name)
    trained_model.freeze()
    model = trained_model
    query_model(path=args.path,
                checkpoint_name=args.checkpoint_name,
                answer_type=query_answer_type,
                config_dict=config)
    toc = time.perf_counter()
    print(f"Running time: {toc - tic:0.4f} seconds")
