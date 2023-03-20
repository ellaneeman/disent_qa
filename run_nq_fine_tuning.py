"""
This code is based on an implementation of T5 fine-tuning for question answering:
https://colab.research.google.com/drive/1WXLtGQmYyrMi484ox9R5ZkJe_4Vm3fny
"""

import argparse
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import json
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from pytorch_lightning.loggers import WandbLogger

torch.cuda.empty_cache()
pl.seed_everything(42)

global t5_tokenizer
global trained_model
global wandb_logger

class NaturalQuestionsDataset(Dataset):
    def __init__(self, data, tokenizer, source_max_token_len=396, target_max_token_len=32):
        self.tokenizer = tokenizer
        self.data = data  # pandas DataFrame object
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        # print(data_row)
        question = data_row['question']
        context = data_row['context']
        answer_text = data_row['answer_text']
        source_encoding = self.tokenizer(question, context, max_length=self.source_max_token_len, padding="max_length",
                                         truncation='only_second', return_attention_mask=True, add_special_tokens=True,
                                         return_tensors="pt")
        target_encoding = self.tokenizer(answer_text, max_length=self.target_max_token_len, padding="max_length",
                                         truncation=True, return_attention_mask=True, add_special_tokens=True,
                                         return_tensors="pt")

        input_ids = source_encoding["input_ids"].flatten()
        attention_mask = source_encoding["attention_mask"].flatten()
        labels = target_encoding["input_ids"].flatten()
        labels[labels == 0] = -100

        return {"question": question, "context": context, "answer_text": answer_text, "input_ids": input_ids,
                "attention_mask": attention_mask, "labels": labels}


class NaturalQuestionsModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, tokenizer, batch_size=8, source_max_token_len=396, target_max_token_len=32):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def setup(self):
        self.train_dataset = NaturalQuestionsDataset(self.train_df, self.tokenizer,
                                                     self.source_max_token_len, self.target_max_token_len)
        self.test_dataset = NaturalQuestionsDataset(self.test_df, self.tokenizer,
                                                    self.source_max_token_len, self.target_max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=True, num_workers=4)


class NaturalQuestionsModel(pl.LightningModule):
    def __init__(self, model_name, learning_rate):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir="models/", return_dict=True)
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        print(self.parameters())
        return AdamW(self.parameters(), lr=self.learning_rate)


class DisentQADataset(Dataset):
    def __init__(self, data, tokenizer, source_max_token_len=396, target_max_token_len=32):
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        question_and_context = data_row['input']
        answer_text = data_row['output']
        source_encoding = self.tokenizer(question_and_context, max_length=self.source_max_token_len,
                                         padding="max_length",
                                         truncation=True, return_attention_mask=True, add_special_tokens=True,
                                         return_tensors="pt")
        if data_row['type'] == "unanswerable":
            target_encoding = self.tokenizer(answer_text, max_length=self.target_max_token_len, padding="max_length",
                                             truncation=True, return_attention_mask=True,
                                             add_special_tokens=False,  # skip eos sign
                                             return_tensors="pt")
        else:
            target_encoding = self.tokenizer(answer_text, max_length=self.target_max_token_len, padding="max_length",
                                             truncation=True, return_attention_mask=True, add_special_tokens=True,
                                             return_tensors="pt")

        input_ids = source_encoding["input_ids"].flatten()  # question_and_context
        attention_mask = source_encoding["attention_mask"].flatten()  # question_and_context
        labels = target_encoding["input_ids"].flatten()  # "answer_text"
        labels[labels == 0] = -100

        return {"question_and_context": question_and_context, "answer_text": answer_text, "input_ids": input_ids,
                "attention_mask": attention_mask, "labels": labels}


#
class DisentQAModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, tokenizer, batch_size=8, source_max_token_len=396, target_max_token_len=32):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def setup(self):
        self.train_dataset = DisentQADataset(self.train_df, self.tokenizer,
                                             self.source_max_token_len, self.target_max_token_len)
        self.test_dataset = DisentQADataset(self.test_df, self.tokenizer,
                                            self.source_max_token_len, self.target_max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=True, num_workers=4)


def get_data_for_fine_tuning(path, n_epochs, learning_rate,
                             s_max_token_len=396, batch_size=8):
    """
    reads csv data from file and concatenates the checkpoint_filename from its params.
    """
    train_df = pd.read_csv(path.split(".")[0] + "_train_split.csv")
    val_df = pd.read_csv(path.split(".")[0] + "_val_split.csv")
    dataset_name = path.split("-")[-1].split(".csv")[0]
    checkpoint_filename = "disent_qa_" + dataset_name + "_" + str(len(train_df)) + "_b" + str(
        batch_size) + "_lr" + str(learning_rate) + "_e" + str(n_epochs) + "_smax" + str(s_max_token_len)
    return train_df, val_df, checkpoint_filename


def fine_tune(train_df, val_df, checkpoint_filename, module, model_name, source_max_token_len, target_max_token_len,
              batch_size, n_epochs, checkpoints_dirpath, learning_rate):
    """
    Fine-tunes a pre-trained T5 model on the train_df and the val_df. Saves the checkpoints to a specified directory.
    """
    data_module = module(train_df, val_df, t5_tokenizer,
                         source_max_token_len=source_max_token_len,
                         target_max_token_len=target_max_token_len,
                         batch_size=batch_size)
    data_module.setup()
    model = NaturalQuestionsModel(model_name=model_name, learning_rate=learning_rate)
    checkpoint_filename = model_name + "_" + checkpoint_filename
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_dirpath,
                                          filename=checkpoint_filename, save_top_k=n_epochs,
                                          verbose=True, monitor="val_loss", mode="min")
    trainer = pl.Trainer(checkpoint_callback=checkpoint_callback, max_epochs=n_epochs, gpus=1,
                         progress_bar_refresh_rate=30, logger=wandb_logger)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    f = open(os.path.dirname(__file__) + "/config.json")
    config = json.load(f)["fine_tune"]
    os.environ["WANDB_API_KEY"] = config["wandb_api_key"]
    wandb_logger = WandbLogger(project="disent_qa")

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False,
                        default="data/natural_questions/v10-simplified_simplified-nq-train_full_no_dec",
                        help="data path prefix for training/val data")
    args = parser.parse_args()

    model_name = config['model_name']
    t5_tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir="models/")
    state = args.state
    path = args.path
    train_df, val_df, checkpoint_filename = get_data_for_fine_tuning(path=path,
                                                                     s_max_token_len=config["source_max_token_len"],
                                                                     batch_size=config["batch_size"],
                                                                     n_epochs=config["n_epochs"],
                                                                     learning_rate=config["learning_rate"])
    fine_tune(train_df, val_df, checkpoint_filename, DisentQAModule, model_name,
              source_max_token_len=config["source_max_token_len"],
              target_max_token_len=config["target_max_token_len"],
              batch_size=config["batch_size"],
              n_epochs=config["n_epochs"],
              checkpoints_dirpath=config["checkpoints_dirpath"])
