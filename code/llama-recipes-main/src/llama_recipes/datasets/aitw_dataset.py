# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/jfleg
# For download and preparation see: recipes/ft_datasets/grammar_dataset/grammar_dataset_process.ipynb


from datasets import load_dataset
from pathlib import Path

from torch.utils.data import Dataset


class aitw(Dataset):
    def __init__(
        self,
        tokenizer,
        json_name=None,
    ):

        try:
            self.dataset = load_dataset(
                "json",
                data_files={"train": [json_name]}
            )
        except Exception as e:
            print("Loading of aitw dataset failed! Please see recipes/ft_datasets/grammar_dataset/grammar_dataset_process.ipynb for details on how to download the dataset.")
            raise e

        # self.dataset = load_dataset("wikihow", "all", data_dir="data/", split=type_path)
        # if num_samples:
        #    self.dataset = self.dataset.select(list(range(0, num_samples)))
        self.tokenizer = tokenizer
        self.print_text = False  # print_text

    def __len__(self):
        return self.dataset["train"].shape[0]

    def convert_to_features(self, example_batch):

        # Create prompt and tokenize contexts and questions

        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch["text"]))

        input_ = example_batch["instruction"]
        target_ = example_batch["output"]

        prompt = input_
        prompt_ids = self.tokenizer.encode(self.tokenizer.bos_token + prompt, add_special_tokens=False)
        label_ids = self.tokenizer.encode(target_ + self.tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt_ids + label_ids,
            "attention_mask": [1] * len(prompt_ids + label_ids),
            "labels": [-100] * len(prompt_ids) + label_ids
        }

        return sample

    def __getitem__(self, index):
        return self.convert_to_features(self.dataset["train"][int(index)])


def get_dataset(
    dataset_config, tokenizer, json_name=None
):
    """cover function for handling loading the working dataset"""
    """dataset loading"""
    if json_name is None:
        currPath = Path.cwd() / "datasets_aitw" / "aitw_train.json"
        print(f"Loading dataset {currPath}")
        json_name = str(currPath)
    dataset = aitw(
        tokenizer=tokenizer,
        json_name=json_name,
    )

    return dataset
