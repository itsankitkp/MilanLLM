import os
import re
import tarfile

import requests
import zipfile

import torch
from tqdm import tqdm
from milanllm.core.utils import tokenize, set_figsize, plt, truncate_pad, load_array
from milanllm.text.vocab import Vocab

dir_path = os.path.dirname(os.path.realpath(__file__))


def read_imdb(is_train):
    """Read the IMDb review dataset text sequences and labels."""
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    data_dir = f"{dir_path}/data"
    path = f"{dir_path}/data/aclImdb_v1.tar.gz"
    imdb_fp = f"{data_dir}/aclImdb"

    if not os.path.exists(imdb_fp):

        data = requests.get(url)

        with open(path, "wb") as f:
            f.write(data.content)
        fp = tarfile.open(path, "r")
        fp.extractall(data_dir)

    data, labels = [], []
    for label in ("pos", "neg"):
        folder_name = os.path.join(imdb_fp, "train" if is_train else "test", label)
        for file in tqdm(os.listdir(folder_name)):
            with open(os.path.join(folder_name, file), "rb") as f:
                review = f.read().decode("utf-8").replace("\n", "")
                data.append(review)
                labels.append(1 if label == "pos" else 0)
    return data, labels


def load_data_imdb(batch_size, num_steps=500):

    train_data = read_imdb(True)
    test_data = read_imdb(False)
    train_tokens = tokenize(train_data[0], token="word")
    test_tokens = tokenize(test_data[0], token="word")
    vocab = Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor(
        [truncate_pad(vocab[line], num_steps, vocab["<pad>"]) for line in train_tokens]
    )
    test_features = torch.tensor(
        [truncate_pad(vocab[line], num_steps, vocab["<pad>"]) for line in test_tokens]
    )
    train_iter = load_array((train_features, torch.tensor(train_data[1])), batch_size)
    test_iter = load_array(
        (test_features, torch.tensor(test_data[1])), batch_size, is_train=False
    )
    return train_iter, test_iter, vocab


def get_snli():
    url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    data_dir = f"{dir_path}/data/snli_1.0"
    path = f"{dir_path}/data/snli_1.0.zip"
    if not os.path.exists(data_dir):
        print("Downloading SNLI dataset...")

        data = requests.get(url)

        with open(path, "wb") as f:
            f.write(data.content)
        fp = zipfile.ZipFile(path, "r")

        print("Extracting SNLI dataset...")

        fp.extractall(data_dir)

    return data_dir


class SNLIDataset(torch.utils.data.Dataset):
    """A customized dataset to load the SNLI dataset.

    Defined in :numref:`sec_natural-language-inference-and-dataset`"""

    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = tokenize(dataset[0])
        all_hypothesis_tokens = tokenize(dataset[1])
        if vocab is None:
            self.vocab = Vocab(
                all_premise_tokens + all_hypothesis_tokens,
                min_freq=5,
                reserved_tokens=["<pad>"],
            )
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print("read " + str(len(self.premises)) + " examples")

    def _pad(self, lines):
        return torch.tensor(
            [
                truncate_pad(self.vocab[line], self.num_steps, self.vocab["<pad>"])
                for line in lines
            ]
        )

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)


def read_snli(data_dir, is_train):

    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub("\\(", "", s)
        s = re.sub("\\)", "", s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub("\\s{2,}", " ", s)
        return s.strip()

    label_set = {"entailment": 0, "contradiction": 1, "neutral": 2}
    file_name = os.path.join(
        data_dir, "snli_1.0_train.txt" if is_train else "snli_1.0_test.txt"
    )
    with open(file_name, "r") as f:
        rows = [row.split("\t") for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels


def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary.

    Defined in :numref:`sec_natural-language-inference-and-dataset`"""

    data_dir = get_snli()
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(
        train_set,
        batch_size,
        shuffle=True,
    )
    test_iter = torch.utils.data.DataLoader(
        test_set,
        batch_size,
        shuffle=False,
    )
    return train_iter, test_iter, train_set.vocab
