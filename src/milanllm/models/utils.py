


import os
import tarfile

import requests
import zipfile

dir_path = os.path.dirname(os.path.realpath(__file__))


def read_imdb( is_train):
    """Read the IMDb review dataset text sequences and labels."""
    url="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    data_dir = f"{dir_path}/data"
    path=f"{dir_path}/data/aclImdb_v1.tar.gz"
    if not os.path.exists(path):
        data = requests.get(url)
        
        with open(path, "wb") as f:
            f.write(data.content)
    fp = tarfile.open(path, 'r')
    fp.extractall(data_dir)

    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
    label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels