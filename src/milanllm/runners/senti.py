from milanllm.models.utils import read_imdb

from milanllm.core.utils import tokenize, set_figsize, plt,truncate_pad, load_array

from milanllm.text.vocab import Vocab
import torch


def main():
    train_data = read_imdb(is_train=True)
    print("# trainings:", len(train_data[0]))
    for x, y in zip(train_data[0][:3], train_data[1][:3]):
        print("label:", y, "review:", x[:60])
    train_tokens = tokenize(train_data[0], token="word")
    vocab = Vocab(train_tokens, min_freq=5, reserved_tokens=["<pad>"])
    set_figsize()
    plt.xlabel("# tokens per review")
    plt.ylabel("count")
    plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50))
    plt.show()
    num_steps = 500 # sequence length
    train_features = torch.tensor([truncate_pad(
    vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    print(train_features.shape)
    train_iter = load_array((train_features, torch.tensor(train_data[1])), 64)
    for X, y in train_iter:
        print('X:', X.shape, ', y:', y.shape)
        break
    print('# batches:', len(train_iter))