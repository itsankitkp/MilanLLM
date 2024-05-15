from ..data_store.fetch import *
from ..text.vocab import *
from ..core.base import *
from ..models.core import Classifier
from torch.nn import functional as F


class MTEngHin(DataModule):

    def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128):
        super(MTEngHin, self).__init__()
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_train = num_train
        self.num_val = num_val
        self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
            self._download()
        )

    def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
        def _build_array(sentences, vocab, is_tgt=False):
            pad_or_trim = lambda seq, t: (
                seq[:t] if len(seq) > t else seq + ["<pad>"] * (t - len(seq))
            )
            sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
            if is_tgt:
                sentences = [["<bos>"] + s for s in sentences]
            if vocab is None:
                vocab = Vocab(sentences, min_freq=1)
            array = torch.tensor([vocab[s] for s in sentences])
            valid_len = (array != vocab["<pad>"]).type(torch.int32).sum(1)
            return array, vocab, valid_len

        src, tgt = self._tokenize(
            self._preprocess(raw_text), self.num_train + self.num_val
        )

        src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)
        tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, True)
        return (
            (src_array, tgt_array[:, :-1], src_valid_len, tgt_array[:, 1:]),
            src_vocab,
            tgt_vocab,
        )

    def _download(self):

        txt = read_file(
            "G:\AI\MilanLLM\MilanLLM\src\milanllm\data_store\hin-eng\hin.txt"
        )
        return txt

    def _preprocess(self, text):
        # Replace non-breaking space with space
        text = text.replace("\u202f", " ").replace("\xa0", " ")
        # Insert space between words and punctuation marks
        no_space = lambda char, prev_char: char in ",.!?" and prev_char != " "
        out = [
            " " + char if i > 0 and no_space(char, text[i - 1]) else char
            for i, char in enumerate(text.lower())
        ]
        return "".join(out)

    def _tokenize(self, text, max_examples=None):
        src, tgt = [], []

        for i, line in enumerate(text.split("\n")):
            if max_examples and i > max_examples:
                break
            parts = line.split("\t")

                # Skip empty tokens
            src.append([t for t in f"{parts[0]} <eos>".split(" ") if t])
            tgt.append([t for t in f"{parts[1]} <eos>".split(" ") if t])
        return src, tgt

    def get_dataloader(self, train):
        idx = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader(self.arrays, train, self.batch_size, idx)
    
    def build(self, src_sentences, tgt_sentences):
        raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(
        src_sentences, tgt_sentences)])

        arrays, _, _ = self._build_arrays(raw_text, self.src_vocab, self.tgt_vocab)
        return arrays