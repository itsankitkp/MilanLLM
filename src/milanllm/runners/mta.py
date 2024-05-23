from ..core.utils import bleu, try_gpu
from ..data_store.fetch import *
from ..text.vocab import *
from ..core.base import *
from ..models.core import *
from ..core.utils import *
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
        raw_text = "\n".join(
            [src + "\t" + tgt for src, tgt in zip(src_sentences, tgt_sentences)]
        )

        arrays, _, _ = self._build_arrays(raw_text, self.src_vocab, self.tgt_vocab)
        return arrays


def nexa():
    data = MTEngHin(batch_size=128)
    embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
    encoder = Seq2SeqEncoder(
        len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout
    )
    decoder = Seq2SeqDecoder(
        len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout
    )
    model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab["<pad>"], lr=0.005)
    trainer = Trainer(max_epochs=30, gradient_clip_val=1)
    trainer.fit(model, data)

    engs = ["I forgot.", "Let him in.", "Unbelievable!", "This is my dog."]
    hindi = ["मैं भूल गया।", "उसे अंदर भेजो।", "अविश्वसनीय!", "यह मेरा कुत्ता है।"]
    preds, _ = model.predict_step(data.build(engs, hindi), try_gpu(), data.num_steps)
    for en, fr, p in zip(engs, hindi, preds):
        translation = []
        for token in data.tgt_vocab.to_tokens(p):
            if token == "<eos>":
                break
            translation.append(token)
            print(
                f"{en} => {translation}, bleu,"
                f'{bleu(" ".join(translation), fr, k=2):.3f}'
            )


def att():
    data = MTEngHin(batch_size=128)
    embed_size, num_hiddens, num_layers, dropout = 256 * 4, 256 * 4, 2 * 4, 0.2
    encoder = Seq2SeqEncoder(
        len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout
    )
    decoder = Seq2SeqAttentionDecoder(
        len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout
    )
    model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab["<pad>"], lr=0.005)
    trainer = Trainer(max_epochs=100, gradient_clip_val=1)
    trainer.fit(model, data)

    engs = ["I forgot.", "Let him in.", "Unbelievable!", "This is my dog."]
    hindi = ["मैं भूल गया।", "उसे अंदर भेजो।", "अविश्वसनीय!", "यह मेरा कुत्ता है।"]
    preds, _ = model.predict_step(data.build(engs, hindi), try_gpu(), data.num_steps)
    for en, fr, p in zip(engs, hindi, preds):
        translation = []
        for token in data.tgt_vocab.to_tokens(p):
            if token == "<eos>":
                break
            translation.append(token)
            print(
                f"{en} => {translation}, bleu,"
                f'{bleu(" ".join(translation), fr, k=2):.3f}'
            )
    _, dec_attention_weights = model.predict_step(
        data.build([engs[-1]], [hindi[-1]]), try_gpu(), data.num_steps, True
    )
    attention_weights = torch.cat([step[0][0][0] for step in dec_attention_weights], 0)
    attention_weights = attention_weights.reshape((1, 1, -1, data.num_steps))
    show_heatmaps(
        attention_weights[:, :, :, : len(engs[-1].split()) + 1].cpu(),
        xlabel="Key positions",
        ylabel="Query positions",
    )
    return model, data


def trans():
    data = MTEngHin(batch_size=128)
    num_hiddens, num_blks, dropout = 256, 2, 0.2
    ffn_num_hiddens, num_heads = 64, 4
    encoder = TransformerEncoder(
        len(data.src_vocab), num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout
    )
    decoder = TransformerDecoder(
        len(data.tgt_vocab), num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout
    )
    model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab["<pad>"], lr=0.001)
    trainer = Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
    trainer.fit(model, data)
    engs = ["I forgot.", "Let him in.", "Unbelievable!", "This is my dog."]
    hindi = ["मैं भूल गया।", "उसे अंदर भेजो।", "अविश्वसनीय!", "यह मेरा कुत्ता है।"]
    preds, _ = model.predict_step(data.build(engs, hindi), try_gpu(), data.num_steps)
    for en, fr, p in zip(engs, hindi, preds):
        translation = []
        for token in data.tgt_vocab.to_tokens(p):
            if token == "<eos>":
                break
            translation.append(token)
            print(
                f"{en} => {translation}, bleu,"
                f'{bleu(" ".join(translation), fr, k=2):.3f}'
            )
    _, dec_attention_weights = model.predict_step(
        data.build([engs[-1]], [hindi[-1]]), try_gpu(), data.num_steps, True
    )
    enc_attention_weights = torch.cat(model.encoder.attention_weights, 0)
    shape = (num_blks, num_heads, -1, data.num_steps)
    enc_attention_weights = enc_attention_weights.reshape(shape)
    show_heatmaps(
        enc_attention_weights.cpu(),
        xlabel="Key positions",
        ylabel="Query positions",
        titles=["Head %d" % i for i in range(1, 5)],
        figsize=(7, 3.5),
    )
    return model, data
