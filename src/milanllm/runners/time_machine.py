from ..data_store.fetch import *
from ..text.vocab import *
from ..core.base import *
from ..models.core import Classifier
from torch.nn import functional as F


class RNN(Module): 

    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_inputs = num_inputs
        self.rnn = nn.RNN(num_inputs, num_hiddens)
    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)

class TimeMachine(DataModule):
    def build(self, raw_text=None, vocab=None):
        txt = get_corpus_from_disk('G:\AI\MilanLLM\MilanLLM\src\milanllm\data_store\corpus.txt')
        tokens = list(txt.lower())
        if vocab is None: vocab = Vocab(tokens)
        corpus = [vocab[token] for token in tokens]
        return corpus, vocab
    
    def __init__(self, batch_size, num_steps, num_train=10000, num_val=5000):
        super().__init__()

        self.batch_size, self.num_steps = batch_size, num_steps
        self.num_train, self.num_val = num_train, num_val

        self.corpus, self.vocab = self.build()
        self.array = torch.tensor([self.corpus[i:i+num_steps+1]   for i in range(len(self.corpus)-num_steps)])
        self.X, self.Y = self.array[:,:-1], self.array[:,1:]
        
    def get_dataloader(self, train):
        idx = slice(0, self.num_train    ) if train else slice(
        self.num_train, self.num_train + self.num_val)
        return self.get_tensorloader([self.X, self.Y], train,self.batch_size, idx)
    

class TimeMachineLLM(Classifier):
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.vocab_size = vocab_size
        self.rnn = rnn
        self.lr=lr
        self.init_params()


    def init_params(self):
        self.linear = nn.LazyLinear(self.vocab_size)
    def output_layer(self, hiddens):
            return self.linear(hiddens).swapaxes(0, 1)
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', torch.exp(l), train=True)
        return l
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', torch.exp(l), train=False)

    def one_hot(self, X):
        # Output shape: (num_steps, batch_size, vocab_size)
        return F.one_hot(X.T, self.vocab_size).type(torch.float32)
    
 
    
    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state)
        return self.output_layer(rnn_outputs)
    
    def predict(self, prefix, num_preds, vocab, device='cuda:0'):
        state, outputs = None, [vocab[prefix[0]]]
        for i in range(len(prefix) + num_preds - 1):
            X = torch.tensor([[outputs[-1]]], device=device)
            embs = self.one_hot(X)
            rnn_outputs, state = self.rnn(embs, state)
            if i < len(prefix) - 1: # Warm-up period
                outputs.append(vocab[prefix[i + 1]])
            else: # Predict num_preds steps
                Y = self.output_layer(rnn_outputs)
                outputs.append(int(Y.argmax(axis=2).reshape(1)))
        return ''.join([vocab.idx_to_token[i] for i in outputs])
    
class LSTM(RNN):
    def __init__(self, num_inputs, num_hiddens):
        Module.__init__(self)
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.rnn = nn.LSTM(num_inputs, num_hiddens)
    def forward(self, inputs, H_C=None):
        return self.rnn(inputs, H_C)
class GRU(RNN): #@save

    def __init__(self, num_inputs, num_hiddens, num_layers, dropout=0):
        Module.__init__(self)
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.rnn = nn.GRU(num_inputs, num_hiddens, num_layers,
        dropout=dropout)

def main():
    data = TimeMachine(batch_size=1024, num_steps=32)
    rnn = RNN(num_inputs=len(data.vocab), num_hiddens=1024)
    model = TimeMachineLLM(rnn, len(data.vocab), lr=0.1)
    trainer = Trainer(max_epochs=100)
    trainer.fit(model, data)
    return model, data

def lstm_main():
    data = TimeMachine(batch_size=1024, num_steps=32)
    lstm = LSTM(num_inputs=len(data.vocab), num_hiddens=32)
    model = TimeMachineLLM(lstm, len(data.vocab), lr=4)
    trainer = Trainer(max_epochs=100)
    trainer.fit(model, data)
    return model, data

def gru_main():
    data = TimeMachine(batch_size=1024, num_steps=32)
    gru = GRU(num_inputs=len(data.vocab), num_hiddens=32, num_layers=2)
    model = TimeMachineLLM(gru, len(data.vocab), lr=1)
    trainer = Trainer(max_epochs=100)
    trainer.fit(model, data)
    return model, data