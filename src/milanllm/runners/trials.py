import torch
from ..models.core import LinearRegression  

from ..core.base import Trainer, DataModule
from ..core.utils import plot


class Data(DataModule):
    def __init__(self, batch_size=16, T=1000, num_train=600, tau=4):
        self.batch_size = batch_size
        self.T = T
        self.num_train = num_train
        self.tau = tau
        self.time = torch.arange(1, T + 1, dtype=torch.float32)
        self.x = torch.sin(0.01 * self.time) + torch.randn(T) * 0.2
    def get_dataloader(self, train):
        features = [self.x[i : self.T-self.tau+i] for i in range(self.tau)]
        self.features = torch.stack(features, 1)
        self.labels = self.x[self.tau:].reshape((-1, 1))
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.features, self.labels], train,32, i)
    

def main():
    data = Data()
    model = LinearRegression(lr=0.01)
    trainer = Trainer(max_epochs=5)
    trainer.fit(model, data)
    onestep_preds = model(data.features).detach().numpy()
    multistep_preds = torch.zeros(data.T)
    multistep_preds[:] = data.x
    # for i in range(data.num_train + data.tau, data.T):
    #     multistep_preds[i] = model(multistep_preds[i - data.tau:i].reshape((1, -1)))
    for i in range(data.num_train + data.tau, data.T):
        multistep_preds[i] = model(multistep_preds[i - data.tau:i].reshape((1, -1)))
    multistep_preds = multistep_preds.detach().numpy()
    plot([data.time[data.tau:], data.time[data.num_train+data.tau:]],
    [onestep_preds, multistep_preds[data.num_train+data.tau:]], 'time',
    'x', legend=['1-step preds', 'multistep preds'], figsize=(6, 3))
