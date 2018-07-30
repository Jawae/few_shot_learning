import random
import torch
import torch.optim as optim


def chunks(n, *args):
    """Yield successive n-sized chunks from l."""
    endpoints = []
    start = 0
    for stop in range(0, len(args[0]), n):
        if stop - start > 0:
            endpoints.append((start, stop))
            start = stop
    random.shuffle(endpoints)
    for start, stop in endpoints:
        yield [a[start: stop] for a in args]


class Wrapper(object):
    def __init__(self, model, device, log_interval=100, epochs=1000, batchsize=1024):
        self.batchsize = batchsize
        self.epochs = epochs
        self.device = device
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=1e-2)
        self.log_interval = log_interval

    def fit(self, *args):

        self.model.train()
        for epoch in range(self.epochs):
            total = 0.0

            for itr, datas in enumerate(chunks(self.batchsize, *args)):
                datas = [torch.from_numpy(data).to(self.device) for data in datas]
                self.optimizer.zero_grad()
                loss = self.model(*datas)
                loss.backward()
                self.optimizer.step()
                total += loss.item()

            print('Train Epoch: {} \tLoss: {:.6e}'.format(epoch, total / (len(args[0]) * 1.0)))
