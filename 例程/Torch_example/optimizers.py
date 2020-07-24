#!/usr/bin/env python3
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))

#plt.scatter(x.numpy(), y.numpy())
#plt.show()

class Net(torch.nn.Module):
    def __init__(self, nFeature=1, nHidden=20, nOutput=1):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(nFeature, nHidden)
        self.predict = torch.nn.Linear(nHidden, nOutput)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

if __name__ == '__main__':
    torchDataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(dataset=torchDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    netSGD = Net()
    netMomentum = Net()
    netRMSprop = Net()
    netAdam = Net()
    nets = [netSGD, netMomentum, netRMSprop, netAdam]

    optSGD = torch.optim.SGD(netSGD.parameters(), lr=LR)
    optMomentum = torch.optim.SGD(netMomentum.parameters(), lr=LR, momentum=0.8)
    optRMSProp = torch.optim.RMSprop(netRMSprop.parameters(), lr=LR, alpha=0.9)
    optAdam = torch.optim.Adam(netAdam.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers = [optSGD, optMomentum, optRMSProp, optAdam]

    lossFunc = torch.nn.MSELoss()
    lossHis = [[], [], [], []]

    for epoch in range(EPOCH):
        print('[+] Epoch :', epoch)
        for step, (bX, bY) in enumerate(loader):
            for net, opt, lHis in zip(nets, optimizers, lossHis):
                output = net(bX)
                loss = lossFunc(output, bY)
                opt.zero_grad()
                loss.backward()
                opt.step()
                lHis.append(loss.data.numpy())

    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for i, lHis in enumerate(lossHis):
        plt.plot(lHis, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()

