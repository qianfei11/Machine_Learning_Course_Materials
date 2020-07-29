#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import init

N_SAMPLES = 2000
BATCH_SIZE = 64
EPOCH = 12
LR = 0.03
N_HIDDEN = 8
ACTIVATION = torch.tanh
B_INIT = -0.2

xTraining = np.linspace(-7, 10, N_SAMPLES)[:, np.newaxis]
noise = np.random.normal(0, 2, xTraining.shape)
yTraining = np.square(xTraining) - 5 + noise

xTesting = np.linspace(-7, 10, 200)[:, np.newaxis]
noise = np.random.normal(0, 2, xTesting.shape)
yTesting = np.square(xTesting) - 5 + noise

xTraining, yTraining = torch.from_numpy(xTraining).float(), torch.from_numpy(yTraining).float()
xTesting, yTesting = torch.from_numpy(xTesting).float(), torch.from_numpy(yTesting).float()

trainDataset = Data.TensorDataset(xTraining, yTraining)
trainLoader = Data.DataLoader(dataset=trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

#plt.scatter(xTraining.numpy(), yTraining.numpy(), c='#FF9359', s=50, alpha=0.2, label='train')
#plt.legend(loc='upper left')

class Net(nn.Module):
    def __init__(self, batchNormalization=False):
        super(Net, self).__init__()
        self.doBn = batchNormalization
        self.fcs = []
        self.bns = []
        self.bnInput = nn.BatchNorm1d(1, momentum=0.5)

        for i in range(N_HIDDEN):
            inputSize = 1 if i == 0 else 10
            fc = nn.Linear(inputSize, 10)
            setattr(self, 'fc%i' % i, fc)
            self.setInit(fc)
            self.fcs.append(fc)
            if self.doBn:
                bn = nn.BatchNorm1d(10, momentum=0.5)
                setattr(self, 'bn%i' % i, bn)
                self.bns.append(bn)
        self.predict = nn.Linear(10, 1)
        self.setInit(self.predict)

    def setInit(self, layer):
        init.normal_(layer.weight, mean=0., std=1.)
        init.constant_(layer.bias, B_INIT)

    def forward(self, x):
        preActivation = [x]
        if self.doBn:
            x = self.bnInput(x)
        layerInput = [x]
        for i in range(N_HIDDEN):
            x = self.fcs[i](x)
            preActivation.append(x)
            if self.doBn:
                x = self.bns[i](x)
            x = ACTIVATION(x)
            layerInput.append(x)
        out = self.predict(x)
        return out, layerInput, preActivation

def plotHistogram(l_in, l_in_bn, pre_ac, pre_ac_bn):
    for i, (ax_pa, ax_pa_bn, ax, ax_bn) in enumerate(zip(axs[0, :], axs[1, :], axs[2, :], axs[3, :])):
        [a.clear() for a in [ax_pa, ax_pa_bn, ax, ax_bn]]
        if i == 0:
            p_range = (-7, 10)
            the_range = (-7, 10)
        else:
            p_range = (-4, 4)
            the_range = (-1, 1)
        ax_pa.set_title('L' + str(i))
        ax_pa.hist(pre_ac[i].data.numpy().ravel(), bins=10, range=p_range, color='#FF9359', alpha=0.5)
        ax_pa_bn.hist(pre_ac_bn[i].data.numpy().ravel(), bins=10, range=p_range, color='#74BCFF', alpha=0.5)
        ax.hist(l_in[i].data.numpy().ravel(), bins=10, range=the_range, color='#FF9359')
        ax_bn.hist(l_in_bn[i].data.numpy().ravel(), bins=10, range=the_range, color='#74BCFF')
        for a in [ax_pa, ax, ax_pa_bn, ax_bn]:
            a.set_yticks(())
            a.set_xticks(())
        ax_pa_bn.set_xticks(p_range)
        ax_bn.set_xticks(the_range)
        axs[0, 0].set_ylabel('PreAct')
        axs[1, 0].set_ylabel('BN PreAct')
        axs[2, 0].set_ylabel('Act')
        axs[3, 0].set_ylabel('BN Act')
    plt.pause(0.01)

if __name__ == '__main__':
    nets = [Net(batchNormalization=False), Net(batchNormalization=True)]
    opts = [torch.optim.Adam(net.parameters(), lr=LR) for net in nets]
    lossFunc = torch.nn.MSELoss()

    f, axs = plt.subplots(4, N_HIDDEN + 1, figsize=(10, 5))
    plt.ion()
    plt.show()
    losses = [[], []]
    for epoch in range(EPOCH):
        layerInputs, preActs = [], []
        for net, l in zip(nets, losses):
            net.eval()
            pred, layerInput, preAct = net(xTesting)
            l.append(lossFunc(pred, yTesting).data.item())
            layerInputs.append(layerInput)
            preActs.append(preAct)
            net.train()
        plotHistogram(*layerInputs, *preActs)

        for step, (bX, bY) in enumerate(trainLoader):
            for net, opt in zip(nets, opts):
                pred, _, _ = net(bX)
                loss = lossFunc(pred, bY)
                opt.zero_grad()
                loss.backward()
                opt.step()
    print('[+] Done')
    plt.ioff()
    plt.figure(2)
    plt.plot(losses[0], c='#FF9359', lw=3, label='Original')
    plt.plot(losses[1], c='#74BCFF', lw=3, label='Batch Normalization')
    plt.xlabel('step')
    plt.ylabel('test loss')
    plt.ylim((0, 2000))
    plt.legend(loc='best')
    [net.eval() for net in nets]    # set eval mode to fix moving_mean and moving_var
    preds = [net(xTesting)[0] for net in nets]
    plt.figure(3)
    plt.plot(xTesting.data.numpy(), preds[0].data.numpy(), c='#FF9359', lw=4, label='Original')
    plt.plot(xTesting.data.numpy(), preds[1].data.numpy(), c='#74BCFF', lw=4, label='Batch Normalization')
    plt.scatter(xTesting.data.numpy(), yTesting.data.numpy(), c='r', s=50, alpha=0.2, label='train')
    plt.legend(loc='best')
    plt.show()

