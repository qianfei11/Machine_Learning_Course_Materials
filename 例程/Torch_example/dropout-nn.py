#!/usr/bin/env python3
import torch
import matplotlib.pyplot as plt

N_SAMPLES = 20
N_HIDDEN = 300
LR = 0.01

xTraining = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
yTraining = xTraining + 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))
xTesting = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
yTesting = xTesting + 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

plt.scatter(xTraining.data.numpy(), yTraining.data.numpy(), c='magenta', s=50, alpha=0.5, label='train')
plt.scatter(xTesting.data.numpy(), yTesting.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
plt.legend(loc='upper left')
plt.ylim((-2.5, 2.5))
plt.show()

if __name__ == '__main__':
    netOverfitting = torch.nn.Sequential(
        torch.nn.Linear(1, N_HIDDEN), 
        torch.nn.ReLU(), 
        torch.nn.Linear(N_HIDDEN, N_HIDDEN), 
        torch.nn.ReLU(), 
        torch.nn.Linear(N_HIDDEN, 1)
    )
    netDropped = torch.nn.Sequential(
        torch.nn.Linear(1, N_HIDDEN), 
        torch.nn.Dropout(0.5), 
        torch.nn.ReLU(), 
        torch.nn.Linear(N_HIDDEN, N_HIDDEN), 
        torch.nn.Dropout(0.5), 
        torch.nn.ReLU(), 
        torch.nn.Linear(N_HIDDEN, 1)
    )
    print(netOverfitting)
    print(netDropped)
    optimizerOverfitting = torch.optim.Adam(netOverfitting.parameters(), lr=LR)
    optimizerDropped = torch.optim.Adam(netDropped.parameters(), lr=LR)
    lossFunc = torch.nn.MSELoss()
    plt.ion()
    for t in range(500):
        predOverfitting = netOverfitting(xTraining)
        predDropped = netDropped(xTraining)
        lossOverfitting = lossFunc(predOverfitting, yTraining)
        lossDropped = lossFunc(predDropped, yTraining)

        optimizerOverfitting.zero_grad()
        optimizerDropped.zero_grad()
        lossOverfitting.backward()
        lossDropped.backward()
        optimizerOverfitting.step()
        optimizerDropped.step()

        if t % 10 == 0:
            netOverfitting.eval()
            netDropped.eval()

            plt.cla()
            testPredOverfitting = netOverfitting(xTesting)
            testPredDropped = netDropped(xTesting)
            plt.scatter(xTraining.data.numpy(), yTraining.data.numpy(), c='magenta', s=50, alpha=0.3, label='train')
            plt.scatter(xTesting.data.numpy(), yTesting.data.numpy(), c='cyan', s=50, alpha=0.3, label='test')
            plt.plot(xTesting.data.numpy(), testPredOverfitting.data.numpy(), 'r-', lw=3, label='overfitting')
            plt.plot(xTesting.data.numpy(), testPredDropped.data.numpy(), 'b--', lw=3, label='dropout(50%)')
            plt.text(0, -1.2, 'overfitting loss=%.4f' % lossFunc(testPredOverfitting, yTesting).data.numpy(), fontdict={'size': 20, 'color': 'red'})
            plt.text(0, -1.5, 'dropout loss=%.4f' % lossFunc(testPredDropped, yTesting).data.numpy(), fontdict={'size': 20, 'color': 'blue'})
            plt.legend(loc='upper left')
            plt.ylim((-2.5, 2.5))
            plt.pause(0.1)

            netOverfitting.train()
            netDropped.train()
    print('[+] Done')
    plt.ioff()
    plt.show()

