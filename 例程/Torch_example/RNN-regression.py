#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02

#steps = np.linspace(0, np.pi * 2, 100, dtype=np.float32)
#xNp = np.sin(steps)
#yNp = np.cos(steps)
#plt.plot(steps, yNp, 'r-', label='target (cos)')
#plt.plot(steps, xNp, 'b-', label='input (sin)')
#plt.legend(loc='best')
#plt.show()

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE, 
            hidden_size=32, 
            num_layers=1, 
            batch_first=True
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, hState):
        rOut, hState = self.rnn(x, hState)
        outs = []
        for timeStep in range(rOut.size(1)):
            outs.append(self.out(rOut[:, timeStep, :]))
        return torch.stack(outs, dim=1), hState

if __name__ == '__main__':
    rnn = RNN()
    print(rnn)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    lossFunc = nn.MSELoss()
    hState = None
    plt.figure(1, figsize=(12, 5))
    plt.ion()
    for step in range(60):
        start, end = step * np.pi, (step + 1) * np.pi
        steps = np.linspace(start, end, TIME_STEP, dtype=np.float32, endpoint=False)
        xNp = np.sin(steps)
        yNp = np.cos(steps)
        x = torch.from_numpy(xNp[np.newaxis, :, np.newaxis])
        y = torch.from_numpy(yNp[np.newaxis, :, np.newaxis])

        prediction, hState = rnn(x, hState)
        hState = hState.data
        loss = lossFunc(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        plt.plot(steps, yNp.flatten(), 'r-')
        plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
        plt.draw()
        plt.pause(0.05)
    print('[+] Done')
    plt.ioff()
    plt.show()

