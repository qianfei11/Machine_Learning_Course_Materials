#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = True

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE, 
            hidden_size=64, 
            num_layers=1, 
            batch_first=True
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        rOut, (hiddenStateN, hiddenStateC) = self.rnn(x, None)
        out = self.out(rOut[:, -1, :]) # (batch, time step, input)
        return out

if __name__ == '__main__':
    trainData = dsets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor(), download=DOWNLOAD_MNIST)
    trainLoader = Data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
    testData = dsets.MNIST(root='./mnist', train=False, transform=transforms.ToTensor())
    testX = testData.data.type(torch.FloatTensor)[:2000] / 255.
    testY = testData.targets.numpy()[:2000]

    rnn = RNN()
    print(rnn)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    lossFunc = nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        for step, (bX, bY) in enumerate(trainLoader):
            bX = bX.view(-1, 28, 28)
            output = rnn(bX)
            loss = lossFunc(output, bY)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                testOutput = rnn(testX)
                predY = torch.max(testOutput, 1)[1].data.numpy().squeeze()
                accuracy = float((predY == testY).astype(int).sum()) / float(testY.size)
                print('[+] Epoch: {}, Loss: {}, Accuracy: {}'.format(epoch, loss, accuracy))
    testOutput = rnn(testX[:10].view(-1, 28, 28))
    predY = torch.max(testOutput, 1)[1].data.numpy()
    print(predY)
    print(testY[:10])

