#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as Data

EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
DOWNLOAD_MNIST = True
N_TEST_IMG = 5

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), 
            nn.Tanh(), 
            nn.Linear(128, 64), 
            nn.Tanh(), 
            nn.Linear(64, 12), 
            nn.Tanh(), 
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12), 
            nn.Tanh(), 
            nn.Linear(12, 64), 
            nn.Tanh(), 
            nn.Linear(64, 128), 
            nn.Tanh(), 
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

if __name__ == '__main__':
    trainData = dsets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor(), download=DOWNLOAD_MNIST)
    trainLoader = Data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)

    autoencoder = AutoEncoder()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    lossFunc = nn.MSELoss()
    for epoch in range(EPOCH):
        for step, (x, bLabel) in enumerate(trainLoader):
            bX = x.view(-1, 28 * 28)
            bY = x.view(-1, 28 * 28)
            encoded, decoded = autoencoder(bX)
            loss = lossFunc(decoded, bY)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print('[+] Epoch: {}, loss: {}'.format(epoch, loss))

