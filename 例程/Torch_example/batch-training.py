#!/usr/bin/env python3
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

BATCH_SIZE = 8

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

if __name__ == '__main__':
    torchDataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(dataset=torchDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    for epoch in range(3):
        for step, (batchX, batchY) in enumerate(loader):
            print('[+] Epoch: {}, Step: {}, batch x: {}, batch y: {}'.format(epoch, step, batchX.numpy(), batchY.numpy()))

