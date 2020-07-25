#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 64
LR_G = 0.0001
LR_D = 0.0001
N_IDEAS = 5
ART_COMPONENTS = 15
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

def artistWorks():
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a - 1)
    paintings = torch.from_numpy(paintings).float()
    return paintings

if __name__ == '__main__':
    G = nn.Sequential(
        nn.Linear(N_IDEAS, 128), 
        nn.ReLU(), 
        nn.Linear(128, ART_COMPONENTS)
    )
    D = nn.Sequential(
        nn.Linear(ART_COMPONENTS, 128), 
        nn.ReLU(), 
        nn.Linear(128, 1), 
        nn.Sigmoid()
    )
    optD = torch.optim.Adam(D.parameters(), lr=LR_D)
    optG = torch.optim.Adam(G.parameters(), lr=LR_G)
    plt.ion()
    for step in range(10000):
        artistPaintings = artistWorks()
        GIdeas = torch.randn(BATCH_SIZE, N_IDEAS, requires_grad=True)
        GPaintings = G(GIdeas)
        probArtist1 = D(GPaintings)
        GLoss = torch.mean(torch.log(1. - probArtist1))
        optG.zero_grad()
        GLoss.backward()
        optG.step()
        probArtist0 = D(artistPaintings)
        probArtist1 = D(GPaintings.detach())
        DLoss = -torch.mean(torch.log(probArtist0))
        optD.zero_grad()
        DLoss.backward(retain_graph=True)
        optD.step()
        if step % 50 == 0:
            plt.cla()
            plt.plot(PAINT_POINTS[0], GPaintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting',)
            plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
            plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
            plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % probArtist0.data.numpy().mean(), fontdict={'size': 13})
            plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -DLoss.data.numpy(), fontdict={'size': 13})
            plt.ylim((0, 3))
            plt.legend(loc='upper right', fontsize=10)
            plt.draw()
            plt.pause(0.01)
    print('[+] Done')
    plt.ioff()
    plt.show()

