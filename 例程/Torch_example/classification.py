#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

nData = torch.ones(100, 2)
x0 = torch.normal(2 * nData, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2 * nData, 1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1)).type(torch.LongTensor)

#plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
#plt.show()

class Net(torch.nn.Module):
    def __init__(self, nFeature, nHidden, nOutput):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(nFeature, nHidden)
        self.predict = torch.nn.Linear(nHidden, nOutput)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

if __name__ == '__main__':
    net = Net(2, 10, 2)
    print(net)
    plt.ion()
    plt.show()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
    lossFunc = torch.nn.CrossEntropyLoss()
    for t in range(100):
        out = net(x)
        loss = lossFunc(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if t % 2 == 0:
            plt.cla()
            prediction = torch.max(F.softmax(out, dim=1), 1)[1]
            predY = prediction.data.numpy().squeeze()
            targetY = y.data.numpy()
            plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=predY, s=100, lw=0, cmap='RdYlGn')
            accuracy = sum(predY == targetY) / 200
            plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size':20, 'color':'red'})
            plt.pause(0.1)
    print('[+] Done')
    plt.ioff()
    plt.show()

