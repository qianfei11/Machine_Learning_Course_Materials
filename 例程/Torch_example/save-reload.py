#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

def save():
    net = torch.nn.Sequential(torch.nn.Linear(1, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1))
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    lossFunc = torch.nn.MSELoss()
    for t in range(100):
        prediction = net(x)
        loss = lossFunc(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(net, 'net.pkl')
    torch.save(net.state_dict(), 'netParams.pkl')

    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

def restoreNet():
    net = torch.load('net.pkl')
    prediction = net(x)

    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

def restoreParams():
    net = torch.nn.Sequential(torch.nn.Linear(1, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1))
    net.load_state_dict(torch.load('netParams.pkl'))
    prediction = net(x)

    plt.subplot(133)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()

if __name__ == '__main__':
    save()
    restoreNet()
    restoreParams()

