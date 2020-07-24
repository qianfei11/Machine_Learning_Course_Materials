#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( # 1 * 28 * 28
            nn.Conv2d(
                in_channels=1, 
                out_channels=16, 
                kernel_size=5, 
                stride=1, 
                padding=2, 
            ), # -> 16 * 28 * 28
            nn.ReLU(), # -> 16 * 28 * 28
            nn.MaxPool2d(kernel_size=2), # -> 16 * 14 * 14
        )
        self.conv2 = nn.Sequential( # 16 * 14 * 14
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, 
                kernel_size=5, 
                stride=1, 
                padding=2
            ), # -> 32, 14, 14
            nn.ReLU(), # -> 32, 14, 14
            nn.MaxPool2d(kernel_size=2), # -> 32, 7, 7
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) # (batch, 32, 7, 7)
        x = x.view(x.size(0), -1) # (batch, 32 * 7 * 7)
        output = self.out(x)
        return output, x

if __name__ == '__main__':
    trainData = torchvision.datasets.MNIST(root='./mnist', train=True, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)
    #plt.imshow(trainData.data[0].numpy(), cmap='gray')
    #plt.title('%i' % trainData.targets[0])
    #plt.show()
    trainLoader = Data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
    testData = torchvision.datasets.MNIST(root='./mnist', train=False)
    testX = torch.unsqueeze(testData.test_data, dim=1).type(torch.FloatTensor)[:2000] / 255.
    testY = testData.test_labels[:2000]

    cnn = CNN()
    print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    lossFunc = nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        for step, (bX, bY) in enumerate(trainLoader):
            output = cnn(bX)[0]
            loss = lossFunc(output, bY)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                testOutput, lastLayer = cnn(testX)
                predY = torch.max(testOutput, 1)[1].data.numpy()
                accuracy = float((predY == testY.data.numpy()).astype(int).sum()) / float(testY.size(0))
                print('[+] Epoch: {}, Loss: {}, Accuracy: {}'.format(epoch, loss, accuracy))
    testOutput = cnn(testX[:10])[0]
    predY = torch.max(testOutput, 1)[1].data.numpy()
    print(predY)
    print(testY[:10].numpy())

