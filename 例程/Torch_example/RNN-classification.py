#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

EPOCH = 1 # 训练次数
BATCH_SIZE = 64
TIME_STEP = 28 # RNN 时间步数 / 图片高度
INPUT_SIZE = 28 # RNN 每步输入值 / 图片每行像素
LR = 0.01
DOWNLOAD_MNIST = True

# (input0, state0) -> LSTM -> (output0, state1)
# (input1, state1) -> LSTM -> (output1, state2)
# ...
# (inputN, stateN) -> LSTM -> (outputN, stateN+1)
# outputN -> Linear -> prediction
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE, 
            hidden_size=64, # 隐含层神经元个数
            num_layers=1, # RNN层数
            batch_first=True # 输入输出将batch作为第一维度的特征值
        )
        self.out = nn.Linear(64, 10) # 输出层

    def forward(self, x):
        # hiddenStateN是分线
        # hiddenStateC是主线
        rOut, (hiddenStateN, hiddenStateC) = self.rnn(x, None) # (n layers, batch, hidden size)
        # 选取最后一个时间点的输出
        out = self.out(rOut[:, -1, :]) # (batch, time step, input size)
        return out

if __name__ == '__main__':
    # 训练数据
    trainData = dsets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor(), download=DOWNLOAD_MNIST)
    trainLoader = Data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
    # 测试数据
    testData = dsets.MNIST(root='./mnist', train=False, transform=transforms.ToTensor())
    testX = testData.data.type(torch.FloatTensor)[:2000] / 255.
    testY = testData.targets.numpy()[:2000]

    rnn = RNN()
    print(rnn)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR) # 优化所有参数
    lossFunc = nn.CrossEntropyLoss() # 标签不是one hot
    for epoch in range(EPOCH):
        for step, (bX, bY) in enumerate(trainLoader):
            bX = bX.view(-1, 28, 28) # reshape to (batch, time step, input size)
            output = rnn(bX)
            loss = lossFunc(output, bY)
            optimizer.zero_grad()
            loss.backward() # 后向传播
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

