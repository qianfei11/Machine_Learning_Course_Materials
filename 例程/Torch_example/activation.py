#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 200)
xNp = x.data.numpy()

yRelu = F.relu(x).data.numpy()
ySigmoid = torch.sigmoid(x).data.numpy()
yTanh = torch.tanh(x).data.numpy()
ySoftmax = F.softplus(x).data.numpy()

plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(xNp, yRelu, c='red', label='ReLU')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(xNp, ySigmoid, c='red', label='Sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(xNp, yTanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(xNp, ySoftmax, c='red', label='Softmax')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.show()

