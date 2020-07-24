#!/usr/bin/env python3
import torch
import numpy as np

npData = np.arange(6).reshape((2, 3))
torchData = torch.from_numpy(npData)
print('npData =>', npData, '\ntorchData =>', torchData)

tensorToArray = torchData.numpy()
print('tensorToArray =>', tensorToArray)

data = np.mat([[1, 2], [3, 4]])
tensor = torch.FloatTensor(data)
print('matrix multiplication (matmul)')
print('data =>', np.matmul(data, data))
print('tensor =>', torch.mm(tensor, tensor))
print('matrix multiplication (dot)')
print('data =>', data.dot(data))
try:
    print('tensor =>', tensor.dot(tensor))
except Exception as e:
    print(e)

