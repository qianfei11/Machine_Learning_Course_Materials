#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actionsValue = self.out(x)
        return actionsValue

class DQN(object):
    def __init__(self):
        self.evalNet, self.targetNet = Net(), Net()
        self.learnStepCounter = 0
        self.memoryCounter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.evalNet.parameters(), lr=LR)
        self.lossFunc = nn.MSELoss()

    def chooseAction(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.rand() < EPSILON:
            actionsValue = self.evalNet.forward(x)
            action = torch.max(actionsValue, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def storeTransititon(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        idx = self.memoryCounter % MEMORY_CAPACITY
        self.memory[idx, :] = transition
        self.memoryCounter += 1

    def learn(self):
        if self.learnStepCounter % TARGET_REPLACE_ITER == 0:
            self.targetNet.load_state_dict(self.evalNet.state_dict())
        self.learnStepCounter += 1
        sampleIdx = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        bMemory = self.memory[sampleIdx, :]
        bS = torch.FloatTensor(bMemory[:, :N_STATES])
        bA = torch.LongTensor(bMemory[:, N_STATES:N_STATES + 1].astype(int))
        bR = torch.FloatTensor(bMemory[:, N_STATES + 1:N_STATES + 2])
        bS_ = torch.FloatTensor(bMemory[:, -N_STATES:])

        qEval = self.evalNet(bS).gather(1, bA)
        qNext = self.targetNet(bS_).detach()
        qTarget = bR + GAMMA * qNext.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.lossFunc(qEval, qTarget)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    dqn = DQN()
    for episode in range(400):
        s = env.reset()
        epR = 0
        while True:
            env.render()
            a = dqn.chooseAction(s)
            s_, r, isDone, info = env.step(a)
            x, xDot, theta, thetaDot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.
            r = r1 + r2
            dqn.storeTransititon(s, a, r, s_)
            epR += r
            if dqn.memoryCounter > MEMORY_CAPACITY:
                dqn.learn()
                if isDone:
                    print('[+] Episode: {}, Episode Reward: {}'.format(episode, round(epR, 2)))
            if isDone:
                break
            s = s_

