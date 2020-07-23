#!/usr/bin/env python3
import numpy as np
import pandas as pd
import tkinter as tk
import time

UNIT = 40 # pixels
MAZE_H = 4 # grid height
MAZE_W = 4 # grid width

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.actionSpace = ['up', 'down', 'left', 'right']
        self.nActions = len(self.actionSpace)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_W * UNIT))
        self.buildMaze()

    def buildMaze(self):
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)
        origin = np.array([20, 20])
        # hell
        hell1Center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(hell1Center[0] - 15, hell1Center[1] - 15, hell1Center[0] + 15, hell1Center[1] + 15, fill='black')
        # hell
        hell2Center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(hell2Center[0] - 15, hell2Center[1] - 15, hell2Center[0] + 15, hell2Center[1] + 15, fill='black')
        # oval
        ovalCenter = origin + UNIT * 2
        self.oval = self.canvas.create_rectangle(ovalCenter[0] - 15, ovalCenter[1] - 15, ovalCenter[0] + 15, ovalCenter[1] + 15, fill='yellow')
        # rect
        self.rect = self.canvas.create_rectangle(origin[0] - 15, origin[1] - 15, origin[0] + 15, origin[1] + 15, fill='red')
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(origin[0] - 15, origin[0] - 15, origin[0] + 15, origin[0] + 15, fill='red')
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        baseAction = np.array([0, 0])
        if action == 0:
            if s[1] > UNIT:
                baseAction[1] -= UNIT
        elif action == 1:
            if s[1] < (MAZE_H - 1) * UNIT:
                baseAction[1] += UNIT
        elif action == 2:
            if s[0] < (MAZE_W - 1) * UNIT:
                baseAction[0] += UNIT
        elif action == 3:
            if s[0] > UNIT:
                baseAction[0] -= UNIT
        self.canvas.move(self.rect, baseAction[0], baseAction[1])
        s_ = self.canvas.coords(self.rect)
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            isDone = True
            s_ = 'terminated'
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            isDone = True
            s_ = 'terminated'
        else:
            reward = 0
            isDone = False
        return s_, reward, isDone

    def render(self):
        time.sleep(0.1)
        self.update()

class QLearningTable:
    def __init__(self, actions, learningRate=0.01, rewardDecay=0.9, greedyPolicy=0.9):
        self.actions = actions
        self.lr = learningRate
        self.gamma = rewardDecay
        self.epsilon = greedyPolicy
        self.qTable = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def chooseAction(self, observation):
        self.checkStateExists(observation)
        if np.random.uniform() < self.epsilon:
            stateAction = self.qTable.loc[observation, :]
            action = np.random.choice(stateAction[stateAction == np.max(stateAction)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.checkStateExists(s_)
        qPred = self.qTable.loc[s, a]
        if s_ != 'terminated':
            qTarget = r + self.gamma * self.qTable.loc[s_, :].max()
        else:
            qTarget = r
        self.qTable.loc[s, a] += self.lr * (qTarget - qPred)

    def checkStateExists(self, state):
        if state not in self.qTable.index:
            self.qTable = self.qTable.append(pd.Series([0] * len(self.actions), index=self.qTable.columns, name=state))

def update():
    for episode in range(20):
        observation = env.reset()
        while True:
            env.render()
            action = RL.chooseAction(str(observation))
            observation_, reward, isDone = env.step(action)
            RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_
            if isDone:
                break
    print('[*] Done')
    env.destroy()

if __name__ == '__main__':
    env = Maze()
    RL = QLearningTable(actions=list(range(env.nActions)))
    env.after(100, update)
    env.mainloop()

