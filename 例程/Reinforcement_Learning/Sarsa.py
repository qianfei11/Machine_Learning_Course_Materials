#!/usr/bin/env python3
import numpy as np
import pandas as pd
import tkinter as tk
import time

UNIT = 40 # pixels
MAZE_H = 4 # grid height
MAZE_W = 4 # grid width

class Maze(tk.Tk, object): # 环境类
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

class RL(object): # 增强学习类
    def __init__(self, actions, learningRate=0.01, rewardDecay=0.9, greedyPolicy=0.9):
        self.actions = actions
        self.lr = learningRate # 学习率
        self.gamma = rewardDecay # 衰减项
        self.epsilon = greedyPolicy # 贪婪系数
        self.qTable = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def checkStateExists(self, state): # 检查状态State是否存在
        if state not in self.qTable.index:
            self.qTable = self.qTable.append(pd.Series([0] * len(self.actions), index=self.qTable.columns, name=state))

    def chooseAction(self, state): # 选择行为Action
        self.checkStateExists(state)
        if np.random.rand() < self.epsilon:
            stateAction = self.qTable.loc[state, :]
            action = np.random.choice(stateAction[stateAction == np.max(stateAction)].index) # 选择Q值最大的行为Action
        else: # 10%的概率随机选择行为Action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args): # 学习函数
        pass

class QLearningTable(RL):
    def __init__(self, actions, learningRate=0.01, rewardDecay=0.9, greedyPolicy=0.9):
        super(QLearningTable, self).__init__(actions, learningRate, rewardDecay, greedyPolicy)

    def learn(self, s, a, r, s_):
        self.checkStateExists(s_)
        qPred = self.qTable.loc[s, a]
        if s_ != 'terminated':
            qTarget = r + self.gamma * self.qTable.loc[s_, :].max()
        else:
            qTarget = r
        self.qTable.loc[s, a] += self.lr * (qTarget - qPred)

class SarsaTable(RL):
    def __init__(self, actions, learningRate=0.01, rewardDecay=0.9, greedyPolicy=0.9):
        super(SarsaTable, self).__init__(actions, learningRate, rewardDecay, greedyPolicy)

    def learn(self, s, a, r, s_, a_):
        self.checkStateExists(s_) # 检测得到的下一个状态State是否存在
        qPred = self.qTable.loc[s, a]
        if s_ != 'terminated': # 如果没有终止
            qTarget = r + self.gamma * self.qTable.loc[s_, a_] # 计算新的Q值
        else: # 状态终止
            qTarget = r
        self.qTable.loc[s, a] += self.lr * (qTarget - qPred) # 更新Q值

def update(): # 更新环境函数
    for episode in range(100):
        stepCounter = 0
        state = env.reset() # 获取初始状态
        action = RL.chooseAction(str(state)) # 获取初始行为
        while True:
            env.render() # 更新环境
            state_, reward, isDone = env.step(action) # 执行当前行为并获取下一状态
            action_ = RL.chooseAction(str(state_)) # 选择下一个行为Action
            RL.learn(str(state), action, reward, str(state_), action_) # 学习并更新Q表
            state = state_ # 更新下一个状态State
            action = action_ # 更新下一个行为Action
            stepCounter += 1
            if isDone:
                break
        if reward == 1:
            print('[+] Success! Episode %s: total_steps = %s' % (episode + 1, stepCounter))
        elif reward == -1:
            print('[!] Failed... Episode %s: total_steps = %s' % (episode + 1, stepCounter))
    print('[*] Done')
    env.destroy()

if __name__ == '__main__':
    env = Maze()
    RL = SarsaTable(actions=list(range(env.nActions)))
    env.after(100, update)
    env.mainloop()

