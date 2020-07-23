#!/usr/bin/env python3
import numpy as np
import pandas as pd
import time
import sys

if len(sys.argv) == 2 and sys.argv[1] == 'SET_SEED':
    seed = int(input('Input seed: '))
    np.random.seed(seed)

NSTATES = 6
ACTIONS = ['left', 'right']
EPSILON = 0.9 # greedy policy
ALPHA = 0.1 # learning rate
LAMBDA = 0.9 # discount factor
MAX_EPSODES = 13
FRESH_TIME = 0.01

def buildQTable(nStates, actions): # 创建Q表
    table = pd.DataFrame(
        np.zeros((nStates, len(actions))),
        columns=actions, # 纵坐标为Actions
    )
    print(table)
    return table

def chooseAction(state, qTable): # 随机选择Action
    stateActions = qTable.iloc[state, :]
    if np.random.rand() > EPSILON or stateActions.all() == False: # 命中了10%的概率或是表中所有Actions的值都为0，随机选择下一个Action
        actionName = np.random.choice(ACTIONS)
    else: # 反之取值最大的Action作为下一个选择
        actionName = stateActions.idxmax()
    return actionName

def getEnvFeedback(S, A): # 通过当前状态和行为获取下一状态和奖励函数值
    if A == 'right': # 如果Action为向右走
        if S == NSTATES - 2: # 如果当前状态在目的地左侧1个位置
            S_ = 'terminate' # 下一个状态设为终止
            R = 1 # 奖励函数值设为1
        else:
            S_ = S + 1 # 下一状态设为当前状态+1
            R = 0 # 奖励函数值为0
    else: # 如果Action为向左走
        R = 0 # 向左走不会到目的地，奖励函数值设为0
        if S == 0: # 如果已经在最左侧
            S_ = S # 下一状态不变
        else:
            S_ = S - 1 # 下一状态设为当前状态-1
    return S_, R

def updateEnv(S, episode, stepCounter): # 更新环境
    envList = ['-'] * (NSTATES - 1) + ['T'] # 初始化环境
    if S == 'terminate': # 如果状态为终止
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, stepCounter)
        print('\r{}'.format(interaction))
        time.sleep(0.5)
    else:
        envList[S] = 'M' # 更新当前位置
        interaction = ''.join(envList)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def rl(): # 增强学习训练函数
    qTable = buildQTable(NSTATES, ACTIONS) # 创建Q表
    for episode in range(MAX_EPSODES): # 循环训练
        stepCounter = 0
        S = 0
        isTerminated = False
        updateEnv(S, episode, stepCounter) # 初始化环境
        while not isTerminated:
            A = chooseAction(S, qTable) # 选择Action
            S_, R = getEnvFeedback(S, A) # 获取下一个状态和当前奖励函数值
            qPred = qTable.loc[S, A] # 获取预测的Q值
            if S_ != 'terminate': # 如果没有到达目的地
                qTarget = R + LAMBDA * qTable.iloc[S_, :].max() # 计算实际的Q值
            else:
                qTarget = R
                isTerminated = True
            qTable.loc[S, A] += ALPHA * (qTarget - qPred) # 更新Q值
            S = S_ # 更新当前状态
            updateEnv(S, episode, stepCounter + 1) # 更新环境
            stepCounter += 1
    return qTable

if __name__ == '__main__':
    qTable = rl()
    print(qTable)

