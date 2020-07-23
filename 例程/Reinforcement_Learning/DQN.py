#!/usr/bin/env python3
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gym
import matplotlib.pyplot as plt

class DeepQNetwork:
    def __init__(self, nActions, nFeatures, learningRate=0.01, rewardDecay=0.9, eGreedy=0.9, replaceTargetIter=300, memorySize=500, batchSize=32, eGreedyIncrement=None, outputGraph=False):
        self.nActions = nActions # 动作数
        self.nFeatures = nFeatures # 神经网络的特征数
        self.lr = learningRate # 学习率
        self.gamma = rewardDecay # 奖励衰减项
        self.epsilonMax = eGreedy # 贪婪系数最大值
        self.replaceTargetIter = replaceTargetIter # 更新targetNet的步数
        self.memorySize = memorySize # 用于记忆的数据数量
        self.batchSize = batchSize # Batch大小
        self.epsilonIncrement = eGreedyIncrement # 贪婪系数变化率
        self.epsilon = 0 if eGreedyIncrement is not None else self.epsilonMax # 贪婪系数
        self.learnStepCounter = 0 # 记录学习的步数
        self.memory = np.zeros((self.memorySize, nFeatures * 2 + 2)) # 创建存储空间
        self.buildNet() # 建立网络
        tParams = tf.get_collection('targetNetParams') # 获取targetNet中的参数
        eParams = tf.get_collection('evalNetParams') # 获取evalNet中的参数
        self.replaceTargetOp = [tf.assign(t, e) for t, e in zip(tParams, eParams)] # 将targetNet中的参数替换为evalNet中的参数
        self.sess = tf.Session()
        if outputGraph:
            tf.summary.FileWriter('log/', self.sess.graph)
        self.sess.run(tf.global_variables_initializer()) # 激活变量
        self.costHis = [] # 记录误差

    def buildNet(self):
        # Build evalNet
        self.s = tf.placeholder(tf.float32, [None, self.nFeatures], name='s') # 输入1：当前的状态State
        self.qTarget = tf.placeholder(tf.float32, [None, self.nActions], name='qTarget') # 输入2：现实Q值
        with tf.variable_scope('evalNet'):
            cNames = ['evalNetParams', tf.GraphKeys.GLOBAL_VARIABLES] # 用于收集evalNet中所有的参数
            nL1 = 10 # 第一层神经元个数
            wInitializer = tf.random_normal_initializer(0., 0.3) # 随机生成权重
            bInitializer = tf.constant_initializer(0.1) # 随机生成偏置

            with tf.variable_scope('l1'): # 第一层
                w1 = tf.get_variable('w1', [self.nFeatures, nL1], initializer=wInitializer, collections=cNames) # 权重
                b1 = tf.get_variable('b1', [1, nL1], initializer=bInitializer, collections=cNames) # 偏置
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1) # 激励函数使用ReLU

            with tf.variable_scope('l2'): # 第二层
                w2 = tf.get_variable('w2', [nL1, self.nActions], initializer=wInitializer, collections=cNames) # 权重
                b2 = tf.get_variable('b2', [1, self.nActions], initializer=bInitializer, collections=cNames) # 偏置
                self.qEval = tf.matmul(l1, w2) + b2 # 估计的Q值
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.qTarget, self.qEval))
        with tf.variable_scope('train'):
            self.trainOp = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # Build targetNet
        self.s_ = tf.placeholder(tf.float32, [None, self.nFeatures], name='s_') # 输入1：下一个状态State
        with tf.variable_scope('targetNet'):
            cNames = ['targetNetParams', tf.GraphKeys.GLOBAL_VARIABLES] # 用于收集targetNet中所有的参数

            with tf.variable_scope('l1'): # 第一层
                w1 = tf.get_variable('w1', [self.nFeatures, nL1], initializer=wInitializer, collections=cNames) # 权重
                b1 = tf.get_variable('b1', [1, nL1], initializer=bInitializer, collections=cNames) # 偏置
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1) # 激励函数使用ReLU

            with tf.variable_scope('l2'): # 第二层
                w2 = tf.get_variable('w2', [nL1, self.nActions], initializer=wInitializer, collections=cNames) # 权重
                b2 = tf.get_variable('b2', [1, self.nActions], initializer=bInitializer, collections=cNames) # 偏置
                self.qNext = tf.matmul(l1, w2) + b2 # 估计的Q值

    def storeTransition(self, s, a, r, s_):
        if not hasattr(self, 'memoryCounter'):
            self.memoryCounter = 0
        transition = np.hstack((s, [a, r], s_))
        idx = self.memoryCounter % self.memorySize
        self.memory[idx, :] = transition
        self.memoryCounter += 1

    def chooseAction(self, observation): # 选择行为Action
        observation = observation[np.newaxis, :] # 变成二维矩阵便于处理
        if np.random.rand() < self.epsilon:
            actionsValue = self.sess.run(self.qEval, feed_dict={self.s: observation}) # 放入evalNet中分析计算行为的值
            action = np.argmax(actionsValue) # 选择值最大的行为Action
        else:
            action = np.random.randint(0, self.nActions) # 10%的概率随机选择行为Action
        return action

    def learn(self):
        if self.learnStepCounter % self.replaceTargetIter == 0: # 判断学习之前是否需要替换参数
            self.sess.run(self.replaceTargetOp)
            print('[+] Target params replaced.')
        if self.memoryCounter > self.memorySize: # 判断存储空间中的数据数量
            sampleIdx = np.random.choice(self.memorySize, size=self.batchSize)
        else:
            sampleIdx = np.random.choice(self.memoryCounter, size=self.batchSize)
        batchMemory = self.memory[sampleIdx, :] # 获取一部分数据作为Batch
        qNext, qEval = self.sess.run([self.qNext, self.qEval], feed_dict={self.s_: batchMemory[:, -self.nFeatures:], self.s: batchMemory[:, :self.nFeatures]}) # 分别计算当前状态和下一状态的Q值
        qTarget = qEval.copy() # 
        batchIdx = np.arange(self.batchSize, dtype=np.int32)
        evalActIdx = batchMemory[:, self.nFeatures].astype(int)
        reward = batchMemory[:, self.nFeatures + 1]
        qTarget[batchIdx, evalActIdx] = reward + self.gamma * np.max(qNext, axis=1)
        _, self.cost = self.sess.run([self.trainOp, self.loss], feed_dict={self.s: batchMemory[:, :self.nFeatures], self.qTarget: qTarget}) # 计算误差值
        self.costHis.append(self.cost) # 存储误差值
        self.epsilon = self.epsilon + self.epsilonIncrement if self.epsilon < self.epsilonMax else self.epsilonMax # 更新贪婪系数
        self.learnStepCounter += 1

    def plotCost(self): # 展示误差
        plt.plot(np.arange(len(self.costHis)), self.costHis)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)
    totalStep = 0
    RL = DeepQNetwork(nActions=env.action_space.n, nFeatures=env.observation_space.shape[0], learningRate=0.01, eGreedy=0.9, replaceTargetIter=100, memorySize=2000, eGreedyIncrement=0.001)
    for episode in range(100):
        observation = env.reset() # 获取第一个状态
        episodeReward = 0
        while True:
            env.render()
            action = RL.chooseAction(observation) # 选择行为Actor
            observation_, reward, isDone, info = env.step(action) # 获取执行行为后得到的相关信息
            x, xDot, theta, thetaDot = observation_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8 # 根据离画面中心距离判断奖励值
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5 # 根据杆子偏离度判断奖励值
            reward = r1 + r2 # 替换奖励值
            RL.storeTransition(observation, action, reward, observation_) # 存储步骤
            episodeReward += reward # 更新奖励值
            if totalStep > 1000:
                RL.learn() # 学习
            if isDone:
                print('[+] episode: {}, episodeReward: {}, epsilon: {}'.format(episode, episodeReward, RL.epsilon)) # 输出
                break
            observation = observation_
            totalStep += 1
    RL.plotCost() # 绘制误差图

