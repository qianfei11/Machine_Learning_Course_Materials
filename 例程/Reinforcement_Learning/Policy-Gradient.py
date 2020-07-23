#!/usr/bin/env python3
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gym
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 400 # renders environment if total episode reward is greater then this threshold
RENDER = False # rendering wastes time

class PolicyGradient:
    def __init__(self, nActions, nFeatures, learningRate=0.01, rewardDecay=0.95, outputGraph=False):
        self.nActions = nActions
        self.nFeatures = nFeatures
        self.lr = learningRate
        self.gamma = rewardDecay
        self.episodeObs, self.episodeAs, self.episodeRs = [], [], []
        self.buildNet()
        self.sess = tf.Session()
        if outputGraph:
            tf.summary.FileWriter('log/', self.sess.graph)
            print('[+] TensorBoard built successfully')
        self.sess.run(tf.global_variables_initializer())

    def buildNet(self): # 建立神经网络
        with tf.name_scope('inputs'):
            self.tfObs = tf.placeholder(tf.float32, [None, self.nFeatures], name='observations')
            self.tfActs = tf.placeholder(tf.int32, [None], name='actionsNum')
            self.tfVt = tf.placeholder(tf.float32, [None], name='actionsValue')
        # fc1
        layer = tf.layers.dense(inputs=self.tfObs, units=10, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3), bias_initializer=tf.constant_initializer(0.1), name='fc1') # 全连接层
        allAct = tf.layers.dense(inputs=layer, units=self.nActions, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3), bias_initializer=tf.constant_initializer(0.1), name='fc2') # 全连接层
        self.allActProb = tf.nn.softmax(allAct, name='actProb') # 求出行为对应的概率

        with tf.name_scope('loss'): # 计算误差
            negLogProb = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=allAct, labels=self.tfActs)
            #negLogProb = tf.reduce_sum(-tf.log(self.allActProb) * tf.one_hot(self.tfActs, self.tfActs), axis=1) # 将目标函数修改为对最小值的求解
            loss = tf.reduce_mean(negLogProb * self.tfVt)

        with tf.name_scope('train'):
            self.trainOp = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def chooseAction(self, observation): # 选择行为
        probWeights = self.sess.run(self.allActProb, feed_dict={self.tfObs: observation[np.newaxis, :]}) # 获取概率
        action = np.random.choice(range(probWeights.shape[1]), p=probWeights.ravel()) # 通过概率选择行为
        return action

    def storeTransition(self, s, a, r): # 存储回合
        self.episodeObs.append(s)
        self.episodeAs.append(a)
        self.episodeRs.append(r)

    def learn(self): # 学习更新参数
        discountedEpRsNorm = self.discountAndNormRewards()
        self.sess.run(self.trainOp, feed_dict={self.tfObs: np.vstack(self.episodeObs), self.tfActs: np.array(self.episodeAs), self.tfVt: discountedEpRsNorm}) # 训练
        self.episodeObs, self.episodeAs, self.episodeRs = [], [], [] # 清空存储空间
        return discountedEpRsNorm

    def discountAndNormRewards(self): # 衰减回合奖励
        discountedEpRs = np.zeros_like(self.episodeRs)
        runningAdd = 0
        for t in reversed(range(len(self.episodeRs))):
            runningAdd = runningAdd * self.gamma + self.episodeRs[t]
            discountedEpRs[t] = runningAdd
        # 数据归一化
        discountedEpRs -= np.mean(discountedEpRs)
        discountedEpRs /= np.std(discountedEpRs)
        return discountedEpRs

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(1) # reproducible, general Policy gradient has high variance
    env = env.unwrapped
    RL = PolicyGradient(nActions=env.action_space.n, nFeatures=env.observation_space.shape[0], learningRate=0.02, rewardDecay=0.99)
    for episode in range(3000):
        observation = env.reset()
        while True:
            if RENDER:
                env.render()
            action = RL.chooseAction(observation)
            observation_, reward, isDone, info = env.step(action)
            RL.storeTransition(observation, action, reward) # 存储当前回合
            if isDone:
                episodeRsSum = sum(RL.episodeRs)
                if 'runningReward' not in globals():
                    runningReward = episodeRsSum
                else:
                    runningReward = runningReward * 0.99 + episodeRsSum * 0.01 # 更新奖励
                if runningReward > DISPLAY_REWARD_THRESHOLD: # 训练到一定程度
                    RENDER = True
                print('[+] episode: {}, reward: {}'.format(episode, runningReward))
                vt = RL.learn()
                if episode == 0:
                    plt.plot(vt) # 绘制回合奖励图
                    plt.xlabel('Episode Steps')
                    plt.ylabel('Normalized State-action value')
                    plt.show()
                break
            observation = observation_

