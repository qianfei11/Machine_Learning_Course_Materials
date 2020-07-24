#!/usr/bin/env python3
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gym

OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200
MAX_EP_STEPS = 1000
RENDER = False
GAMMA = 0.9 # 贪婪系数
LR_A = 0.001 # Learning Rate of Actor
LR_C = 0.01 # Learning Rate of Critic

class Actor(object):
    def __init__(self, sess, nFeatures, nActions, lr=0.001):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, nFeatures], 'state') # 输入1：状态
        self.a = tf.placeholder(tf.int32, None, 'act') # 输入2：动作
        self.tdError = tf.placeholder(tf.float32, None, 'tdError') # 输入3：奖励

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(inputs=self.s, units=20, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0., .1), bias_initializer=tf.constant_initializer(0.1), name='l1') # 第一层
            self.actsProb = tf.layers.dense(inputs=l1, units=nActions, activation=tf.nn.softmax, kernel_initializer=tf.random_normal_initializer(0., .1), bias_initializer=tf.constant_initializer(0.1), name='actsProb') # 第二层输出每个动作的概率

        with tf.variable_scope('expV'):
            logProb = tf.log(self.actsProb[0, self.a])
            self.expV = tf.reduce_mean(logProb * self.tdError) # loss

        with tf.variable_scope('train'):
            self.trainOp = tf.train.AdamOptimizer(lr).minimize(-self.expV) # min(expV) = max(-expV)

    def learn(self, s, a, td): # 学习
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.tdError: td}
        _, expV = self.sess.run([self.trainOp, self.expV], feed_dict)
        return expV

    def chooseAction(self, s): # 选择行为
        s = s[np.newaxis, :]
        probs = self.sess.run(self.actsProb, {self.s: s})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())

class Critic(object):
    def __init__(self, sess, nFeatures, lr=0.01):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, nFeatures], 'state') # 输入1：当前状态
        self.v_ = tf.placeholder(tf.float32, [1, 1], 'vNext') # 输入2：下一个奖励折现值
        self.r = tf.placeholder(tf.float32, None, 'r') # 输入3：当前奖励

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(inputs=self.s, units=20, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0., .1), bias_initializer=tf.constant_initializer(0.1), name='l1') # 第一层
            self.v = tf.layers.dense(inputs=l1, units=1, activation=None, kernel_initializer=tf.random_normal_initializer(0., .1), bias_initializer=tf.constant_initializer(0.1), name='V') # 第二层

        with tf.variable_scope('squaredTDError'):
            self.tdError = self.r + GAMMA * self.v_ - self.v # 时间差分值的平方
            self.loss = tf.square(self.tdError) # loss

        with tf.variable_scope('train'):
            self.trainOp = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_): # 学习奖励机制
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, {self.s: s_})
        tdError, _ = self.sess.run([self.tdError, self.trainOp], {self.s: s, self.v_: v_, self.r: r})
        return tdError # 返回给Actor

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(1)
    env = env.unwrapped
    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n
    sess = tf.Session()
    actor = Actor(sess, nFeatures=N_F, nActions=N_A, lr=LR_A)
    critic = Critic(sess, nFeatures=N_F, lr=LR_C)
    sess.run(tf.global_variables_initializer())
    if OUTPUT_GRAPH:
        tf.summary.FileWriter('log/', sess.graph)
    for episode in range(MAX_EPISODE):
        s = env.reset()
        t = 0
        trackR = []
        while True:
            if RENDER:
                env.render()
            a = actor.chooseAction(s) # 获取动作
            s_, r, isDone, info = env.step(a) # 执行动作
            if isDone:
                r = -20
            trackR.append(r) # 保存奖励值
            tdError = critic.learn(s, r, s_) # Critic学习奖励值
            actor.learn(s, a, tdError) # Actor根据tdError更新状态
            s = s_
            t += 1
            if isDone or t >= MAX_EP_STEPS:
                episodeRsSum = sum(trackR)
                if 'runningReward' not in globals():
                    runningReward = episodeRsSum
                else:
                    runningReward = runningReward * 0.95 + episodeRsSum * 0.05 # 更新此轮奖励
                if runningReward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True
                print('[+] episode: {}, reward: {}'.format(episode, runningReward))
                break

