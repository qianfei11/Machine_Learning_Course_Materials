#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=False)

# Visualize decoder setting
# Parameters
learningRate = 0.01 # 学习率为0.01
trainingEpochs = 5 # 训练5组
batchSize = 256 # 一组Batch共256个数据
examplesToShow = 10 # 展示10个样例

# Network Parameters
nInput = 784 # MNIST data input (img shape: 28 * 28)

# tf Graph input (only pictures)
X = tf.placeholder('float', [None, nInput])

# hidden layer settings
nHidden1 = 256 # 1st layer num features
nHidden2 = 128 # 2nd layer num features
weights = { # 权重w
    'encoderH1': tf.Variable(tf.random_normal([nInput, nHidden1])),
    'encoderH2': tf.Variable(tf.random_normal([nHidden1, nHidden2])),
    'decoderH1': tf.Variable(tf.random_normal([nHidden2, nHidden1])),
    'decoderH2': tf.Variable(tf.random_normal([nHidden1, nInput])),
}
biases = { # 偏置b
    'encoderB1': tf.Variable(tf.random_normal([nHidden1])),
    'encoderB2': tf.Variable(tf.random_normal([nHidden2])),
    'decoderB1': tf.Variable(tf.random_normal([nHidden1])),
    'decoderB2': tf.Variable(tf.random_normal([nInput])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation # 1
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoderH1']), biases['encoderB1']))
    # Decoder Hidden layer with sigmoid activation # 2
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['encoderH2']), biases['encoderB2']))
    return layer2

# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation # 1
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoderH1']), biases['decoderB1']))
    # Decoder Hidden layer with sigmoid activation # 2
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['decoderH2']), biases['decoderB2']))
    return layer2

# Construct model
encoderOp = encoder(X)
decoderOp = decoder(encoderOp)

# 因为自编码器中输入和输出相同，都设为X
# Prediction
yPred = decoderOp # 预测的结果
# Targets (Labels) are the input data.
yTrue = X # 真实的结果

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(yTrue - yPred, 2)) # 计算MSE
optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost) # 使用Adam算法优化

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    totalBatch = int(mnist.train.num_examples / batchSize) # 计算Batch的数量
    # Training cycle
    for epoch in range(trainingEpochs):
        # Loop over all batches
        for i in range(totalBatch):
            batchXs, batchYs = mnist.train.next_batch(batchSize) # max(x) = 1, min(x) = 0
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batchXs}) # 优化本层参数
        # Display logs per epoch step
        print('Epoch: {}, cost = {:.9f}'.format((epoch + 1), c))
    print('Optimization Finished!')

    # Applying encode and decode over test set
    decodeImages = sess.run(yPred, feed_dict={X:mnist.test.images[:examplesToShow]}) # 获取解压缩得到的图片
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examplesToShow):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(decodeImages[i], (28, 28)))
    plt.show()
