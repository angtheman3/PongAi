import tensorflow as tf
import cv2
import pong
import numpy as np
import random
from collections import deque

# Hyperparameters
ACTIONS = 3
GAMMA = 0.99
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
EXPLORE = 500000
OBSERVE = 50000
REPLAY_MEMORY = 500000
BATCH_SIZE = 100


def create_network():
    W_conv1 = tf.Variable(tf.zeros([8, 8, 4, 32]))
    b_conv1 = tf.Variable(tf.zeros([32]))

    W_conv2 = tf.Variable(tf.zeros([4, 4, 32, 64]))
    b_conv2 = tf.Variable(tf.zeros([64]))

    W_conv3 = tf.Variable(tf.zeros([3, 3, 64, 64]))
    b_conv3 = tf.Variable(tf.zeros([64]))

    W_fc4 = tf.Variable(tf.zeros([3136, 784]))
    b_fc4 = tf.Variable(tf.zeros([784]))

    W_fc5 = tf.Variable(tf.zeros([784, ACTIONS]))
    b_fc5 = tf.Variable(tf.zeros([ACTIONS]))

    s = tf.placeholder("float", [None, 84, 84, 4])

    conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides=[1, 4, 4, 1], padding="VALID") + b_conv1)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W_conv2, strides=[1, 2, 2, 1], padding="VALID") + b_conv2)
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides=[1, 1, 1, 1], padding="VALID") + b_conv3)

    conv3_flat = tf.reshape(conv3, [-1, 3136])
    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)
    fc5 = tf.matmul(fc4, W_fc5) + b_fc5

    return s, fc5


def train_network(input_layer, output_layer, session):
    argmax = tf.placeholder("float", [None, ACTIONS])
    gt = tf.placeholder("float", [None])

    action = tf.reduce_sum(tf.multiply(output_layer, argmax), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(action - gt))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    game = pong.PongGame()
    D = deque()

    frame = game.getPresentFrame()
    frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    input_t = np.stack((frame, frame, frame, frame), axis=2)

    saver = tf.train.Saver()
    session.run(tf.initialize_all_variables())

    t = 0
    epsilon = INITIAL_EPSILON

    while True:
        output_t = output_layer.eval(feed_dict={input_layer: [input_t]})[0]
        argmax_t = np.zeros([ACTIONS])

        if random.random() <= epsilon:
            max_index = random.randrange(ACTIONS)
        else:
            max_index = np.argmax(output_t)
        argmax_t[max_index] = 1

        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        reward_t, frame = game.getNextFrame(argmax_t)
        frame = preprocess_frame(frame)
        input_t1 = np.append(frame, input_t[:, :, 0:3], axis=2)

        D.append((input_t, argmax_t, reward_t, input_t1))

        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if t > OBSERVE:
            minibatch = random.sample(D, BATCH_SIZE)
            train_minibatch(minibatch, input_layer, output_layer, train_step, gt, argmax, session)

        input_t = input_t1
        t += 1

        if t % 10000 == 0:
            saver.save(session, './pong-dqn', global_step=t)

        print("TIMESTEP", t, "/ EPSILON", epsilon, "/ ACTION", max_index, "/ REWARD", reward_t, "/ Q_MAX %e" % np.max(output_t))


def preprocess_frame(frame):
    frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(frame, (84, 84, 1))


def train_minibatch(minibatch, input_layer, output_layer, train_step, gt, argmax, session):
    input_batch = [data[0] for data in minibatch]
    argmax_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    input_t1_batch = [data[3] for data in minibatch]

    gt_batch = []
    out_batch = output_layer.eval(feed_dict={input_layer: input_t1_batch})

    for i in range(0, len(minibatch)):
        gt_batch.append(reward_batch[i] + GAMMA * np.max(out_batch[i]))

    train_step.run(feed_dict={gt: gt_batch, argmax: argmax_batch, input_layer: input_batch})


def main():
    session = tf.InteractiveSession()
    input_layer, output_layer = create_network()
    train_network(input_layer, output_layer, session)


if __name__ == "__main__":
    main()

