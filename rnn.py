import numpy as np 
import cPickle as pickle
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import matplotlib.pyplot as plt
from operator import add
import math
import csv
import sys

NUM_STOCKS = 4
OUT_DIMENS = NUM_STOCKS # SPY, SLV, GLD, USO
IN_DIMENS = (3 * 2) + (NUM_STOCKS * 18) # input dimensionality: 3 economic factor (2 dimens), 4 securities with 18 dimens

# hyperparameters
SEQUENCE_LENGTH = 10
RNN_NEURONS = 100 # number of hidden layer neurons
RNN_LAYERS = 2 # number of hidden layer neurons
BATCH_SIZE = 20 # number of episodes before gradient descent 
BATCH_INCREMENT = 2 # after every batch, increase BATCH_SIZE by this amount (converge fast, then stabily)
LEARNING_RATE = 0.01 # feel free to play with this to train faster or more stably.
GAMMA = 0.96 # discount factor for reward
EXPLORATION_RATE = 0.05
TRADE_EXPLORATION_RATE = 0.05

TRADE_THRESHOLD = 0.5 # 0.2
TRADE_THRESHOLD_MULTIPLIER = 1.0 # NN outputs 0->1, but full range should be 0->2 because (sum(abs(port[i]-prev_port[i])))
TRADE_FEE = 0.0002
TRADE_REWARD_PENALTY = 0.00

NO_DISCOUNT = 0

AVG_REWARD_INCREMENT = 0
EQUITY_BONUS_MULT = 10.0
USE_Q_TABLE = 0
Q_TABLE_MULT = 0.05
LOW_TRADING_PENALTY = -0.3
LOW_TRADING_THRESH = 0

DROPOUT_KEEP_PROB = 0.90

USE_DONE_REWARD = 0

INDEX_START = 427 # October 8, 2009
INDEX_END = 2207
TOTAL_STEPS = INDEX_END - INDEX_START

# Number of days in the future that we're trying to predict
FUTURE_FORECAST = 10

TRAIN_START = INDEX_START
TRAIN_END = INDEX_END - 700 - FUTURE_FORECAST
TEST_START = INDEX_END - 700 - FUTURE_FORECAST
TEST_END = INDEX_END - FUTURE_FORECAST

DECAY_ITERATIONS = 100
DECAY_RATE = 0.99

print "SEQUENCE_LENGTH: %d" % SEQUENCE_LENGTH
print "RNN_NEURONS: %d" % RNN_NEURONS
print "RNN_LAYERS: %d" % RNN_LAYERS
print "BATCH_SIZE: %d" % BATCH_SIZE
print "LEARNING_RATE: %f" % LEARNING_RATE
print "GAMMA: %f" % GAMMA
print "IN_DIMENS: %d" % IN_DIMENS
print "OUT_DIMENS: %d" % OUT_DIMENS
print "EXPLORATION_RATE: %f" % EXPLORATION_RATE
print "TRADE_EXPLORATION_RATE: %f" % TRADE_EXPLORATION_RATE
print "TRADE_THRESHOLD: %f" % TRADE_THRESHOLD
print "TRADE_FEE: %f" % TRADE_FEE
print "TRADE_REWARD_PENALTY: %f" % TRADE_REWARD_PENALTY
print "DROPOUT_KEEP_PROB: %f" % DROPOUT_KEEP_PROB
print "NO_DISCOUNT: %d" % NO_DISCOUNT
print "AVG_REWARD_INCREMENT: %d" % AVG_REWARD_INCREMENT
print "USE_Q_TABLE: %d" % USE_Q_TABLE
print "Q_TABLE_MULT: %d" % Q_TABLE_MULT
print "EQUITY_BONUS_MULT: %f" % EQUITY_BONUS_MULT
print "USE_DONE_REWARD: %d" % USE_DONE_REWARD
print "LOW_TRADING_PENALTY: %f" % LOW_TRADING_PENALTY
print "LOW_TRADING_THRESH: %d" % LOW_TRADING_THRESH
print
print

def get_observation(index):
    step = input_data[index][1:-1] # Don't use first column (timestamp), or last column (is "\0")
    step = np.array(map(float, step))
    return step

def get_next_sequence(index):
    sequence = []
    for i in range(index - SEQUENCE_LENGTH, index):
        sequence.append(get_observation(i))
    return sequence 

def get_next_profit(index):
    observation = get_observation(index + 1)
    next_spy_change = float(observation[6])
    next_slv_change = float(observation[6+18])
    next_gld_change = float(observation[6+36])
    next_uso_change = float(observation[6+54])
    profits = [next_spy_change, next_slv_change, next_gld_change, next_uso_change]
    return np.array(map(float, profits))

# Use min-max normalization to avoid issues with negative numbers
def normalize(portfolio):
    p_min = min(portfolio)
    p_max = max(portfolio)
    p_diff = p_max - p_min

    if p_diff == 0:
        min_max_normal = [ 1 for p in portfolio ]
    else:
        min_max_normal = [ (p - p_min) / p_diff for p in portfolio ]
    
    mm_sum = sum(min_max_normal)
    normalized = [ p / mm_sum for p in min_max_normal ]

    # print "%s -> %s" % (portfolio, normalized)
    return normalized

tf.reset_default_graph()

x = tf.placeholder("float", [SEQUENCE_LENGTH, IN_DIMENS])
y = tf.placeholder("float", [OUT_DIMENS])

learning_rate = tf.placeholder(tf.float32, shape=[])

# Define weights
weights = {
           'out': tf.Variable(tf.random_normal([RNN_NEURONS, OUT_DIMENS]))
          }
biases =  {
           'out': tf.Variable(tf.random_normal([OUT_DIMENS]))
          }

def RNN(x, weights, biases):
    #x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, IN_DIMENS])
    x = tf.split(0, SEQUENCE_LENGTH, x)
    lstm_cell = rnn_cell.BasicLSTMCell(RNN_NEURONS, forget_bias = 1.0)
    stacked_lstm = rnn_cell.MultiRNNCell([lstm_cell] * RNN_LAYERS)
    outputs, states = rnn.rnn(stacked_lstm, x, dtype=tf.float32)
    return (outputs, states, tf.matmul(outputs[-1], weights['out']) + biases['out'])

rnn_outputs, rnn_states, rnn_pred = RNN(x, weights, biases)

cost = tf.reduce_sum(tf.square(rnn_pred - y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

init = tf.initialize_all_variables()

input_data = list(csv.reader(open("data/market.csv")))

forecast_move = []
sequences = []
for index in range(INDEX_START, INDEX_END - FUTURE_FORECAST):
    change = get_next_profit(index)
    for i in range(1, FUTURE_FORECAST):
        change = map(add, change, get_next_profit(index + i))
    forecast_move.append(change)

    sequences.append(get_next_sequence(index))

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    RNN_TRAINING_ITERATIONS = 100
    iteration = 0

    while iteration < RNN_TRAINING_ITERATIONS:
        iteration += 1

        train_loss = 0
        test_loss = 0

        for index in range(TRAIN_START, TRAIN_END):
            index -= INDEX_START

            sequence_x = sequences[index]
            sess.run(optimizer, feed_dict = {x: sequence_x, y: forecast_move[index], learning_rate: LEARNING_RATE})

            train_loss += sess.run(cost, feed_dict = {x: sequence_x, y: forecast_move[index]})

            if iteration % 5 == 0 and index % 100 == 0:
                print "Index %d" % index
                print "Next: %s" % forecast_move[index]
                print "Pred: %s" % sess.run(rnn_pred, feed_dict = {x: sequence_x})
                print

        if iteration % 5 == 0:
            print "TEST"

        for index in range(TEST_START, TEST_END):
            index -= TEST_START
            test_loss += sess.run(cost, feed_dict = {x: sequence_x, y: forecast_move[index]})

            if iteration % 5 == 0 and index % 100 == 0:
                print "Index %d" % index
                print "Next: %s" % forecast_move[index]
                print "Pred: %s" % sess.run(rnn_pred, feed_dict = {x: sequence_x})
                print

        print "Iteration %d: train loss %f, test loss %f" % (iteration, train_loss, test_loss)

        LEARNING_RATE *= 0.99
        print "Learning Rate: %f" % LEARNING_RATE


