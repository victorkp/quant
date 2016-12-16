import numpy as np 
import cPickle as pickle
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from operator import add
from operator import sub
import math
import csv
import sys

NUM_STOCKS = 4

# Output relative portfolio values (SPY, SLV, GLD, USO, Cash) and trade threshhold
RL_OUT_DIMENS = (NUM_STOCKS + 1) + 1

# From market.csv
### 3 economic factor (2 dimens)
### 4 securities with 18 dimens
OBSERVATION_DIMENS = (3 * 2) + (NUM_STOCKS * 18)

# RNN Input dimensionality:
### 3 economic factor (2 dimens)
### 4 securities with 18 dimens
### equity
### current loss/gain on stocks
### boolean if trade happened
### RL_OUT_DIMENS dimen prior output (previous "action")
RNN_IN_DIMENS = OBSERVATION_DIMENS + 1 + NUM_STOCKS + 1 + RL_OUT_DIMENS

# RNN Output dimensionality:
### 3 economic factor (2 dimens)
### 4 securities with 18 dimens
RNN_OUT_DIMENS = 4 #OBSERVATION_DIMENS

# hyperparameters
SEQUENCE_LENGTH = 10
RNN_NEURONS = RNN_IN_DIMENS / 2     # number of hidden layer neurons in the RNN
RNN_LAYERS = 3                      # number of layers of neurons in the RNN

RL_LAYER_1_NEURONS = RNN_NEURONS           # Neurons in the RL-NN's first layer
RL_LAYER_2_NEURONS = 2 * RNN_NEURONS / 3   # Neurons in the RL-NN's second layer
RL_LAYER_3_NEURONS = RNN_NEURONS / 2       # Neurons in the RL-NN's third layer

BATCH_SIZE = 8 # number of episodes before gradient descent 
BATCH_INCREMENT = 2 # after every batch, increase BATCH_SIZE by this amount (converge fast, then stabily)
LEARNING_RATE = 0.01 # feel free to play with this to train faster or more stably.
GAMMA = 0.95 # discount factor for reward
EXPLORATION_RATE = 0.05
TRADE_EXPLORATION_RATE = 0.03

TRADE_THRESHOLD = 0.25 # 0.2
TRADE_THRESHOLD_MULTIPLIER = 1.0 # NN outputs 0->1, but full range should be 0->2 because (sum(abs(port[i]-prev_port[i])))
TRADE_FEE = 0 #.0002
TRADE_REWARD_PENALTY = 0.02

RNN_FORECAST = 7

NO_DISCOUNT = 0

AVG_REWARD_INCREMENT = 0
EQUITY_BONUS_MULT = 2.0
USE_Q_TABLE = 0
Q_TABLE_MULT = 0.05
LOW_TRADING_PENALTY = -1.0
LOW_TRADING_THRESH = 50

DROPOUT_KEEP_PROB = 0.90

USE_DONE_REWARD = 0

INDEX_START = 427 # October 8, 2009
INDEX_END = 2207 - RNN_FORECAST
TOTAL_STEPS = INDEX_END - INDEX_START

TRAIN_START = INDEX_START
TRAIN_END = INDEX_END 

DECAY_ITERATIONS = 100
DECAY_RATE = 0.99

print "SEQUENCE_LENGTH: %d" % SEQUENCE_LENGTH
print "RNN_NEURONS: %d" % RNN_NEURONS
print "RNN_LAYERS: %d" % RNN_LAYERS
print "RL_LAYER_1_NEURONS: %d" % RL_LAYER_1_NEURONS
print "RL_LAYER_2_NEURONS: %d" % RL_LAYER_2_NEURONS
print "RL_LAYER_3_NEURONS: %d" % RL_LAYER_3_NEURONS
print "BATCH_SIZE: %d" % BATCH_SIZE
print "LEARNING_RATE: %f" % LEARNING_RATE
print "GAMMA: %f" % GAMMA
print "RNN_IN_DIMENS: %d" % RNN_IN_DIMENS
print "RL_OUT_DIMENS: %d" % RL_OUT_DIMENS
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
print

def initial_observation_state():
    state = []
    state.append(1.0) # 1.0 equity to start

    state.append(1.0) # No gain or loss on stocks yet
    state.append(1.0) # No gain or loss on stocks yet
    state.append(1.0) # No gain or loss on stocks yet
    state.append(1.0) # No gain or loss on stocks yet

    state.append(0.0) # No prior trade

    state.append(0.2) # Equally balanced portfolio to start
    state.append(0.2)
    state.append(0.2)
    state.append(0.2)
    state.append(0.2)

    state.append(0.5) # Some mid-point for trade threshold

    return state

def get_initial_sequence(index):
    sequence = []
    for i in reversed(range(0, SEQUENCE_LENGTH)):
        # sequence.append(get_observation(index - i, None))
        sequence.append(get_observation(index - i, initial_observation_state()))
    return sequence

# Returns a [1 x RNN_IN_DIMENS] vector to input into the RNN
def get_observation(index, other_state):
    step = input_data[index][1:-1] # Don't use first column (timestamp), or last column (is "\0")
    if other_state is not None:
        step += other_state
    step = np.array(map(float, step))
    return step

def get_next_sequence(index):
    sequence = []
    for i in range(index - SEQUENCE_LENGTH, index):
        sequence.append(get_observation(i))
    return sequence 

# Return weighted averages of values over next RNN_FORECAST days
def get_rnn_target(index):
    observation = get_observation(index, None)
    next_spy_change = float(observation[6])
    next_slv_change = float(observation[6+18])
    next_gld_change = float(observation[6+36])
    next_uso_change = float(observation[6+54])

    target = [next_spy_change, next_slv_change, next_gld_change, next_uso_change]
    denominator = 1.0
    factor = 1.0
    
    for i in range(1, RNN_FORECAST):
        denominator += factor
        observation = get_observation(i, None)

        next_spy_change = float(observation[6])
        next_slv_change = float(observation[6+18])
        next_gld_change = float(observation[6+36])
        next_uso_change = float(observation[6+54])
        new_target = [next_spy_change, next_slv_change, next_gld_change, next_uso_change]

        for j in range(0, len(target)):
            target[j] += new_target[j] * factor
        factor *= 0.75

    for i in range(1, len(target)):
        target[i] /= denominator

    return target


def get_next_profit(index):
    observation = get_observation(index + 1, None)
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

def discount_rewards(r, equity, equities, trades):
    """ take 1D float array of rewards and compute discounted reward """
    # (Number of time steps) X (6 Reward Signals)
    discounted_r = np.zeros((r.shape[0], RL_OUT_DIMENS), dtype=float)

    if NO_DISCOUNT == 0:
        running_add = [0, 0, 0, 0, 0, 0]
        for t in reversed(xrange(0, r.shape[0])):
            for i in range(0, RL_OUT_DIMENS - 1):
                running_add[i] = running_add[i] * GAMMA + r[t][i]
                discounted_r[t][i] += running_add[i] + equities[t]

            # Unadultered trade reward
            discounted_r[t][5] = r[t][5]

    if EQUITY_BONUS_MULT != 0:
        for t in reversed(xrange(0, r.shape[0])):
            for i in range(0, RL_OUT_DIMENS - 1):
                discounted_r[t][i] += EQUITY_BONUS_MULT * (equities[t] - 1.0)
                

    # No trade reward or penalty
    # discounted_r[t][5] = 0

    return discounted_r


tf.reset_default_graph()

rnn_x = tf.placeholder("float", [SEQUENCE_LENGTH, RNN_IN_DIMENS])
rnn_y = tf.placeholder("float", [RNN_OUT_DIMENS])

rnn_learning_rate = tf.placeholder(tf.float32, shape=[])

# Define weights
rnn_weights = {
              'out': tf.Variable(tf.random_normal([RNN_NEURONS, RNN_OUT_DIMENS]))
             }
rnn_biases =  {
               'out': tf.Variable(tf.random_normal([RNN_OUT_DIMENS]))
              }


# Recurrent Neural Network accepts observation as input
def RNN(x, weights, biases):
    x = tf.reshape(x, [-1, RNN_IN_DIMENS])
    x = tf.split(0, SEQUENCE_LENGTH, x)
    lstm_cell = rnn_cell.BasicLSTMCell(RNN_NEURONS, forget_bias = 1.0, state_is_tuple=False)
    stacked_lstm = rnn_cell.MultiRNNCell([lstm_cell] * RNN_LAYERS, state_is_tuple=False)
    outputs, states = rnn.rnn(stacked_lstm, x, dtype=tf.float32)
    return (outputs, states, tf.matmul(outputs[-1], weights['out']) + biases['out'])

# Reinforcement Learning Neural Network that takes RNN's internal state as input,
# and builds a portfolio composition from that
def RL_NN():
    rnn_state_input = tf.placeholder(tf.float32, [None, 2 * RNN_NEURONS * RNN_LAYERS], name="rnn_state")
    rl_dropout = tf.placeholder(tf.float32, [])

    W1 = tf.get_variable("W1", shape=[2 * RNN_NEURONS * RNN_LAYERS, RL_LAYER_1_NEURONS],
               initializer=tf.contrib.layers.xavier_initializer())
    B1 = tf.Variable(tf.zeros([RL_LAYER_1_NEURONS]), name="B1")
    layer1 = tf.nn.dropout(tf.nn.bias_add(tf.matmul(rnn_state_input, W1), B1), rl_dropout)

    W2 = tf.get_variable("W2", shape=[RL_LAYER_1_NEURONS, RL_LAYER_2_NEURONS],
               initializer=tf.contrib.layers.xavier_initializer())
    B2 = tf.Variable(tf.zeros([RL_LAYER_2_NEURONS]), name="B2")
    layer2 = tf.nn.dropout(tf.nn.bias_add(tf.matmul(layer1,W2), B2), rl_dropout)

    W3 = tf.get_variable("W3", shape=[RL_LAYER_2_NEURONS, RL_LAYER_3_NEURONS],
               initializer=tf.contrib.layers.xavier_initializer())
    B3 = tf.Variable(tf.zeros([RL_LAYER_3_NEURONS]), name="B3")
    layer3 = tf.nn.dropout(tf.nn.bias_add(tf.matmul(layer2,W3), B3), rl_dropout)

    W4 = tf.get_variable("W4", shape=[RL_LAYER_3_NEURONS, RL_OUT_DIMENS], # Output for each security, cash, trade 
               initializer=tf.contrib.layers.xavier_initializer())
    B4 = tf.Variable(tf.zeros([RL_OUT_DIMENS]), name="B4")
    score = tf.nn.bias_add(tf.matmul(layer3,W4), B4)

    train_network = tf.nn.sigmoid(score)

    input_y = tf.placeholder(tf.float32, [None,RL_OUT_DIMENS], name="input_y") # Prior output
    reward_signal = tf.placeholder(tf.float32, [None, RL_OUT_DIMENS], name="reward_signal")

    loss = -tf.reduce_sum(train_network * reward_signal) # add constant to avoid NaN
    #loss = -tf.reduce_sum(tf.square(train_network - input_y) * reward_signal) # add constant to avoid NaN

    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss) # Our optimizer

    return (rnn_state_input, input_y, reward_signal, train_network, learning_rate, optimizer, rl_dropout)

# RL NN inputs, learning rate, optimizer
rl_input, rl_y, rl_reward, rl_predictor, rl_learning_rate, rl_optimizer, rl_dropout = RL_NN()

# RNN Predictor, Loss, and Optimizer
rnn_outputs, rnn_states, rnn_pred = RNN(rnn_x, rnn_weights, rnn_biases)
rnn_cost = tf.reduce_sum(tf.square(rnn_pred - rnn_y))
rnn_optimizer = tf.train.AdamOptimizer(learning_rate = rnn_learning_rate).minimize(rnn_cost)

init = tf.initialize_all_variables()

input_data = list(csv.reader(open("data/market.csv")))

rnn_targets = []
for i in range(TRAIN_START, TRAIN_END):
    rnn_targets.append(get_rnn_target(i))



# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # RNN bootstrap
    rnn_bootstrap_iters = 5
    bootstrap_learn_rate = LEARNING_RATE
    for iter in range(0,rnn_bootstrap_iters):
        print "Bootstrapping RNN %d/%d" % (iter, rnn_bootstrap_iters)
        sequence_x = get_initial_sequence(INDEX_START)
        for i in range(INDEX_START, INDEX_END):
            sess.run(rnn_optimizer, feed_dict = {rnn_x: sequence_x,
                                                 rnn_y: rnn_targets[i - INDEX_START],
                                                 rnn_learning_rate: bootstrap_learn_rate})
            sequence_x.pop(0)
            sequence_x.append(input_data[i][1:-1] + initial_observation_state())
        bootstrap_learn_rate *= 0.95

    RNN_TRAINING_ITERATIONS = 10000
    iteration = 0

    # Batch data for batch update
    batch_rl_x      = []
    batch_rl_y      = []
    batch_rl_reward = []
    batch_rnn_x     = []

    # Data for one episode
    episode_rl_x      = []
    episode_rl_y      = []
    episode_rl_reward = []
    episode_rnn_x     = []

    equity = 1.0  # 1 unit of money, to start
    equities = [1.0]  # Array of equity at each time step

    # Just statistics, don't play into RL NN or RNN 
    min_portfolio = [1, 1, 1, 1, 1 ]
    max_portfolio = [0, 0, 0, 0, 0]
    average_portfolio = [0, 0, 0, 0, 0]
    last_portfolio = [0.2, 0.2, 0.2, 0.2, 0.2]
    trade_thresh_average = 0.5
    trade_thresh_min = 10.0
    trade_thresh_max = 0.0

    # Keep track of average profit or loss for each equity, fed
    # in as part of RNN's input
    gain_loss_averages = [1.0, 1.0, 1.0, 1.0]
    min_gain_loss = [10, 10, 10, 10]
    max_gain_loss = [0, 0, 0, 0]
    
    # Bookkeeping for number of trades, last trade, profit since last trade
    trade_count = 0
    last_trade = INDEX_START
    last_trade_equity = 1.0

    next_update = BATCH_SIZE

    sequence = []

    while iteration < RNN_TRAINING_ITERATIONS:
        iteration += 1

        train_loss = 0
        test_loss = 0

        for index in range(TRAIN_START, TRAIN_END):
            index -= INDEX_START

            if index == 0:
                equity = 1.0
                commission_fees = 0
                equities = [1.0]  # Array of equity at each time step
                trade_thresh_average = 0.5
                trade_thresh_min = 10.0
                trade_thresh_max = 0.0
                last_trade = INDEX_START
                last_trade_equity = 1.0
                sequence = get_initial_sequence(INDEX_START)
                episode_rl_x      = []
                episode_rl_y      = []
                episode_rl_reward = []
                episode_rnn_x     = []

            # Periodically run without dropout to get an idea of performance
            keep_prob = DROPOUT_KEEP_PROB
            if iteration % 10 == 0:
                if index == 0:
                    print
                    print "=== EVALUATION RUN ==="
                keep_prob = 1.0

            # Run input sequence through RNN...
            # sequence_array = np.reshape(sequence, [SEQUENCE_LENGTH, RNN_IN_DIMENS])
            last_rnn_state, rnn_output = sess.run((rnn_states, rnn_outputs), feed_dict = {rnn_x: sequence})

            # ... then use RNN's internal state as input to an RL NN which does the portfolio composition
            rl_input_x = np.reshape(last_rnn_state, [1, 2 * RNN_NEURONS * RNN_LAYERS])
            portfolio_raw = list(np.reshape(sess.run(rl_predictor, feed_dict={rl_input: rl_input_x, rl_dropout: keep_prob}), [RL_OUT_DIMENS]))

            trade_threshold = portfolio_raw.pop() * TRADE_THRESHOLD_MULTIPLIER
            trade_thresh_average = trade_thresh_average * (0.99) + 0.01 * trade_threshold
            trade_thresh_min = min(trade_thresh_min, trade_threshold)
            trade_thresh_max = max(trade_thresh_max, trade_threshold)

            # Add random noise to portfolio for exploration and re-normalize
            portfolio = normalize(portfolio_raw)
            portfolio = [p + (EXPLORATION_RATE * np.random.uniform()) for p in portfolio_raw]
            portfolio = normalize(portfolio)

            if math.isnan(portfolio[0]):
                print "PORTFOLIO IS NaN"
                exit()

            # Determine if a trade occurs
            portfolio_diff = 0.0
            for i in range(0, len(portfolio)):
                portfolio_diff += abs(portfolio[i] - last_portfolio[i])

            make_trade = 1.0 if (portfolio_diff > trade_threshold or np.random.uniform() < TRADE_EXPLORATION_RATE) else 0.0

            if make_trade > 0.0:
                # Portfolio changed somewhat significantly
                # So a trade fee is incurred, and the portfolio is updated
                equity -= TRADE_FEE
                commission_fees += 1
                last_trade = index
                trade_profit = (equity - last_trade_equity) / last_trade_equity
                last_trade_equity = equity

                trade_stocks_profit = gain_loss_averages[:]
                for i in range (0, NUM_STOCKS):
                    if portfolio[i] == 0 or last_portfolio[i] == 0:
                        gain_loss_averages[i] = 1.0
                    else:
                        if portfolio[i] > last_portfolio[i]:
                            gain_loss_averages[i] = ((portfolio[i] - last_portfolio[i]) + last_portfolio[i] * gain_loss_averages[i]) / portfolio[i]
                        else:
                            gain_loss_averages[i] = ((last_portfolio[i] - portfolio[i]) + portfolio[i] * gain_loss_averages[i]) / last_portfolio[i]
                
                trade_stocks_profit = map(sub, trade_stocks_profit, gain_loss_averages)

                last_portfolio = portfolio
            else:
                trade_profit = 0.0

            # Keep track of RNN and RL inputs and RL's Y value
            episode_rnn_x.append(np.array(sequence))
            episode_rl_x.append(rl_input_x)

            # Update statistics
            average_portfolio[0] += last_portfolio[0]
            average_portfolio[1] += last_portfolio[1]
            average_portfolio[2] += last_portfolio[2]
            average_portfolio[3] += last_portfolio[3]
            average_portfolio[4] += last_portfolio[4]

            for i in range(0, NUM_STOCKS + 1):
                min_portfolio[i] = min(last_portfolio[i], min_portfolio[i])
                max_portfolio[i] = max(last_portfolio[i], max_portfolio[i])

            for i in range(0, NUM_STOCKS):
                min_gain_loss[i] = min(gain_loss_averages[i], min_gain_loss[i])
                max_gain_loss[i] = max(gain_loss_averages[i], max_gain_loss[i])


            # Advance a timestep
            index += 1
            next_observation = input_data[index][1:-1] # Don't use first column (timestamp), or last column (is "\0")
            
            # Update from next time observation
            next_spy_change = float(next_observation[6])
            next_slv_change = float(next_observation[6+18])
            next_gld_change = float(next_observation[6+36])
            next_uso_change = float(next_observation[6+54])
            gain_loss_averages[0] *= 1 + next_spy_change
            gain_loss_averages[1] *= 1 + next_slv_change
            gain_loss_averages[2] *= 1 + next_gld_change
            gain_loss_averages[3] *= 1 + next_uso_change
            spy_reward = next_spy_change * last_portfolio[0]
            slv_reward = next_slv_change * last_portfolio[1]
            gld_reward = next_gld_change * last_portfolio[2]
            uso_reward = next_uso_change * last_portfolio[3]
            profit = spy_reward + slv_reward + gld_reward + uso_reward
            equity += equity * profit

            equities.append(equity)

            # RL's Y value is mostly about ensuring that trade_threshold is something reasonable,
            # and also assigning a higher weight to best stock

            #max_stock = max(next_spy_change, next_gld_change, next_slv_change, next_uso_change, 0)
            #episode_rl_y.append([1 if next_spy_change == max_stock else 0,
            #                     1 if next_slv_change == max_stock else 0,
            #                     1 if next_gld_change == max_stock else 0,
            #                     1 if next_uso_change == max_stock else 0,
            #                     1 if 0 == max_stock else 0,
            #                     0.5]) # portfolio output

            episode_rl_y.append([0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0.5]) # portfolio output

            # The various positions in the portfolio are growing and shrinking due to growth
            # Note that for dividends this assumes automatic-reinvestment into the underlying security
            last_portfolio[0] *= (1 + next_spy_change)
            last_portfolio[1] *= (1 + next_slv_change)
            last_portfolio[2] *= (1 + next_gld_change)
            last_portfolio[3] *= (1 + next_uso_change)
            last_portfolio = normalize(last_portfolio)

            # Generate a reward signal for the RL neural net
            # if make_trade:
            #     reward = trade_stocks_profit + [0, trade_profit - TRADE_REWARD_PENALTY]
            # else:
            #      if index - last_trade < 100:
            #         # reward = [0, 0, 0, 0, 0, 0]
            #         reward = [next_spy_change, next_slv_change, next_gld_change, next_uso_change, 0, 0]
            #      else:
            #         # reward = [0, 0, 0, 0, 0, LOW_TRADING_PENALTY * ((index - last_trade) / 100.0)]
            #         reward = [next_spy_change, next_slv_change, next_gld_change, next_uso_change, 0, LOW_TRADING_PENALTY * ((index - last_trade) / 100.0)]
            # reward = [next_spy_change, next_slv_change, next_gld_change, next_uso_change, 0, 0]
            reward = [spy_reward, slv_reward, gld_reward, uso_reward, 0, (trade_profit - TRADE_REWARD_PENALTY) if (make_trade != 0) else 0]

            episode_rl_reward.append(reward)

            # Add to RNN's expected output
            next_observation = map(float, next_observation)

            # Set up sequence for next run of RNN
            next_observation.append(equity)
            next_observation.extend(gain_loss_averages)
            next_observation.append(make_trade)
            next_observation.extend(last_portfolio)
            next_observation.append(trade_threshold)
            sequence.pop(0)
            sequence.append(np.array(next_observation))


        ### Print end of episode statistics

        average_portfolio = [p / (INDEX_END - INDEX_START) for p in average_portfolio]

        print "Equity: %s" % equity
        print "Commission fees: %d = %f in fees" % (commission_fees, commission_fees * TRADE_FEE)
        print "Average trade threshold: %f (%f -> %f)" % (trade_thresh_average, trade_thresh_min, trade_thresh_max)
        # print "Done Reward: %s" % reward
        if iteration % min(40, BATCH_SIZE) == 0:  # Don't flood output so often
            print "Learning Rate: %s, Exploration Rate: %s" % (LEARNING_RATE, EXPLORATION_RATE)
            print "Avg Portfolio: %s" % average_portfolio
            print "Min Portfolio: %s" % min_portfolio
            print "Max Portfolio: %s" % max_portfolio
            print "Min Gain or Loss: %s" % min_gain_loss
            print "Max Gain or Loss: %s" % max_gain_loss
        print "Iteration: %d" % iteration
        print


        # Discount this episode's rewards, x, y and add to batch
        episode_x = np.vstack(episode_rl_x)
        episode_y = np.vstack(episode_rl_y)
        episode_r = np.vstack(episode_rl_reward)
        discounted_r = discount_rewards(episode_r, equity, equities, commission_fees)

        batch_rl_x.append(episode_x)
        batch_rl_y.append(episode_y)
        batch_rl_reward.append(discounted_r)
        batch_rnn_x.append(episode_rnn_x)

        if iteration == next_update:
            next_update += BATCH_SIZE
            BATCH_SIZE += BATCH_INCREMENT

            # Do our batch update
            for i in range(0, len(batch_rl_x)):
                # Run batch update on RL optimizer
                sess.run(rl_optimizer, feed_dict={rl_input: batch_rl_x[i],
                                                  rl_y: batch_rl_y[i],
                                                  rl_reward: batch_rl_reward[i],
                                                  rl_learning_rate: LEARNING_RATE,
                                                  rl_dropout: DROPOUT_KEEP_PROB })

                # Some more randomly selected updates on RNN (stop at 1100, because that's end of training set)
                for j in range(0, min(1100, len(batch_rnn_x[i]))):
                    if np.random.uniform() > 0.9:
                        sess.run(rnn_optimizer, feed_dict = {rnn_x: batch_rnn_x[i][j],
                                                             rnn_y: rnn_targets[j],
                                                             rnn_learning_rate: LEARNING_RATE})


            LEARNING_RATE *= 0.99
            print
            print "Learning Rate: %f" % LEARNING_RATE
            print

            # Reset batch data
            batch_rl_x = []
            batch_rl_y = []
            batch_rl_reward = []
            batch_rnn_x = []


