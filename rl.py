
# coding: utf-8

# # Reinforcement Learning in Tensorflow Tutorial 2
# ## The Cart-Pole Task
# 
# Parts of this tutorial are based on code by [Andrej Karpathy](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5) and [korymath](https://gym.openai.com/evaluations/eval_a0aVJrGSyW892vBM04HQA).

import numpy as np 
import cPickle as pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import csv
import sys

NUM_STOCKS = 4
OUT_DIMENS = 5 # SPY, SLV, GLD, USO, Cash
IN_DIMENS = (3 * 2) + (NUM_STOCKS * 11) + OUT_DIMENS + 1 + NUM_STOCKS # input dimensionality: 3 economic factor (2 dimens), 4 securities with 11 dimens, 5 dimen prior output, equity, loss/gain on stocks

# hyperparameters
LAYER_1_NEURONS = 40 # number of hidden layer neurons
LAYER_2_NEURONS = 40 # number of hidden layer neurons
BATCH_SIZE = 30 # number of episodes before gradient descent 
BATCH_INCREMENT = 5 # after every batch, increase BATCH_SIZE by this amount (converge fast, then stabily)
LEARNING_RATE = 1e-2 # feel free to play with this to train faster or more stably.
GAMMA = 0.96 # discount factor for reward
EXPLORATION_RATE = 0.05

TRADE_THRESHOLD = 0.2 # 0.2
TRADE_FEE = 0.000
TRADE_REWARD_PENALTY = 0.005

NO_DISCOUNT = 0
EQUITY_BONUS_MULT = 0.0
AVG_REWARD_INCREMENT = 1
LOW_TRADING_PENALTY = -0.3
LOW_TRADING_THRESH = 500

DROPOUT_KEEP_PROB = 0.85

USE_DONE_REWARD = 0

INDEX_START = 427
INDEX_END = 2207
TOTAL_STEPS = INDEX_END - INDEX_START

DECAY_ITERATIONS = 50
DECAY_RATE = 0.99

print "LAYER_1_NEURONS: %d" % LAYER_1_NEURONS
print "LAYER_2_NEURONS: %d" % LAYER_2_NEURONS
print "BATCH_SIZE: %d" % BATCH_SIZE
print "LEARNING_RATE: %f" % LEARNING_RATE
print "GAMMA: %f" % GAMMA
print "IN_DIMENS: %d" % IN_DIMENS
print "OUT_DIMENS: %d" % OUT_DIMENS
print "EXPLORATION_RATE: %f" % EXPLORATION_RATE
print "TRADE_THRESHOLD: %f" % TRADE_THRESHOLD
print "TRADE_FEE: %f" % TRADE_FEE
print "TRADE_REWARD_PENALTY: %f" % TRADE_REWARD_PENALTY
print "DROPOUT_KEEP_PROB: %f" % DROPOUT_KEEP_PROB
print "NO_DISCOUNT: %d" % NO_DISCOUNT
print "AVG_REWARD_INCREMENT: %d" % AVG_REWARD_INCREMENT
print "EQUITY_BONUS_MULT: %f" % EQUITY_BONUS_MULT
print "USE_DONE_REWARD: %d" % USE_DONE_REWARD
print "LOW_TRADING_PENALTY: %f" % LOW_TRADING_PENALTY
print "LOW_TRADING_THRESH: %d" % LOW_TRADING_THRESH
print
print

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

def start_observation_state():
    state = []
    state.append(0.2) # Equally balanced portfolio to start
    state.append(0.2)
    state.append(0.2)
    state.append(0.2)
    state.append(0.2)
    state.append(1.0) # 1.0 equity to start
    state.append(1) # No gain or loss on stocks yet
    state.append(1) # No gain or loss on stocks yet
    state.append(1) # No gain or loss on stocks yet
    state.append(1) # No gain or loss on stocks yet
    return state

def discount_rewards(r, equity, trades):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0

    if NO_DISCOUNT != 0:
        for t in xrange(0, r.size):
            discounted_r[t] = r[t]
    else:
        for t in reversed(xrange(0, r.size)):
            running_add = running_add * GAMMA + r[t]
            discounted_r[t] = running_add

    if AVG_REWARD_INCREMENT != 0:
        for t in xrange(0, r.size):
            discounted_r[t] += (EQUITY_BONUS_MULT * equity / TOTAL_STEPS)

    if trades < LOW_TRADING_THRESH:
        for t in xrange(0, r.size):
            discounted_r[t] += (LOW_TRADING_THRESH - trades / LOW_TRADING_THRESH) * LOW_TRADING_PENALTY

    return discounted_r

tf.reset_default_graph()

observations = tf.placeholder(tf.float32, [None,IN_DIMENS] , name="input_x")
W1 = tf.get_variable("W1", shape=[IN_DIMENS, LAYER_1_NEURONS],
           initializer=tf.contrib.layers.xavier_initializer())
B1 = tf.Variable(tf.zeros([LAYER_1_NEURONS]), name="B1")
layer1 = tf.nn.dropout(tf.nn.bias_add(tf.matmul(observations, W1), B1), DROPOUT_KEEP_PROB)
layer1_eval = tf.nn.bias_add(tf.matmul(observations, W1), B1)

W2 = tf.get_variable("W2", shape=[LAYER_1_NEURONS, LAYER_2_NEURONS],
           initializer=tf.contrib.layers.xavier_initializer())
B2 = tf.Variable(tf.zeros([LAYER_2_NEURONS]), name="B2")
layer2 = tf.nn.dropout(tf.nn.bias_add(tf.matmul(layer1,W2), B2), DROPOUT_KEEP_PROB)
layer2_eval = tf.nn.bias_add(tf.matmul(layer1, W2), B2)

W3 = tf.get_variable("W3", shape=[LAYER_2_NEURONS, OUT_DIMENS], # 4 dimen output for each security
           initializer=tf.contrib.layers.xavier_initializer())
B3 = tf.Variable(tf.zeros([5]), name="B3")
score = tf.nn.bias_add(tf.matmul(layer2,W3), B3)
score_eval = tf.nn.bias_add(tf.matmul(layer2,W3), B3)

train_network = tf.nn.sigmoid(score) # Has DROPOUT_KEEP_PROB for neurons
eval_network  = tf.nn.sigmoid(score)  # Disables neuron dropout

# train_network = tf.nn.log_softmax(score) # Has DROPOUT_KEEP_PROB for neurons
# eval_network  = tf.nn.log_softmax(score)  # Disables neuron dropout

# Seems to get stuck at lower equity values than sigmoid
train_network = tf.nn.relu(score) # Has DROPOUT_KEEP_PROB for neurons
eval_network  = tf.nn.relu(score)  # Disables neuron dropout

#From here we define the parts of the network needed for learning a good policy.
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32,[None,5], name="input_y") # Prior output
advantages = tf.placeholder(tf.float32,name="reward_signal")

# loss = -tf.reduce_mean((tf.log(input_y - train_network)) * advantages) # add constant to avoid NaN
# loss = -tf.reduce_sum((tf.log(input_y - train_network) + 1e-7) * advantages) # add constant to avoid NaN
# loss = -tf.reduce_sum(tf.log((input_y - tf.clip_by_value(train_network,1e-5,1.0)) * advantages)) # add constant to avoid NaN
loss = -tf.reduce_sum(tf.log(tf.clip_by_value(train_network,1e-5,1.0)) * advantages) # add constant to avoid NaN

newGrads = tf.gradients(loss,tvars)

# Once we have collected a series of gradients from multiple episodes, we apply them.
# We don't just apply gradeients after every episode in order to account for noise in the reward signal.
adam = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE) # Our optimizer
W1Grad = tf.placeholder(tf.float32,name="w_grad1") # Placeholders to send the final gradients through when we update.
W2Grad = tf.placeholder(tf.float32,name="w_grad2")
W3Grad = tf.placeholder(tf.float32,name="w_grad3")
B1Grad = tf.placeholder(tf.float32,name="b_grad1") # Placeholders to send the final gradients through when we update.
B2Grad = tf.placeholder(tf.float32,name="b_grad2")
B3Grad = tf.placeholder(tf.float32,name="b_grad3")
batchGrad = [W1Grad,W2Grad,W3Grad,B1Grad,B2Grad,B3Grad]
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))


xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 100000
init = tf.initialize_all_variables()

input_data = list(csv.reader(open("data/market.csv")))
input_index = INDEX_START
equity = 1.0  # 1 unit of money, to start

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # TODO no env
    input_index = INDEX_START # Ignore csv header
    equity = 1.0  # 1 unit of money, to start
    observation = input_data[input_index][1:IN_DIMENS-OUT_DIMENS-NUM_STOCKS] # Ignore timestamp, don't want that in weights
    observation += start_observation_state()
    observation = np.array(map(float, observation))

    # Reset the gradient placeholder. We will collect gradients in 
    # gradBuffer until we are ready to update our policy network. 
    gradBuffer = sess.run(tvars)
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    min_portfolio = [1, 1, 1, 1, 1 ]
    max_portfolio = [0, 0, 0, 0, 0]
    average_portfolio = [0, 0, 0, 0, 0]
    last_portfolio = [0.2, 0.2, 0.2, 0.2, 0.2]

    gain_loss_averages = [1.0, 1.0, 1.0, 1.0]
    min_gain_loss = [10, 10, 10, 10]
    max_gain_loss = [0, 0, 0, 0]
    
    commission_fees = 0

    prev_equity = 0
    same_equity = 0
    SAME_EQUITY_EXIT = float("inf")

    current_batch = 0

    while episode_number <= total_episodes:
        if input_index == 1:
            equity = 1.0

        # print gain_loss_averages

        # Make sure the observation is in a shape the network can handle.
        x = np.reshape(observation,[1,IN_DIMENS])
        
        # Run neural network to get a portfolio, which then normalizes to 100%
        # neural_network = (eval_network) if (episode_number % BATCH_SIZE == 0) else (train_network)
        neural_network = train_network
        portfolio_raw = np.reshape(sess.run(neural_network, feed_dict={observations: x}), [NUM_STOCKS + 1])
        portfolio = normalize(portfolio_raw)

        # Add random noise to portfolio for exploration
        # print
        # print "Before Explore: %s" % portfolio
        portfolio = [p + (EXPLORATION_RATE * np.random.uniform()) for p in portfolio_raw]

        # Normalize portfolio again after random noise
        portfolio = normalize(portfolio)
        # print "After  Explore: %s" % portfolio
        # print

        if math.isnan(portfolio[0]):
            print
            print "PORTFOLIO IS NaN"
            print
            exit()

        ys.append(portfolio) # portfolio output
        xs.append(x) # observation

        portfolio_diff = 0.0
        for i in range(0, len(portfolio)):
            portfolio_diff += abs(portfolio[i] - last_portfolio[i])

        # print "Last Portfolio: %s" % last_portfolio
        # print "Diff:           %f" % portfolio_diff

        # print "Port Diff %s" % portfolio_diff
        if portfolio_diff > TRADE_THRESHOLD:
            # Portfolio changed somewhat significantly
            # So a trade fee is incurred, and the portfolio is updated
            equity -= TRADE_FEE
            commission_fees += 1

            # print
            # print "Last Portfolio: %s" % last_portfolio
            # print "    %s" % gain_loss_averages
            # print "New  Portfolio: %s" % portfolio

            # If we are increasing our position, the average purchase price drifts towards 1,
            # otherwise the average price remains unchanged
            for i in range (0, NUM_STOCKS):
                if portfolio[i] == 0 or last_portfolio[i] == 0:
                    gain_loss_averages[i] = 1.0
                else:
                    if portfolio[i] > last_portfolio[i]:
                        gain_loss_averages[i] = ((portfolio[i] - last_portfolio[i]) + last_portfolio[i] * gain_loss_averages[i]) / portfolio[i]
                    else:
                        gain_loss_averages[i] = ((last_portfolio[i] - portfolio[i]) + portfolio[i] * gain_loss_averages[i]) / last_portfolio[i]

            # print "    %s" % gain_loss_averages
            # print

            last_portfolio = portfolio

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

        # print
        # print "======================="
        # print "Time Step: %d" % input_index

        # Reward is this day's gains or losses compared to tomorrow, with some penalty for changes in portfolio
        input_index += 1
        observation = input_data[input_index][1:IN_DIMENS-OUT_DIMENS-NUM_STOCKS] # Ignore timestamp, don't want that in weights
        next_spy_change = float(observation[6])
        next_slv_change = float(observation[6+11])
        next_gld_change = float(observation[6+22])
        next_uso_change = float(observation[6+33])

        # Update averaged gain/loss
        gain_loss_averages[0] *= 1 + next_spy_change
        gain_loss_averages[1] *= 1 + next_slv_change
        gain_loss_averages[2] *= 1 + next_gld_change
        gain_loss_averages[3] *= 1 + next_uso_change

        #print "SPY: %s" % next_spy_change
        #print "SLV: %s" % next_slv_change
        #print "GLD: %s" % next_gld_change
        #print "USO: %s" % next_uso_change
        #print

        spy_reward = next_spy_change * last_portfolio[0]
        slv_reward = next_slv_change * last_portfolio[1]
        gld_reward = next_gld_change * last_portfolio[2]
        uso_reward = next_uso_change * last_portfolio[3]
        # print "SPY Reward: %s" % spy_reward
        # print "SLV Reward: %s" % slv_reward
        # print "GLD Reward: %s" % gld_reward
        # print "USO Reward: %s" % uso_reward

        profit = spy_reward + slv_reward + gld_reward + uso_reward
        equity += equity * profit
        reward = 4*profit # seems arbitrary, already tried just profit

        # The various positions in the portfolio are growing and shrinking due to growth
        # Note that for dividends this assumes automatic-reinvestment into the underlying security
        last_portfolio[0] *= (1 + next_spy_change)
        last_portfolio[1] *= (1 + next_slv_change)
        last_portfolio[2] *= (1 + next_gld_change)
        last_portfolio[3] *= (1 + next_uso_change)
        last_portfolio = normalize(last_portfolio)

        # Add new state of the portfolio to the network's input vector
        for p in last_portfolio:
            observation.append(p)
        observation.append(equity)
        for g in gain_loss_averages:
            observation.append(g)
        observation = np.array(map(float, observation))

        if portfolio_diff > TRADE_THRESHOLD:
            reward -= TRADE_REWARD_PENALTY

        # if profit < -0.20:
        #     print "Bad Day: %s" % profit
        #     print "Last Portfolio: %s" % last_portfolio
        #     print "SPY Change: %s" % next_spy_change
        #     print "SLV Change: %s" % next_slv_change
        #     print "GLD Change: %s" % next_gld_change
        #     print "USO Change: %s" % next_uso_change
        #     print "SPY Reward: %s" % spy_reward
        #     print "SLV Reward: %s" % slv_reward
        #     print "GLD Reward: %s" % gld_reward
        #     print "USO Reward: %s" % uso_reward

        # print "Reward: %s" % reward
        # print "Equity : %s" % equity 
        # print

        done = (input_index == INDEX_END) or (equity <= 0.01)

        # Final reward is portfolio's liquid value, plus 1.0 bonus for finishing
        # If didn't make it to end, final reward is % complete to end
        if done:
            if USE_DONE_REWARD != 0:
                if equity <= 0.01:
                    reward = -1 + -3 * (TOTAL_STEPS - (input_index - INDEX_START))/TOTAL_STEPS
                else:
                    reward = 5.0 * equity

            average_portfolio = [p / input_index for p in average_portfolio]

            # if episode_number % BATCH_SIZE == 0:
            #     print "===== EVAL ====="
            print "Equity: %s at time step %d" % (equity, input_index - INDEX_START)
            print "Commission fees: %d" % commission_fees
            # print "Done Reward: %s" % reward
            print "Learning Rate: %s, Exploration Rate: %s" % (LEARNING_RATE, EXPLORATION_RATE)
            print "Avg Portfolio: %s" % average_portfolio
            print "Min Portfolio: %s" % min_portfolio
            print "Max Portfolio: %s" % max_portfolio
            print "Min Gain or Loss: %s" % min_gain_loss
            print "Max Gain or Loss: %s" % max_gain_loss
            print "Iteration: %d" % episode_number
            print

            # Converged to an overfit
            if commission_fees == 0:
                same_equity += 1
                if same_equity == SAME_EQUITY_EXIT:
                    exit()
            else:
                prev_equity = equity
                same_equity == 0

            
        reward_sum += reward
        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

        if done:
            episode_number += 1

            # Don't record anything from evaluation runs
            # if episode_number % BATCH_SIZE != 1:
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            tfp = tfps
            xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[] # reset array memory

            discounted_epr = discount_rewards(epr, equity, commission_fees)

            # Reset to start
            commission_fees = 0
            equity = 1.0
            input_index = INDEX_START
            observation = input_data[input_index][1:IN_DIMENS-OUT_DIMENS-NUM_STOCKS] # Ignore timestamp, don't want that in weights
            observation += start_observation_state()

            average_portfolio = [0, 0, 0, 0, 0]
            min_portfolio = [1, 1, 1, 1, 1]
            max_portfolio = [0, 0, 0, 0, 0]


            # Don't record anything from evaluation runs
            # if episode_number % BATCH_SIZE != 1:
            # Get the gradient for this episode, and save it in the gradBuffer
            tGrad = sess.run(newGrads,feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            for ix,grad in enumerate(tGrad):
                gradBuffer[ix] += grad
                
            # If we have completed enough episodes, then update the policy network with our gradients.
            if episode_number % BATCH_SIZE == 0 and episode_number / BATCH_SIZE != current_batch:
                sess.run(updateGrads,feed_dict={W1Grad:gradBuffer[0], W2Grad:gradBuffer[1], W3Grad:gradBuffer[2], B1Grad:gradBuffer[3], B2Grad:gradBuffer[4], B3Grad:gradBuffer[5]})
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
                
                # Give a summary of how well our network is doing for each batch of episodes.
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print
                print 'Average reward for episode %f.  Total average reward %f.' % (reward_sum/BATCH_SIZE, running_reward/BATCH_SIZE)
                reward_sum = 0

                BATCH_SIZE += BATCH_INCREMENT
                current_batch += 1
                print "Increasing batch size to %d" % BATCH_SIZE
                print "Have run %d batch updates" % current_batch
                print
            
            input_index = INDEX_START # Ignore csv header
            equity = 1.0  # 1 unit of money, to start
            observation = input_data[input_index][1:IN_DIMENS-OUT_DIMENS-NUM_STOCKS] # Ignore timestamp, don't want that in weights
            observation += start_observation_state()
            observation = np.array(map(float, observation))

            if episode_number % DECAY_ITERATIONS == 0:
                EXPLORATION_RATE *= 0.97
                LEARNING_RATE *= 0.97
                LEARNING_RATE = max(5e-4, LEARNING_RATE)
                EXPLORATION_RATE = max(0.03, EXPLORATION_RATE)
        
print episode_number,'Episodes completed.'

