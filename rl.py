
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

# hyperparameters
H = 40 # number of hidden layer neurons
H2 = 40 # number of hidden layer neurons
batch_size = 50 # every how many episodes to do a param update?
learning_rate = 5e-3 # feel free to play with this to train faster or more stably.
gamma = 0.8 # discount factor for reward
OUT_DIMENS = 5 # SPY, SLV, GLD, USO, Cash
IN_DIMENS = (3 * 2) + (4 * 11) + OUT_DIMENS + 1 # input dimensionality: 3 economic factor (2 dimens), 4 securities with 11 dimens, 5 dimen prior output, equity
exploration_rate = 0.1

TRADE_THRESHOLD = 0.2 # 0.2
TRADE_FEE = 0.000
TRADE_REWARD_PENALTY = 0.001

NO_DISCOUNT = 0
AVG_REWARD_INCREMENT = 1
LOW_TRADING_PENALTY = -0.1
LOW_TRADING_THRESH = 150

DROPOUT_KEEP_PROB = 0.8

USE_DONE_REWARD = 0

TOTAL_STEPS = 2207

print "H: %d" % H
print "H2: %d" % H2
print "batch_size: %d" % batch_size
print "learning_rate: %f" % learning_rate
print "gamma: %f" % gamma
print "IN_DIMENS: %d" % IN_DIMENS
print "OUT_DIMENS: %d" % OUT_DIMENS
print "exploration_rate: %f" % exploration_rate
print "TRADE_THRESHOLD: %f" % TRADE_THRESHOLD
print "TRADE_FEE: %f" % TRADE_FEE
print "TRADE_REWARD_PENALTY: %f" % TRADE_REWARD_PENALTY
print "DROPOUT_KEEP_PROB: %f" % DROPOUT_KEEP_PROB
print "NO_DISCOUNT: %d" % NO_DISCOUNT
print "AVG_REWARD_INCREMENT: %d" % AVG_REWARD_INCREMENT
print "USE_DONE_REWARD: %d" % USE_DONE_REWARD
print "LOW_TRADING_PENALTY: %f" % LOW_TRADING_PENALTY
print "LOW_TRADING_THRESH: %d" % LOW_TRADING_THRESH

# Use min-max normalization to avoid issues with negative numbers
def normalize(portfolio):
    p_min = min(portfolio)
    p_max = max(portfolio)
    p_diff = p_max - p_min
    min_max_normal = [ (p - p_min) / p_diff for p in portfolio ]
    
    mm_sum = sum(min_max_normal)
    normalized = [ p / mm_sum for p in min_max_normal ]

    # print "%s -> %s" % (portfolio, normalized)
    return normalized

def discount_rewards(r, equity, trades):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0

    if NO_DISCOUNT != 0:
        for t in xrange(0, r.size):
            discounted_r[t] = r[t]
    else:
        for t in reversed(xrange(0, r.size)):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add

    if AVG_REWARD_INCREMENT != 0:
        for t in xrange(0, r.size):
            discounted_r[t] += (equity / TOTAL_STEPS)

    if trades < LOW_TRADING_THRESH:
        for t in xrange(0, r.size):
            discounted_r[t] += LOW_TRADING_PENALTY

    return discounted_r

# In[5]:

tf.reset_default_graph()

#This defines the network as it goes from taking an observation of the environment to 
#giving a probability of chosing to the action of moving left or right.
observations = tf.placeholder(tf.float32, [None,IN_DIMENS] , name="input_x")
W1 = tf.get_variable("W1", shape=[IN_DIMENS, H],
           initializer=tf.contrib.layers.xavier_initializer())
B1 = tf.Variable(tf.zeros([H]), name="B1")
layer1 = tf.nn.dropout(tf.add(tf.matmul(observations, W1), B1), DROPOUT_KEEP_PROB)
#layer1 = tf.nn.relu(tf.add(tf.matmul(observations, W1), B1))

# layer1 = tf.nn.relu(tf.matmul(observations, W1))

W2 = tf.get_variable("W2", shape=[H, H2],
           initializer=tf.contrib.layers.xavier_initializer())
B2 = tf.Variable(tf.zeros([H2]), name="B2")
layer2 = tf.nn.dropout(tf.nn.bias_add(tf.matmul(layer1,W2), B2), DROPOUT_KEEP_PROB)
#layer2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(layer1,W2), B2))

#layer2 = tf.nn.relu(tf.matmul(layer1, W2))

W3 = tf.get_variable("W3", shape=[H2, OUT_DIMENS], # 4 dimen output for each security
           initializer=tf.contrib.layers.xavier_initializer())
B3 = tf.Variable(tf.zeros([5]), name="B3")
score = tf.nn.bias_add(tf.matmul(layer2,W3), B3)

#score = tf.matmul(layer2,W3)
probability = tf.nn.sigmoid(score)

#From here we define the parts of the network needed for learning a good policy.
tvars = tf.trainable_variables()
print tvars
input_y = tf.placeholder(tf.float32,[None,5], name="input_y") # Prior output
advantages = tf.placeholder(tf.float32,name="reward_signal")

# The loss function. This sends the weights in the direction of making actions 
# that gave good advantage (reward over time) more likely, and actions that didn't less likely.
loss = -tf.reduce_mean((tf.log(input_y - probability)) * advantages)
newGrads = tf.gradients(loss,tvars)

# Once we have collected a series of gradients from multiple episodes, we apply them.
# We don't just apply gradeients after every episode in order to account for noise in the reward signal.
adam = tf.train.AdamOptimizer(learning_rate=learning_rate) # Our optimizer
W1Grad = tf.placeholder(tf.float32,name="w_grad1") # Placeholders to send the final gradients through when we update.
W2Grad = tf.placeholder(tf.float32,name="w_grad2")
W3Grad = tf.placeholder(tf.float32,name="w_grad3")
B1Grad = tf.placeholder(tf.float32,name="b_grad1") # Placeholders to send the final gradients through when we update.
B2Grad = tf.placeholder(tf.float32,name="b_grad2")
B3Grad = tf.placeholder(tf.float32,name="b_grad3")
batchGrad = [W1Grad,W2Grad,W3Grad,B1Grad,B2Grad,B3Grad]
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))


# ### Advantage function
# This function allows us to weigh the rewards our agent recieves. In the context of the Cart-Pole task, we want actions that kept the pole in the air a long time to have a large reward, and actions that contributed to the pole falling to have a decreased or negative reward. We do this by weighing the rewards from the end of the episode, with actions at the end being seen as negative, since they likely contributed to the pole falling, and the episode ending. Likewise, early actions are seen as more positive, since they weren't responsible for the pole falling.


# ### Running the Agent and Environment

# Here we run the neural network agent, and have it act in the CartPole environment.

# In[8]:

xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 100000
init = tf.initialize_all_variables()

input_data = list(csv.reader(open("data/market.csv")))
input_index = 1
equity = 1.0  # 1 unit of money, to start

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # TODO no env
    input_index = 1 # Ignore csv header
    equity = 1.0  # 1 unit of money, to start
    observation = input_data[input_index][1:IN_DIMENS-OUT_DIMENS] # Ignore timestamp, don't want that in weights
    observation.append(0.2) # Equally balanced portfolio to start
    observation.append(0.2)
    observation.append(0.2)
    observation.append(0.2)
    observation.append(0.2)
    observation.append(1.0) # 1.0 equity to start
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
    
    commission_fees = 0

    while episode_number <= total_episodes:
        if input_index == 1:
            equity = 1.0

        # Make sure the observation is in a shape the network can handle.
        x = np.reshape(observation,[1,IN_DIMENS])
        
        # Run the policy network and get an action to take. Normalize output from Neural Net to sum to 1
        portfolio_raw = np.reshape(sess.run(probability,feed_dict={observations: x}), [5])
        portfolio = normalize(portfolio_raw)

        # Add random noise to portfolio for exploration
        portfolio = [p + (0.5 * exploration_rate * np.random.uniform() - exploration_rate) for p in portfolio_raw]

        # Normalize portfolio again after random noise
        portfolio = normalize(portfolio)

        ys.append(portfolio) # portfolio output
        xs.append(x) # observation

        portfolio_diff = 0.0
        for i in range(0, len(portfolio)):
            portfolio_diff += abs(portfolio[i] - last_portfolio[i])

        # print "Port Diff %s" % portfolio_diff
        if portfolio_diff > TRADE_THRESHOLD:
            # Portfolio changed somewhat significantly
            # So a trade fee is incurred, and the portfolio is updated
            equity -= TRADE_FEE
            commission_fees += 1
            last_portfolio = portfolio

        # Update statistics
        average_portfolio[0] += last_portfolio[0]
        average_portfolio[1] += last_portfolio[1]
        average_portfolio[2] += last_portfolio[2]
        average_portfolio[3] += last_portfolio[3]
        average_portfolio[4] += last_portfolio[4]
        min_portfolio[0] = min(last_portfolio[0], min_portfolio[0])
        min_portfolio[1] = min(last_portfolio[1], min_portfolio[1])
        min_portfolio[2] = min(last_portfolio[2], min_portfolio[2])
        min_portfolio[3] = min(last_portfolio[3], min_portfolio[3])
        min_portfolio[4] = min(last_portfolio[3], min_portfolio[4])
        max_portfolio[0] = max(last_portfolio[0], max_portfolio[0])
        max_portfolio[1] = max(last_portfolio[1], max_portfolio[1])
        max_portfolio[2] = max(last_portfolio[2], max_portfolio[2])
        max_portfolio[3] = max(last_portfolio[3], max_portfolio[3])
        max_portfolio[4] = max(last_portfolio[3], max_portfolio[4])

        # print
        # print "======================="
        # print "Time Step: %d" % input_index

        # Reward is this day's gains or losses compared to tomorrow, with some penalty for changes in portfolio
        input_index += 1
        observation = input_data[input_index][1:IN_DIMENS-OUT_DIMENS] # Ignore timestamp, don't want that in weights
        next_spy_change = float(observation[6])
        next_slv_change = float(observation[6+11])
        next_gld_change = float(observation[6+22])
        next_uso_change = float(observation[6+33])

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

        done = (input_index == TOTAL_STEPS) or (equity <= 0.01)

        # Final reward is portfolio's liquid value, plus 1.0 bonus for finishing
        # If didn't make it to end, final reward is % complete to end
        if done:
            if USE_DONE_REWARD != 0:
                if equity <= 0.01:
                    reward = -1 + -3 * (TOTAL_STEPS - input_index)/TOTAL_STEPS
                else:
                    reward = 5.0 * equity

            average_portfolio = [p / input_index for p in average_portfolio]

            print "Equity: %s at time step %d" % (equity, input_index)
            print "Commission fees: %d" % commission_fees
            print "Done Reward: %s" % reward
            print "Learning Rate: %s, Exploration Rate: %s" % (learning_rate, exploration_rate)
            print "Avg Portfolio: %s" % average_portfolio
            print "Min Portfolio: %s" % min_portfolio
            print "Max Portfolio: %s" % max_portfolio
            print "Iteration: %d" % episode_number
            print

            
        reward_sum += reward
        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

        if done: 
            episode_number += 1
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            tfp = tfps
            xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[] # reset array memory

            # compute the discounted reward backwards through time
            # print "EPR: %s" % epr
            discounted_epr = discount_rewards(epr, equity, commission_fees) # TODO perhaps figure this out better discount_rewards(epr)
            # print "DISCOUNTED EPR: %s" % discounted_epr
            # size the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            # print "MEAN EPR: %s" % discounted_epr
            discounted_epr /= np.std(discounted_epr)
            #print "STD EPR: %s" % discounted_epr

            # Reset to start
            commission_fees = 0
            equity = 1.0
            input_index = 1
            observation = input_data[input_index][1:IN_DIMENS-OUT_DIMENS] # Ignore timestamp, don't want that in weights
            observation.append(1.0) # Equally balanced portfolio to start
            observation.append(1.0)
            observation.append(1.0)
            observation.append(1.0)

            average_portfolio = [0, 0, 0, 0, 0]
            min_portfolio = [1, 1, 1, 1, 1]
            max_portfolio = [0, 0, 0, 0, 0]


            
            # Get the gradient for this episode, and save it in the gradBuffer
            tGrad = sess.run(newGrads,feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            for ix,grad in enumerate(tGrad):
                gradBuffer[ix] += grad
                
            # If we have completed enough episodes, then update the policy network with our gradients.
            if episode_number % batch_size == 0: 
                sess.run(updateGrads,feed_dict={W1Grad:gradBuffer[0], W2Grad:gradBuffer[1], W3Grad:gradBuffer[2], B1Grad:gradBuffer[3], B2Grad:gradBuffer[4], B3Grad:gradBuffer[5]})
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
                
                # Give a summary of how well our network is doing for each batch of episodes.
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print
                print 'Average reward for episode %f.  Total average reward %f.' % (reward_sum/batch_size, running_reward/batch_size)
                print
                reward_sum = 0
            
            # TODO no env
            input_index = 1 # Ignore csv header
            equity = 1.0  # 1 unit of money, to start
            observation = input_data[input_index][1:IN_DIMENS-OUT_DIMENS] # Ignore timestamp, don't want that in weights
            observation.append(0.2) # Equally balanced portfolio to start
            observation.append(0.2)
            observation.append(0.2)
            observation.append(0.2)
            observation.append(0.2)
            observation.append(1.0) # 1.0 equity to start
            observation = np.array(map(float, observation))

            if episode_number % 100 == 0:
                exploration_rate *= 0.95
                learning_rate *= 0.95
                learning_rate = max(5e-4, learning_rate)
                exploration_rate = max(0.03, exploration_rate)
                print
                print "Run %d episodes" % episode_number
                print "Explore rate : %s" % exploration_rate
                print
        
print episode_number,'Episodes completed.'

