# Active Portfolio Management and Composition#

## Directory Overview ##
   + runs/  - directory of experiment output and the python files that produced it
   + data/
     + data-in/   - contains the raw data gathered from Yahoo Finance and the Bureau of Economic Statistics
     + market.csv - used as input into RCNN and traditional neural network
     + Makefile   - use `make q-table-` or `make market` to build q-out.csv or market.csv
     + build-market-input.pl - called by `make market` to produce market.csv from data-in/* files
     + build-q-table.pl      - called by `make q-table` to produce q-out.csv from data-in/* files

   + rl-paper.py   - the traditional deep reinforcement learning algorithm used for paper results
   + rcnn-paper.py - the RCNN algorithm used for paper results

   + rl-sample-convergence-output   -  sample output run of rl-paper.py that converged to 100% SPY
   + rcnn-sample-convergence-output -  sample output run of rcnn-paper.py that converges to ~1.88x equity

   + rl-development.py   - a later version of the deep RL algorithm (with worse results)
   + rcnn-development.py - a later version of the RCNN algorithm (with worse results, longer convergence)

   + rl.sh   - convenience script for backing up rl-development.sh, runs rl-development.sh, and sends output to a file in runs/
               example usage: `./rl.sh test1` backs up to runs/rl-test1.py and outputs to runs/rl-test1
   + rcnn.sh - convenience script for backing up rcnn-development.sh, runs rcnn-development.sh, and sends output to a file in runs/
               example usage: `./rcnn.sh test1` backs up to runs/rcnn-test1.py and outputs to runs/rcnn-test1


## Reproducing Results ##

### Reinforcement Learning Neural Network ###
Simply run `python rl-paper.py`. Convergence to holding 100% SPY should occur within 100 iterations.
You may notice a `NaN` error that occurs once training makes no progress - this is normal.

When ending equity is consistently returning ~1.73, you know that this neural net
has converged to holding 100% SPY. If the neural net does not converge to SPY 
within 100 iterations, it may be stuck in another local minima due to random
inialization. Typically, getting to 100% SPY convergence happens one in five runs
or so. Fortunately, getting to 100 iterations should only take a few minutes
on a decent desktop.

### RCNN Training ###
Results were reported from the output of `python rcnn-paper.py`.
As mentioned in the paper, convergence of the RCNN is finicky at best.
It is wise to run about a dozen instances of `python rcnn-paper.py` to get
a good idea of where the algorithm converges to (thanks to random inialization
and random exploration). 

Convergence, or near-convergence for results may happen as late as 
2000 iterations. On a 4GHz i7 920, a dozen instances will reach
2000 iterations after roughly 8-10 hours. Instances that are still
performing poorly after a few hundred iterations can generally be
killed early, as they tend to not make as much progress in the long
run as those that start performing somewhat well in the first 300 iterations.
