#!/bin/bash
# Make backujp of py for this run
cp rl-development.py rl-$1.py

python rl-development.py > rl-$1 &
jobs

# Kill descendents (rcnn.py) running in the background when
# This script exits
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

sleep 4
tail -f rl-$1

