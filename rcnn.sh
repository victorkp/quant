#!/bin/bash
cp rcnn-development.py runs/rcnn-$1.py
stdbuf -oL python rcnn-development.py > runs/rcnn-$1 &
jobs

# Kill descendents (rcnn.py) running in the background when
# This script exits
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

sleep 4
tail -f runs/rcnn-$1
