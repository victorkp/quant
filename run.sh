#!/bin/bash
cp rl.py rl-$.py
(python rl.py > rl-$1 &) && tail -f rl-$1
