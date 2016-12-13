#!/bin/bash
cp rl.py rl-$1.py
(python rl.py > rl-$1 &) && sleep 4 && tail -f rl-$1
