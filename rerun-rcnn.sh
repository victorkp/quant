#!/bin/bash
(python runs/rcnn-$1.py > runs/rcnn-$1 &) && sleep 20 && tail -f runs/rcnn-$1
