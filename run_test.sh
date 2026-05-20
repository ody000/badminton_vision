#!/bin/bash
# Load environment and run 2-player architecture test

module load python/3.11
module load pytorch/2.0
module load cuda/11.8

python3 test_2player_arch.py --dataset-dir data/input/train/player2
