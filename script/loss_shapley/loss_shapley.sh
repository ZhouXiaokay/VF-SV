##!/bin/bash

for seed in 40 41
  python3 /home/zxk/codes/vfps_mi_diversity/script/loss_shapley/shapley_lr.py --dataset --seed $seed
done