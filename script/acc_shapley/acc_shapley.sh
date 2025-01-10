##!/bin/bash

for seed in 40 41 42 43 44
do
for dataset in "bank" "credit" "adult" "web" "phishing" "heart-disease"
do
  python3 /home/zxk/codes/vfps_mi_diversity/script/acc_shapley/shapley_lr.py --dataset $dataset --seed $seed
done
done