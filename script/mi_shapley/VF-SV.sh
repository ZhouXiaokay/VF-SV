##!/bin/bash

for seed in 40 41 42
do
for dataset in "web" "phishing" "heart-disease"
do
  python3 /home/zxk/codes/vfps_mi_diversity/script/mi_shapley/mi_all_reduce_shapley.py --dataset $dataset --seed $seed
  python3 /home/zxk/codes/vfps_mi_diversity/script/mi_shapley/mi_RP_lsh_adaptive_sampling_fagin_batch_shapley.py --dataset $dataset --seed $seed
  python3 /home/zxk/codes/vfps_mi_diversity/baselines/VF_PS/script/VF-PS.py --dataset $dataset --seed $seed
done
done