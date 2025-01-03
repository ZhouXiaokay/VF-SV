#!/bin/bash
for i in 1 2
do
for dataset in "bank" "credit" "adult" "web" "phishing" "heart-disease"
do
  python3 /home/zxk/codes/vfps_mi_diversity/baselines/VFDV_IM/script/mlp_shapley_reweight.py --dataset $dataset
done
done
for i in 1 2
do
for dataset in "bank" "credit" "adult" "web" "phishing" "heart-disease"
do
  python3 /home/zxk/codes/vfps_mi_diversity/script/mi_shapley/mi_RP_lsh_all_reduce_shapley.py --dataset $dataset
done
done