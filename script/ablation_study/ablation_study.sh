#!/bin/bash
# This is a script for the ablation study
#python3 /home/zxk/codes/vfps_mi_diversity/script/mi_shapley/mi_RP_lsh_fagin_batch_shapley.py --dataset $dataset
#python3 /home/zxk/codes/vfps_mi_diversity/script/mi_shapley/mi_RP_lsh_all_reduce_shapley.py --dataset $dataset
for i in 1 2
do
for dataset in "bank" "credit" "adult" "web" "phishing" "heart-disease"
do
  python3 /home/zxk/codes/vfps_mi_diversity/script/mi_shapley/mi_RP_lsh_all_reduce_shapley.py --dataset $dataset
  python3 /home/zxk/codes/vfps_mi_diversity/script/mi_shapley/mi_RP_all_reduce_shapley.py --dataset $dataset
done
done