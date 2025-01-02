#!/bin/bash
# This is a test script for the VF-SV example
#python3 /home/zxk/codes/vfps_mi_diversity/transmission/tenseal_shapley/tenseal_shapley_server.py
#python3 /home/zxk/codes/vfps_mi_diversity/transmission/tenseal_shapley/tenseal_shapley_server.py
#for i in 1 2
#do
for dataset in "web" "credit" "bank" "phishing"
do
  python3 mi_fagin_batch.py --dataset $dataset
done
#done