#!/bin/bash
#for i in 1 2
#do
for dataset in "bank" "credit" "adult" "web" "phishing" "heart-disease"
do
  python3 /home/zxk/codes/vfps_mi_diversity/baselines/DIG_FL/script/DIG_FL_encrypt_batch.py --dataset $dataset
done
#done

