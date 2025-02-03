##!/bin/bash
#   python3 /home/zxk/codes/vfps_mi_diversity/script/mi_shapley/mi_all_reduce_shapley.py --dataset $dataset --seed $seed
  python3 /home/zxk/codes/vfps_mi_diversity/baselines/VF_PS/script/VF-PS.py --dataset $dataset --seed $seed

for dataset in "adult" "web" "heart-disease"
do
for proj_size in 5 10 15 20 25
do
  python3 /home/zxk/codes/vfps_mi_diversity/script/mi_shapley/mi_RP_lsh_adaptive_sampling_fagin_batch_shapley.py --dataset $dataset --proj_size $proj_size
done
done