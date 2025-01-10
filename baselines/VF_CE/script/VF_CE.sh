for dataset in "bank" "credit" "adult" "web" "phishing" "heart-disease"
do
  python3 /home/zxk/codes/vfps_mi_diversity/baselines/VF_CE/script/VF_CE.py --dataset $dataset
done