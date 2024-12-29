#!/bin/bash
# This is a test script for the VF-PS example
for i in 1 2
do
  for dataset in "web" "phishing" "adult"
  do
    python3 mi_fagin_batch.py --dataset $dataset
  done
done