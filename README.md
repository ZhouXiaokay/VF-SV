# PS-MI
## Introduction
This is a repository for the paper "PS-MI: Accurate, Efficient, and Private Data Valuation in Vertical Federated Learning".

## Requirements
` grpcio==1.34.1`  
`grpcio-tools==1.34.1`  
`tenseal==0.4.0`  
`numpy==1.19.5`  
`pandas==1.1.5`  
`scikit-learn==0.24.1`  
`torch==1.8.1`



## Parameters
 The parameters are defined in the `conf/args.py` file.



## How to Run
1. Data Preparation:
  * Download the dataset from: UCI, Kaggle website
  * Put the dataset in the `data` folder.
  * The tools for data preprocessing are in the `data_loader` folder.
2. Run the scripts:
  * script/mi)shapley/PS-Mi.sh: Run the PS-Mi algorithm.