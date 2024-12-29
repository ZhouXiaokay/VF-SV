# VF-SV
## Introduction


## Parameters
 The parameters are defined in the `conf/args.py` file.

## Baselines
How to run the workflow?
* **All Reduce**:
  1. `python /transmission/tenseal_shapley/tenseal_shapley_server.py`  
  2. `python /tenseal_script/mi_shapley/mi_all_reduce_shapley.py`  
* **VF-PS**:
  1. `python /transmission/tenseal_shapley/tenseal_shapley_server.py`
  2. `python /tenseal_script/VF-PS/mi_fagin_batch.py`