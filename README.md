# VF-SV
## Introduction


## Parameters
 The parameters are defined in the `conf/args.py` file.

## Baselines
How to run the workflow?  First，run the server, then run the client.  
**Server:** `python /transmission/tenseal_shapley/tenseal_shapley_server.py`  
**Client:**
* **All Reduce**:
  2. `python /tenseal_script/mi_shapley/mi_all_reduce_shapley.py`  
* **VF-PS**:
  2. `python /tenseal_script/VF-PS/mi_fagin_batch.py`