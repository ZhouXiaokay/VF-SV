# VF-SV
## Introduction


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
 Homomorphic encryption: transmission/tenseal_shapley/generate_ctx.py



## Baselines
How to run the workflow?  First，run the server, then run the client.  
**Server:** `python /transmission/tenseal_shapley/tenseal_shapley_server.py`  
**Client:**
* **All Reduce**:
  2. `python /tenseal_script/mi_shapley/mi_all_reduce_shapley.py`  
* **VF-PS**:
  2. `python /tenseal_script/VF-PS/mi_fagin_batch.py`

## Ours
**VF-SV**:
  1. `python /script/mi_shapley`