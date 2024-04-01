# SIMPLE
Status: we are still organizing the code for readability currently :) 
## Requirements
- python >= 3.6.13
- pytorch >= 1.8.1
- pandas >= 1.1.5
- numpy >= 1.19.5
- dgl >= 0.6.1
- pyyaml >= 5.4.1
- tqdm >= 4.61.0
- pybind11 >= 2.6.2
- g++ >= 7.5.0
- openmp >= 201511
- numba 0.54.1

Compile C++ temporal sampler (from TGL) first with the following command
> python SIMPLE/setup.py build_ext --inplace

## Datasets
We use four datasets in the paper: LASTFM, WIKITALK, STACKOVERFLOW and GDELT.
For LASTFM and GDELT, they can be downloaded from AWS S3 bucket using the `down.sh` script. 
For WIKITALK and STACKOVERFLOW, they can be downloaded from http://snap.stanford.edu/data/wiki-talk-temporal.html and https://snap.stanford.edu/data/sx-stackoverflow.html respectively.
Note that for WIKITALK and STACKOVERFLOW, they need to be preprocessed after obtaining the raw data from the links above. For example:
> python preprocess.py --data \<NameOfDataset> --txt \<PathOfRawData>

## Usage
To generate buffer plans by SIMPLE, run:
> python SIMPLE/buffer_plan_preprocessing.py --data \<NameOfDataset> --config \<TrainingConfiguration> --dim_edge_feat \<EdgeFeatDimension> --dim_node_feat \<NodeFeatDimension> --mem_dim \<MemoryDimension> --threshold \<UserDefinedBudget>

Exemplar training:
> python main.py --data WIKITALK --config config/TGN_WIKITALK.yml --gpu 0 --threshold 0.1
