### HallmarkGraph: a cancer hallmark informed graph neural network for classifying hierarchical tumor subtypes
We present a graph neural network, HallmarkGraph, the first biologically informed model developed to classify hierarchical tumor subtypes in human cancer. Inspired by cancer hallmarks, the model’s architecture integrates transcriptome profiles and gene regulatory interactions to perform multi-label classification. We evaluate the model on a comprehensive pan-cancer cohort comprising 11,476 samples from 26 primary cancers with 405 subtypes. 

## The current version is to provide reviewers with reproducible experimental results.
The repository contains the following strucutre and files:
```bash
main/
  └──code/
        └── HallmarkGraph.py
  └──data/
        ├── clean_data.csv
        └── clean_label.xlsx
  └──adjacency_matrix/
        ├── Undirected_0...matrix.npz
        ├── ...
        └── Undirected_9...matrix.npz
  └──best_model/
        ├── BioGCN_net_(0.4)_target_1.h5
        ├── ...
        └── BioGCN_net_(0.4)_target_8.h5
```

## Pre-requisites: 
* Linux (Tested on Ubuntu 18.04) 
* NVIDIA GPU (Tested on Nvidia GeForce RTX 2080 Ti x 16) with CUDA 11.0 and cuDNN 7.5
* Python (3.8), tensorflow (2.8.2), keras (2.8.0), shap (0.45.1), scikit-learn (1.4.1), matplotlib (3.9.2)    

## How to eproduce the results of the paper:

1. You first need to download the data and trained models and store them in `data` and `best-model` folders, respectively (see readme.md in the folder).
2. Run the file `code/HallmarkGraph.py`
3. If you want to predict hard samples, please set `Whether_to_predict_hard_stamples = TRUE` in `code/HallmarkGraph.py`.
4. If you want to calculate the shap, please set `Whether_to_calculate_the_shap = TRUE` in `code/HallmarkGraph.py`.
