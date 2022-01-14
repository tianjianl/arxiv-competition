## About
This repository contains a PaddlePaddle implementation of a GNN for classification on a smaller ogbn-arxiv dataset. 
  
  
If you want to see the results, you can find them in [this notebook in Chinese](resgcn.ipynb). or [this notebook in English](ResGCN_eng.ipynb)

## Training
in order to start training, simply type 
```
sh train.sh
```
then all of the requirements should be installed automatically.  
after the training loop, two .csv files will be generated. The two files
`submission.csv` is the unsmoothed output and `submission_cs.csv` is the smoothed output of the model.

- `graphmodel_1.py`contains the code for our graph neural network with residual connections.  
- `correctandsmooth.py`contains the code for the Compare & Smooth trick described in [this paper](https://arxiv.org/abs/2010.13993)
- `unimpmodel.py` contains the code for the Baidu UniMP model described in [this paper](https://arxiv.org/pdf/2009.03509)

## Papers Related
Combining Label Propagation and Simple Models Out-performs Graph Neural Networks
https://arxiv.org/abs/2010.13993

Semi-Supervised Classification with Graph Convolutional Networks
https://arxiv.org/abs/1609.02907

Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification
https://arxiv.org/abs/2009.03509
