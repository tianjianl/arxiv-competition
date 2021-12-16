## About
This repository contains a PaddlePaddle implementation of a GNN for classification on a smaller ogbn-arxiv dataset, code written by Tianjian Li.


## Training
in order to start training, simply type 
```
python train.py
```
then two .csv files will be generated. The two files

```
submission.csv submission_cs.csv
```
is the unsmoothed output and smoothed output of the model.

## Papers Related
Combining Label Propagation and Simple Models Out-performs Graph Neural Networks
https://arxiv.org/abs/2010.13993

Semi-Supervised Classification with Graph Convolutional Networks
https://arxiv.org/abs/1609.02907

