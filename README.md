## About
This repository contains a PaddlePaddle implementation of a GNN for classification on a smaller ogbn-arxiv dataset, code written by Tianjian Li.


## Training
in order to start training, simply type 
```
python train.py
```
then two .csv files will be generated. The two files
`submission.csv` is the unsmoothed output and `submission_cs.csv` is the smoothed output of the model.

- `graphmodel_1.py`contains the code for our graph neural network with residual connections.  
- `compareandsmooth.py`contains the code for the Compare & Smooth trick described in https://arxiv.org/abs/2010.13993.


## Papers Related
Combining Label Propagation and Simple Models Out-performs Graph Neural Networks
https://arxiv.org/abs/2010.13993

Semi-Supervised Classification with Graph Convolutional Networks
https://arxiv.org/abs/1609.02907

