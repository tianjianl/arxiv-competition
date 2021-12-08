#!/usr/bin/env python
# coding: utf-8

# # 论文引用网络节点分类比赛
# 
# ## 赛题介绍
# 
# 
# 图神经网络（Graph Neural Network）是一种专门处理图结构数据的神经网络，目前被广泛应用于推荐系统、金融风控、生物计算中。图神经网络的经典问题主要有三种，包括节点分类、连接预测和图分类三种。本次比赛是图神经网络7日打卡课程的大作业，主要让同学们熟悉如何图神经网络处理节点分类问题。
# 
# 数据集为论文引用网络，图由大量的学术论文组成，节点之间的边是论文的引用关系，每一个节点提供简单的词向量组合的节点特征。我们的目的是给每篇论文推断出它的论文类别。


get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install pgl easydict -q -t /home/aistudio/external-libraries')

import sys 
sys.path.append('/home/aistudio/external-libraries')
import pgl
import paddle.fluid as fluid
import paddle.nn as nn
import numpy as np
import time
import pandas as pd



from easydict import EasyDict as edict

config = {
    "model_name": "GAT",
    "num_class": 35,
    "num_layers": 2,
    "dropout": 0.5,
    "learning_rate": 0.01,
    "weight_decay": 0.0005,
    "edge_dropout": 0.00
}

config = edict(config)

from collections import namedtuple

Dataset = namedtuple("Dataset", 
               ["graph", "num_classes", "train_index",
                "train_label", "valid_index", "valid_label", "test_index", "node_feat", "edges"])

def load_edges(num_nodes, self_loop=True, add_inverse_edge=True):

    edges = pd.read_csv("work/edges.csv", header=None, names=["src", "dst"]).values

    if add_inverse_edge:
        edges = np.vstack([edges, edges[:, ::-1]])

    if self_loop:
        src = np.arange(0, num_nodes)
        dst = np.arange(0, num_nodes)
        self_loop = np.vstack([src, dst]).T
        edges = np.vstack([edges, self_loop])
    
    return edges

def load():

    node_feat = np.load("work/feat.npy")
    num_nodes = node_feat.shape[0]
    edges = load_edges(num_nodes=num_nodes, self_loop=True, add_inverse_edge=True)
    graph = pgl.graph.Graph(num_nodes=num_nodes, edges=edges, node_feat={"feat": node_feat})
    
    df = pd.read_csv("work/train.csv")
    node_index = df["nid"].values
    node_label = df["label"].values
    train_part = int(len(node_index) * 0.8)
    train_index = node_index[:train_part]
    train_label = node_label[:train_part]
    valid_index = node_index[train_part:]
    valid_label = node_label[train_part:]
    test_index = pd.read_csv("work/test.csv")["nid"].values
    dataset = Dataset(graph=graph, 
                    train_label=train_label,
                    train_index=train_index,
                    valid_index=valid_index,
                    valid_label=valid_label,
                    test_index=test_index, num_classes=35, node_feat = node_feat, edges = edges)
    return dataset


dataset = load()

train_index = dataset.train_index
train_label = np.reshape(dataset.train_label, [-1 , 1])
train_index = np.expand_dims(train_index, -1)

val_index = dataset.valid_index
val_label = np.reshape(dataset.valid_label, [-1, 1])
val_index = np.expand_dims(val_index, -1)

test_index = dataset.test_index
test_index = np.expand_dims(test_index, -1)
test_label = np.zeros((len(test_index), 1), dtype="int64")
num_class = dataset.num_classes

import pgl
from pgl.sampling import subgraph
from pgl.graph import Graph
import graphmodel_1
from graphmodel_1 import Model
import paddle
import paddle.nn as nn
import numpy as np
import time
#Using CPU
#place = fluid.CPUPlace()
#Using GPU
place = fluid.CUDAPlace(0)
model = Model(config)
lr = 0.01
##lr = paddle.optimizer.lr.ExponentialDecay(learning_rate=config.get("learning_rate", 0.01), gamma=0.9, verbose=True)
optim = paddle.optimizer.Adam(learning_rate = lr, parameters = model.parameters())

epoch = 1000

node_feat = dataset.node_feat
edges = dataset.edges
graph = dataset.graph

for epoch in range(epoch):
    # Full Batch 训练
    # 设定图上面那些节点要获取
    # node_index: 训练节点的nid    
    # node_label: 训练节点对应的标签
    # input shape = [N, Cin]
    # output shape [N, Co]
    
    node_feat = paddle.to_tensor(node_feat)
    train_index = paddle.to_tensor(train_index)
    train_label = paddle.to_tensor(train_label)
    x = paddle.gather(node_feat, train_index)
    
    print(x.shape)
    ##pgl.sampling.subgraph(graph, nodes, eid=None, edges=None, with_node_feat=True, with_edge_feat=True)
    g = subgraph(graph=graph, nodes=train_index, edges=edges)
    g.tensor()

    loss, pred, acc = model(g, x, train_label)
    
    loss.backward()
    optim.step()
    optim.clear_grad()
    
    # Full Batch 验证
    # 设定图上面那些节点要获取
    # node_index: 训练节点的nid    
    # node_label: 训练节点对应的标签
    
    val_index = paddle.to_tensor(val_index)
    val_label = paddle.to_tensor(val_label)
    x_val = paddle.gather(node_feat, val_index)
    g = subgraph(graph=graph, nodes=val_index, edges=edges)
    g.tensor()
    val_loss, val_pred, val_acc = model(g, x_val, val_label)
    print("Epoch", epoch, "Train Acc", acc, "Valid Acc", val_acc)
    



node_feat = paddle.to_tensor(node_feat)
test_index = paddle.to_tensor(test_index)
test_label = paddle.to_tensor(test_label)
x_test = paddle.gather(node_feat, test_index)
    
##pgl.sampling.subgraph(graph, nodes, eid=None, edges=None, with_node_feat=True, with_edge_feat=True)
g = subgraph(graph=graph, nodes=test_index, edges=edges)
g.tensor()
_, pred, _ = model(g, x_test, test_label)


test_index = np.array(test_index)
pred = np.array(pred)
submission = pd.DataFrame(data={
                            "nid": test_index.reshape(-1),
                            "label": pred.reshape(-1)
                        })
submission.to_csv("submission.csv", index=False)

