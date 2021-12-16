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
# 
# 
# 
# 

# ## 运行方式
# 本次基线基于飞桨PaddlePaddle 2.2.0版本，若本地运行则可能需要额外安装pgl、easydict、pandas等模块。
# 
# ## 本地运行
# 下载左侧文件夹中的所有py文件（包括build_model.py, model.py）,以及work目录，然后在右上角“文件”->“导出Notebook到py”，这样可以保证代码是最新版本），执行导出的py文件即可。完成后下载submission.csv提交结果即可。
# 
# ## AI Studio (Notebook)运行
# 依次运行下方的cell，完成后下载submission.csv提交结果即可。若运行时修改了cell，推荐在右上角重启执行器后再以此运行，避免因内存未清空而产生报错。 Tips：若修改了左侧文件夹中数据，也需要重启执行器后才会加载新文件。

# ## 代码整体逻辑
# 
# 1. 读取提供的数据集，包含构图以及读取节点特征（用户可自己改动边的构造方式）
# 
# 2. 配置化生成模型，用户也可以根据教程进行图神经网络的实现。
# 
# 3. 开始训练
# 
# 4. 执行预测并产生结果文件
# 

# In[1]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install pgl easydict -q -t /home/aistudio/external-libraries')


# In[2]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')


# In[3]:


import pgl
import paddle
import paddle.fluid as fluid
import paddle.nn as nn
import numpy as np
import time
import pandas as pd


# In[4]:


from easydict import EasyDict as edict

config = {
    "model_name": "GraphSAGE",
    "num_class": 35,
    "num_layers": 2,
    "dropout": 0.5,
    "learning_rate": 0.01,
    "weight_decay": 0.0005,
    "edge_dropout": 0.00
}

config = edict(config)


# ## 数据加载模块
# 
# 这里主要是用于读取数据集，包括读取图数据构图，以及训练集的划分。

# In[5]:


from collections import namedtuple

Dataset = namedtuple("Dataset", 
               ["graph", "num_classes", "train_index",
                "train_label", "valid_index", "valid_label", "test_index", "node_feat", "edges", "node_label"])

def load_edges(num_nodes, self_loop=True, add_inverse_edge=True):
    # 从数据中读取边
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
    # 从数据中读取点特征和边，以及数据划分
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
                    test_index=test_index, num_classes=35, node_feat = node_feat, edges = edges, node_label=node_label)
    return dataset


# In[6]:


dataset = load()

train_index = dataset.train_index
train_label = paddle.to_tensor(np.reshape(dataset.train_label, [-1 , 1]))
train_index = paddle.to_tensor(np.expand_dims(train_index, -1))

val_index = dataset.valid_index
val_label = paddle.to_tensor(np.reshape(dataset.valid_label, [-1, 1]))
val_index = paddle.to_tensor(np.expand_dims(val_index, -1))

test_index = dataset.test_index
test_index = paddle.to_tensor(np.expand_dims(test_index, -1))
test_label = paddle.to_tensor(np.zeros((len(test_index), 1), dtype="int64"))
num_class = dataset.num_classes


# ## 组网模块
# 
# 这里是组网模块，目前已经提供了一些预定义的模型，包括**GCN**, **GAT**, **APPNP**等。可以通过简单的配置，设定模型的层数，hidden_size等。你也可以深入到model.py里面，去奇思妙想，写自己的图神经网络。

# In[7]:


import pgl
from pgl.sampling import subgraph
from pgl.graph import Graph
import graphmodel_1
from graphmodel_1 import Model
import paddle
import paddle.nn as nn
import numpy as np
import time
 #使用CPU
#place = fluid.CPUPlace()
# 使用GPU
place = fluid.CUDAPlace(0)
model = Model(config)
lr = 0.01
#lr = paddle.optimizer.lr.ExponentialDecay(learning_rate=config.get("learning_rate", 0.01), gamma=0.9, verbose=True)
optim = paddle.optimizer.Adam(learning_rate = lr, parameters = model.parameters())


# ## 开始训练过程
# 
# 图神经网络采用FullBatch的训练方式，每一步训练就会把所有整张图训练样本全部训练一遍。
# 
# 

# In[8]:


epoch = 200
# 将图数据变成 feed_dict 用于传入Paddle Excecutor
criterion = paddle.nn.loss.CrossEntropyLoss()


edges = dataset.edges
graph = dataset.graph
graph.tensor()
for epoch in range(epoch):
    # Full Batch 训练
    # 设定图上面那些节点要获取
    # node_index: 训练节点的nid    
    # node_label: 训练节点对应的标签
    # input shape = [N, Cin]
    # output shape [N, Co]

    #pgl.sampling.subgraph(graph, nodes, eid=None, edges=None, with_node_feat=True, with_edge_feat=True)
    #g = subgraph(graph=graph, nodes=train_index, edges=edges)
    #g.tensor()
    
    pred = model(graph, graph.node_feat["feat"])
    print(pred.shape)
    pred = paddle.gather(pred, train_index)
    loss = criterion(pred, train_label)
    loss.backward()
    acc = paddle.metric.accuracy(input=pred, label=train_label, k=1)
    optim.step()
    optim.clear_grad()
    
    # Full Batch 验证
    # 设定图上面那些节点要获取
    # node_index: 训练节点的nid    
    # node_label: 训练节点对应的标签
    #g = subgraph(graph=graph, nodes=val_index, edges=edges)
    #g.tensor()
    val_pred = model(graph, graph.node_feat["feat"])
    val_pred = paddle.gather(val_pred, val_index)
    val_acc = paddle.metric.accuracy(input=val_pred, label=val_label, k=1)
    print("Epoch", epoch, "Train Acc", acc, "Valid Acc", val_acc)



# ## 对测试集进行预测
# 
# 训练完成后，我们对测试集进行预测。预测的时候，由于不知道测试集合的标签，我们随意给一些测试label。最终我们获得测试数据的预测结果。
# 

# ## 保存模型参数准备Correct and smooth
# 这里我们调用paddle提供的接口save来保存模型参数为model_state_dict，然后生成预测label。

# In[9]:


test_pred = model(graph, graph.node_feat["feat"])
test_pred = paddle.gather(test_pred, test_index)
test_pred = paddle.argmax(test_pred, axis=1)
test_index = np.array(test_index)
test_pred = np.array(test_pred)
submission = pd.DataFrame(data={
                            "nid": test_index.reshape(-1),
                            "label": test_pred.reshape(-1)
                        })
submission.to_csv("submission.csv", index=False)

#saving the state dict for smoothing final results 
paddle.save(model.state_dict(), "model_state_dict")




# ## Correct and Smooth部分
# 如果我们使用MLP，就需要调用Correct部分，但我们使用了GAT就需要调用Smooth部分。

# In[10]:


from correctandsmooth import LayerPropagation, CorrectAndSmooth

model_state_dict = paddle.load('model_state_dict')
model.load_dict(model_state_dict)
y_pred = model(graph, graph.node_feat['feat']) 

y_soft = nn.functional.softmax(y_pred)

cas = CorrectAndSmooth(50, 0.979, 'DAD', 50, 0.756, 'DAD', 20.)

mask_idx = paddle.concat([train_index, val_index])
node_label = paddle.to_tensor(np.reshape(dataset.node_label, [-1 , 1]))

mask_label = paddle.gather(node_label, mask_idx)
mask_label = paddle.nn.functional.one_hot(mask_label, num_classes=35)
y_soft = cas.smooth(graph, y_soft, mask_label, mask_idx)



# ## 生成提交文件
# 
# 最后一步，我们可以使用pandas轻松生成提交文件，最后下载 submission.csv 提交就好了。

# In[11]:


pred = paddle.argmax(y_soft, axis=-1, keepdim=True)
test_index = paddle.to_tensor(test_index)
pred = paddle.gather(pred, test_index)
test_index = np.array(test_index)
pred = np.array(pred)

test_index = np.array(test_index)
pred = np.array(pred)
submission = pd.DataFrame(data={
                            "nid": test_index.reshape(-1),
                            "label": pred.reshape(-1)
                        })
submission.to_csv("submission_cs.csv", index=False)


# ### 2. One More Thing
# 
# 如果大家还想要别的奇思妙想，可以参考以下论文，他们都在节点分类上有很大提升。
# 
# * Predict then Propagate: Graph Neural Networks meet Personalized PageRank (https://arxiv.org/abs/1810.05997)
# 
# * Simple and Deep Graph Convolutional Networks (https://arxiv.org/abs/2007.02133)
# 
# * Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification (https://arxiv.org/abs/2009.03509)
# 
# * Combining Label Propagation and Simple Models Out-performs Graph Neural Networks (https://arxiv.org/abs/2010.13993)
# 
# 
# 大家也可以看看github的 [UniMP算法](https://github.com/PaddlePaddle/PGL/tree/main/ogb_examples/nodeproppred/unimp) 这个例子，里面有相似的数据集，并且最近也是SOTA效果，有帮助👏欢迎点Star
