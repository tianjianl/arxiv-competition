import pgl
import pgl.nn as pnn
import paddle
import paddle.nn as nn 
import numpy as np
import time

class Model(paddle.nn.Layer): 
    def __init__(self, config):
        super(Model, self).__init__() 
        self.hidden_size = config.get("hidden_size", 128)
        self.num_layers = config.get("num_layers", 1)
        self.dropout = config.get("dropout", 0.5)
        self.edge_dropout = config.get("edge_dropout", 0.0)
        
        self.model_name = config.get("model_name", 'GCN')
        if self.model_name == 'GCN':
            self.graphmodel = pnn.GCNConv(100, self.hidden_size)
            self.graphmodel_1 = pnn.GCNConv(self.hidden_size, self.hidden_size)
            self.graphmodel_2 = pnn.GCNConv(self.hidden_size, self.hidden_size)
        elif self.model_name == 'GAT':
            self.graphmodel = pnn.GATConv(100, int(self.hidden_size/3), feat_drop=self.dropout, num_heads = 3)
            self.graphmodel_1 = pnn.GATConv(self.hidden_size, int(self.hidden_size/3), feat_drop=self.dropout, num_heads = 3)
            self.graphmodel_2 = pnn.GATConv(self.hidden_size, int(self.hidden_size/3), feat_drop=self.dropout, num_heads = 3)
        elif self.model_name == 'GraphSAGE':
            self.graphmodel = pnn.GraphSageConv(100, self.hidden_size, aggr_func="max")
            self.graphmodel_1 = pnn.GraphSageConv(self.hidden_size, self.hidden_size, aggr_func="max")
            
        self.resfc_1 = nn.Linear(100, 
                        self.hidden_size, 
                        weight_attr = paddle.ParamAttr(name = 'res_w_1'), 
                        bias_attr = paddle.ParamAttr(name = 'res_b_1'))
        
        self.resfc_2 = nn.Linear(self.hidden_size,
                        self.hidden_size, 
                        weight_attr = paddle.ParamAttr(name = 'res_w_2'),
                        bias_attr = paddle.ParamAttr(name = 'res_b_2'))

        self.resfc_3 = nn.Linear(self.hidden_size,
                        self.hidden_size, 
                        weight_attr = paddle.ParamAttr(name = 'res_w_3'),
                        bias_attr = paddle.ParamAttr(name = 'res_b_3'))
        self.fc = nn.Linear(self.hidden_size, 35, weight_attr = paddle.ParamAttr(name = 'output_w_0'))

    def forward(self, graph, x_input):

        #shape of x is [Num, Channels]
        x = x_input
        #shape of label is [Num, ]
        
        for i in range(self.num_layers):
            if i == 0:
                x_res = x
                x = self.graphmodel(graph, x)
                x_res = self.resfc_1(x_res)
                x = x + x_res
                x = nn.functional.relu(x)
            elif i == 1:
                x_res = x
                x = self.graphmodel_1(graph, x)
                x_res = self.resfc_2(x_res)
                x = x + x_res
                x = nn.functional.relu(x)
            else:
                x_res = x
                x = self.graphmodel_2(graph, x)
                x_res = self.resfc_3(x_res)
                x = x + x_res
                x = nn.functional.relu(x)

        #now x is [num, hidden_size]

        x = self.fc(x)
        #[num, 35]
        return x


