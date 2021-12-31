import pgl
import pgl.nn as pnn
import paddle
import paddle.nn as nn 
import numpy as np
import time

class UniMP(paddle.nn.Layer): 
    def __init__(self, config):
        super(UniMP, self).__init__() 
        self.hidden_size = config.get("hidden_size", 128)
        self.num_layers = config.get("num_layers", 1)
        self.feat_drop = config.get("dropout", 0.4)
        self.attn_drop = config.get("attndrop", 0.4)
        self.num_classes = config.get("num_classes", 35)
        self.num_heads = config.get("num_heads", 1)
        self.layers = nn.LayerList()
        
        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(
                    pnn.TransformerConv(100, 
                                        int(self.hidden_size/self.num_heads), 
                                        num_heads=self.num_heads, 
                                        feat_drop=self.feat_drop, 
                                        attn_drop=self.attn_drop, 
                                        gate=True))    
            else:
                self.layers.append(
                    pnn.TransformerConv(self.hidden_size, 
                                        int(self.hidden_size/self.num_heads), 
                                        num_heads=self.num_heads, 
                                        feat_drop=self.feat_drop, 
                                        attn_drop=self.attn_drop, 
                                        gate=True))
    
        self.fc = nn.Linear(self.hidden_size, self.num_classes, weight_attr=paddle.ParamAttr(name='output_w_0'))
    def forward(self, graph, x_input):

        #shape of x is [Num, Channels]
        x = x_input

        for layer in self.layers:
            x = layer(graph, x)
            
        #now x is [num, hidden_size]
        x = self.fc(x)
        #[num, 35]
        return x


