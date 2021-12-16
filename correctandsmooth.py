import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class LayerPropagation(nn.Layer):
    def __init__(self, num_layers, alpha, adj='DAD'):
        super(LayerPropagation, self).__init__()
        self.num_layers = num_layers
        self.alpha = alpha
        self.adj = adj

    @paddle.no_grad()
    def forward(self, graph, labels, mask=None, post_step=lambda y: paddle.clip(y, min=0, max=1)):

        def send_func(src_feat, dst_feat, edge_feat):
            return {'m':src_feat['h']}

        def recv_func(msg):
            return msg.reduce_sum(msg['m'])

        y = labels
        if mask is not None:
            y = paddle.zeros_like(labels)
            y[mask] = labels[mask]

        last = (1 - self.alpha) * y
        degree = graph.indegree()
        norm = paddle.cast(degree, dtype=paddle.float32)
        y = paddle.cast(y, dtype = paddle.float32)
        norm = paddle.clip(norm, min=1.0)
        norm = paddle.pow(norm, -0.5 if self.adj == 'DAD' else -1)
        norm = paddle.reshape(norm, [-1, 1])

        for i in range(self.num_layers):
            if i % 10 == 0:
                print("now at layer:",i)
            if self.adj in ['AD', 'DAD']:
                y = norm * y
                
            graph.node_feat['h'] = y
            msg = graph.send(send_func, src_feat={'h' : y})
            graph.recv(recv_func, msg)

            y = self.alpha * graph.node_feat.pop("h")
            if self.adj in ['DAD', 'DA']:
                y = y * norm

            y = post_step(last + y)

        return y

class CorrectAndSmooth(nn.Layer):
    def __init__(self, 
                num_correction_layers,
                correction_alpha,
                correction_adj,
                num_smoothing_layers,
                smoothing_alpha,
                smoothing_adj,
                autoscale=True,
                scale=1.):
        super(CorrectAndSmooth, self).__init__()     
        self.autoscale = autoscale
        self.scale = scale
        self.lp1 = LayerPropagation(num_correction_layers, correction_alpha, correction_adj)
        self.lp2 = LayerPropagation(num_smoothing_layers, smoothing_alpha, smoothing_adj)

    def correct(self, graph, y_soft, y_true, mask):
        numel = mask.shape[0]
        error = paddle.zeros(y_soft.shape)
        error[mask] = y_true - y_soft[mask]

        if self.autoscale:
            smoothed_error = self.prop1(g, error, post_step = lambda y: paddle.clip(y, min=-1, max=1))
            sigma = error[mask].abs().sum() / numel
            scale = sigma / smoothed_error.abs().sum(dim=1, keepdim=True)
            scale[scale.isinf() | (scale > 1000) ] = 1.0

            result = y_soft + scale * smoothed_error
            result[result.isnan()] = y_soft[result.isnan()]
            return result
        else:
            def fix_input(x):
                x[mask] = error[mask]
                return x

            smoothed_error = self.lp1(g, error, post_step=fix_input)
            result = y_soft + self.scale * smoothed_error
            result[result.isnan()] = y_soft[result.isnan()]
            return result


    def smooth(self, graph, y_soft, y_true, mask):
        numel = mask.shape[0]
        assert y_true.shape[0] == numel

        y_soft[mask] = y_true
        return self.lp2(graph, y_soft)