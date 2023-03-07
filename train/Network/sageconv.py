"""Torch Module for GraphSAGE layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch
from torch import nn
from torch.nn import functional as F

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair, check_eq_shape
# from ....utils import expand_as_pair,

# import dgl.nn as dglnn


# [docs]
class SAGEConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConv, self).__init__()
        # super(SAGEConv, self).__init__()
        valid_aggre_types = {'mean', 'gcn', 'pool', 'lstm'}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                'Invalid aggregator_type. Must be one of {}. '
                'But got {!r} instead.'.format(valid_aggre_types, aggregator_type)
            )

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation


        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)

        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)

        # self.edge_mp = nn.Linear(1, out_feats, bias=False)

        # self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)


        self.fc_edge = nn.Linear(out_feats*2, 1)
        # self.fc
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        elif bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()



# [docs]
    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)


    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox['m'] # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self._in_src_feats)),
             m.new_zeros((1, batch_size, self._in_src_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'neigh': rst.squeeze(0)}


    #revised by Yu Cheng
    def edge_msg_fn(self, edges):

        # h_src = edges.src['h']
        # print("h_src", h_src.shape)
        # h_dst = edges.dst['neigh']
        # print("h_dst", h_dst.shape)
        # e = torch.cat([h, h_n], dim=1)
        # src, dst, _ = edges.edges()
        # print(src)
        # e = torch.cat((self._in_src_feats[edges.src['id']], self._in_dst_feats[edges.dst['id']]), dim=1)
        # e = torch.cat([h_src, h_dst, edges.data['_edge_weight']], dim=1)
        # print(edges.data['_edge_weight'].shape)
        #第一种 concat全部
        # e = self.fc_edge(torch.cat([edges.src['h'], edges.dst['neigh'], edges.data['_edge_weight']], dim=1)).squeeze() #+ self.edge_mp(edges.data['_edge_weight'].view(-1, 1))
        #第二种 节点concat做映射和原始权重相加
        e = self.fc_edge(torch.cat([edges.src['h'], edges.dst['neigh']], 1)).squeeze() + edges.data['_edge_weight']

        # print("e", e.shape)
        return {'_edge_weight': e}



        # e = self.edge_mlp(e)



# [docs]
    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N_{dst}, D_{out})`
            where :math:`N_{dst}` is the number of destination nodes in the input graph,
            :math:`D_{out}` is the size of the output feature.
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
                # print("tuple")
            else:
                # print("no")
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            #此为消息函数，把起点的h特征拷贝到边的m特征上。copy_src为内置函数。
            msg_fn = fn.copy_u('h', 'm')
            # if edge_weight is not None:
            #     assert edge_weight.shape[0] == graph.number_of_edges()
            #     graph.edata['_edge_weight'] = edge_weight
            #     print("edge", graph.edata['_edge_weight'].shape)
            #     # graph.apply_edges(self.edge_msg_fn)
            #     # graph.edata['_edge_weight'] =
            #     # 如果边的权重不为空，将起点的h特征*边权重， 生成消息m
            #     msg_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            # graph.edata
            # egdes = graph.edata
            h_self = feat_dst
            # print("h_self", h_self.shape)

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                # print("zero")
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats
            lin_before = self._in_dst_feats > self._out_feats
            # Message Passing
            # msg_fn = fn.copy_u('h', 'm')
            if self._aggre_type == 'mean':
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                # print("src_shape", graph.srcdata['h'].shape)
                graph.dstdata['neigh'] = self.fc_self(feat_dst) if lin_before else feat_dst
                # msg_fn = fn.copy_u('h', 'm')
                if edge_weight is not None:
                    assert edge_weight.shape[0] == graph.number_of_edges()
                    # print("edge_here", edge_weight.shape)
                    graph.edata['_edge_weight'] = edge_weight
                    # print("graph_edge", graph.edata['_edge_weight'].shape)
                    graph.apply_edges(self.edge_msg_fn)
                    msg_fn = fn.u_mul_e('h', '_edge_weight', 'm')
                    # graph.edata['_edge_weight'] =
                    # 如果边的权重不为空，将起点的h特征*边权重， 生成消息m
                else:
                    msg_fn = fn.copy_u('h', 'm')

                graph.update_all(msg_fn, fn.mean('m', 'neigh'))

                h_neigh = graph.dstdata['neigh']
                # print("h_neigh", h_neigh.shape)
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)

            elif self._aggre_type == 'gcn':
                check_eq_shape(feat)
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                if isinstance(feat, tuple):  # heterogeneous
                    graph.dstdata['h'] = self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                else:
                    if graph.is_block:
                        graph.dstdata['h'] = graph.srcdata['h'][:graph.num_dst_nodes()]
                    else:
                        graph.dstdata['h'] = graph.srcdata['h']

                msg_fn = fn.copy_u('h', 'm')
                if edge_weight is not None:
                    assert edge_weight.shape[0] == graph.number_of_edges()
                    graph.edata['_edge_weight'] = edge_weight
                    graph.apply_edges(self.edge_msg_fn)
                    # graph.edata['_edge_weight'] =
                    # 如果边的权重不为空，将起点的h特征*边权重， 生成消息m
                    msg_fn = fn.u_mul_e('h', '_edge_weight', 'm')

                graph.update_all(msg_fn, fn.sum('m', 'neigh'))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)

            elif self._aggre_type == 'pool':
                graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                graph.update_all(msg_fn, fn.max('m', 'neigh'))
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])
            elif self._aggre_type == 'lstm':
                graph.srcdata['h'] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])
            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == 'gcn':
                rst = h_neigh
                # add bias manually for GCN
                if self.bias is not None:
                    rst = rst + self.bias
            else:
                rst = self.fc_self(h_self) + h_neigh
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            del h_self, h_neigh
            return rst

# # class Edge_Mlp(nn.module):
# #     def __init__(self, in_feats, out_feats):
# #         super(Edge_Mlp, self).__init__()
# #         # self.nrc = kwargs["no_readout_concatenate"]
# #         # self.conv_layer_size = kwargs["conv_layers"]
# #         self.hid_feats = 100
# #         self.relu = nn.GELU()
# #         self.fc1 = nn.Linear(in_feats, self.hid_feats)
# #         self.bn1 = nn.GroupNorm(4, self.hid_feats)
# #         self.fc2 = nn.Linear(self.hid_feats, self.hid_feats)
# #         self.bn2 = nn.GroupNorm(4, self.hid_feats)
# #         self.fc3 = nn.Linear(self.hid_feats, out_feats)
# #
# #     def forward(self, x):
# #         x = self.fc1(x)
# #         x = self.bn1(x)
# #
# #         x = self.relu(x)
# #         x = self.fc2(x)
# #         x = self.bn2(x)
#
#         x = self.relu(x)
#         x = self.fc3(x)
#
#         return x
#
