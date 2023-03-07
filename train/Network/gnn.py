import torch.nn as nn
import torch
# import dgl.nn as dglnn
import torch.nn.functional as F
import dgl.nn as dglnn
# dglnn.SAGE
# from dgl.heterograph import DGLBlock
# from dgl import function as fn
# from dgl.transform import reverse
from train.Network.mlp import Net
from train.Network.sageconv import SAGEConv
# from ..Network.mlp import Net
class WeightedGCN4(nn.Module):
    def __init__(self, cell_init_feats, cell_init_test, out_feats, feature_size, batch_feats=None, **kwargs):
        super().__init__()
        self.hid_feats = kwargs["hidden_size"]

        self.opw = kwargs["only_pathway"]
        self.npw = kwargs["no_pathway"]
        self.pwagg = kwargs["pathway_aggregation"]

        self.nrc = kwargs["no_readout_concatenate"]
        self.ov = kwargs["overlap"]
        self.ci = kwargs["cell_init"]

        self.em_layer_size  = kwargs["embedding_layers"]
        self.ro_layer_size = kwargs["readout_layers"]
        self.conv_layer_size = kwargs["conv_layers"]

        self.ac = kwargs["activation"]
        self.sa = kwargs["subpath_activation"]
        self.agg = kwargs["agg_function"]
        self.nm = kwargs["normalization"]
        self.opr = kwargs["output_relu"]

        self.res = kwargs["residual"]
        self.inres = kwargs["initial_residual"]
        self.cis = kwargs["cell_init_res"]

        # print(self.cis)
        self.nbf = kwargs["no_batch_features"]
        self.mdd = kwargs["model_dropout"]
        self.edd = kwargs["edge_dropout"]

        self.pwalpha = kwargs["pathway_alpha"]

        self.feature_size = feature_size
        self.batch_feats = batch_feats
        self.out_feats =  out_feats

        self.input3 = cell_init_feats
        self.input3_test = cell_init_test
        if self.cis:
            self.mlp = Net(cell_init_feats.shape[1], **kwargs)
        # 做contrast
        self.sample_h1, self.sample_h2 = None, None


        if self.batch_feats is not None:
            self.extra_encoder = nn.Linear(self.batch_feats.shape[1], self.hid_feats)
        if self.ov:
            self.ov_encoder = nn.Linear(2, self.hid_feats)

        if self.ci == 'none':
            self.embed_cell = nn.Embedding(2, self.hid_feats)
        else:
            self.embed_cell = nn.Linear(100, self.hid_feats)

        self.embed_feat = nn.Embedding(self.feature_size, self.hid_feats)

        self.input_linears = nn.ModuleList()
        self.input_acts = nn.ModuleList()
        self.input_norm = nn.ModuleList()

        self.res_linears = nn.ModuleList()

        for i in range((self.em_layer_size - 1) * 2):
            self.input_linears.append(nn.Linear(self.hid_feats, self.hid_feats))

        if self.ac == 'gelu':
            for i in range((self.em_layer_size - 1) * 2):
                self.input_acts.append(nn.GELU())
        elif self.ac == 'prelu':
            for i in range((self.em_layer_size - 1) * 2):
                self.input_acts.append(nn.PReLU())
        elif self.ac == 'relu':
            for i in range((self.em_layer_size - 1) * 2):
                self.input_acts.append(nn.ReLU())
        elif self.ac == 'leaky_relu':
            for i in range((self.em_layer_size - 1) * 2):
                self.input_acts.append(nn.LeakyReLU())

        if self.nm == 'batch':
            for i in range((self.em_layer_size - 1) * 2):
                self.input_norm.append(nn.BatchNorm1d(self.hid_feats))
        elif self.nm == 'layer':
            for i in range((self.em_layer_size - 1) * 2):
                self.input_norm.append(nn.LayerNorm(self.hid_feats))
        elif self.nm == 'group': #这种
            for i in range((self.em_layer_size - 1) * 2):
                self.input_norm.append(nn.GroupNorm(4, self.hid_feats))

        if self.opw:
            self.edges = ['entail', 'pathway']
        elif self.npw:
            self.edges = ['entail', 'occur']
        else:
            self.edges = ['entail', 'occur', 'pathway']


        # revised by Yu Cheng
        # self.conv_layers = nn.ModuleList()
        # if self.res == 'res_cat':
        #     self.conv_layers.append(
        #         dglnn.HeteroGraphConv(dict(zip(self.edges,
        #         [dglnn.SAGEConv(in_feats=self.hid_feats, out_feats=self.hid_feats, aggregator_type=self.agg, norm=None)
        #         for i in range(len(self.edges))])), aggregate='stack'))
        #     for i in range(self.conv_layer_size - 1):
        #         self.conv_layers.append(dglnn.HeteroGraphConv(dict(zip(self.edges, [
        #             dglnn.SAGEConv(in_feats=self.hid_feats * 2, out_feats=self.hid_feats, aggregator_type=self.agg,
        #                            norm=None) for i in range(len(self.edges))])), aggregate='stack'))
        #
        # else:
        #     for i in range(self.conv_layer_size):
        #         self.conv_layers.append(dglnn.HeteroGraphConv(dict(zip(self.edges, [
        #             dglnn.SAGEConv(in_feats=self.hid_feats, out_feats=self.hid_feats, aggregator_type=self.agg,
        #                            norm=None) for i in range(len(self.edges))])), aggregate='stack'))

        self.conv_layers = nn.ModuleList()
        if self.res == 'res_cat':
            self.conv_layers.append(
                dglnn.HeteroGraphConv(dict(zip(self.edges,
                                               [dglnn.SAGEConv(in_feats=self.hid_feats, out_feats=self.hid_feats,
                                                               aggregator_type=self.agg, norm=None)
                                                for i in range(len(self.edges))])), aggregate='stack'))
            for i in range(self.conv_layer_size - 1):
                self.conv_layers.append(dglnn.HeteroGraphConv(dict(zip(self.edges, [
                    dglnn.SAGEConv(in_feats=self.hid_feats * 2, out_feats=self.hid_feats, aggregator_type=self.agg,
                                   norm=None) for i in range(len(self.edges))])), aggregate='stack'))

        else:
            for i in range(self.conv_layer_size):
                self.conv_layers.append(dglnn.HeteroGraphConv(dict(zip(self.edges, [
                    dglnn.SAGEConv(in_feats=self.hid_feats, out_feats=self.hid_feats, aggregator_type=self.agg,
                                   norm=None) for i in range(len(self.edges))])), aggregate='stack'))

        self.conv_acts = nn.ModuleList()
        self.conv_norm = nn.ModuleList()
        if self.ac == 'gelu':
            for i in range(self.conv_layer_size * 2):
                self.conv_acts.append(nn.GELU())
        elif self.ac == 'prelu':
            for i in range(self.conv_layer_size * 2):
                self.conv_acts.append(nn.PReLU())
        elif self.ac == 'relu':
            for i in range(self.conv_layer_size * 2):
                self.conv_acts.append(nn.ReLU())
        elif self.ac == 'leaky_relu':
            for i in range(self.conv_layer_size * 2):
                self.conv_acts.append(nn.LeakyReLU())

        if self.nm == 'batch':
            for i in range(self.conv_layer_size * len(self.edges)):
                self.conv_norm.append(nn.BatchNorm1d(self.hid_feats))
        elif self.nm == 'layer':
            for i in range(self.conv_layer_size * len(self.edges)):
                self.conv_norm.append(nn.LayerNorm(self.hid_feats))
        elif self.nm == 'group':
            for i in range(self.conv_layer_size * len(self.edges)):
                self.conv_norm.append(nn.GroupNorm(4, self.hid_feats))

        self.att_linears = nn.ModuleList()
        if self.pwagg == 'attention':
            for i in range(self.conv_layer_size):
                self.att_linears.append(nn.Linear(self.hid_feats, self.hid_feats))
        elif self.pwagg == 'one_gate':
            for i in range(self.conv_layer_size):
                self.att_linears.append(nn.Linear(self.hid_feats * 3, self.hid_feats))
        elif self.pwagg == 'two_gate':
            for i in range(self.conv_layer_size * 2):
                self.att_linears.append(nn.Linear(self.hid_feats * 2, self.hid_feats))
        elif self.pwagg == 'cat':
            for i in range(self.conv_layer_size):
                self.att_linears.append(nn.Linear(self.hid_feats * 2, self.hid_feats))

        self.readout_linears = nn.ModuleList()
        self.readout_acts = nn.ModuleList()

        # if self.nrc:
        #     for i in range(self.ro_layer_size-1):
        #         self.readout_linears.append(nn.Linear(self.hid_feats, self.hid_feats))
        #     self.readout_linears.append(nn.Linear(self.hid_feats, out_feats))
        # else:
        #     for i in range(self.ro_layer_size-1):
        #         self.readout_linears.append(nn.Linear(self.hid_feats*self.conv_layer_size, self.hid_featsats*self.conv_layer_size))
        #     self.readout_linears.append(nn.Linear(self.hid_feats*self.conv_layer_size, out_feats))
        #
        if self.nrc:
            for i in range(self.ro_layer_size - 1):
                self.readout_linears.append(nn.Linear(self.hid_feats, self.hid_feats))
            self.readout_linears.append(nn.Linear(self.hid_feats, self.out_feats))
        else:
            for i in range(self.ro_layer_size - 1):
                if self.cis:
                    self.readout_linears.append(
                        nn.Linear(self.hid_feats * (self.conv_layer_size + 1), self.hid_feats * (self.conv_layer_size + 1)))
                    # print("here")
                else:
                    self.readout_linears.append(nn.Linear(self.hid_feats * self.conv_layer_size, self.hid_feats * self.conv_layer_size))
            if self.cis:
                self.readout_linears.append(nn.Linear(self.hid_feats * (self.conv_layer_size + 1), self.out_feats))
            else:
                self.readout_linears.append(nn.Linear(self.hid_feats * self.conv_layer_size, self.out_feats))

        if self.ac == 'gelu':
            for i in range(self.ro_layer_size - 1):
                self.readout_acts.append(nn.GELU())
        elif self.ac == 'prelu':
            for i in range(self.ro_layer_size - 1):
                self.readout_acts.append(nn.PReLU())
        elif self.ac == 'relu':
            for i in range(self.ro_layer_size - 1):
                self.readout_acts.append(nn.ReLU())
        elif self.ac == 'leaky_relu':
            for i in range(self.ro_layer_size - 1):
                self.readout_acts.append(nn.LeakyReLU())

    def attention_agg(self, layer, h0, h):
        # h: h^{l-1}, dimension: (batch, hidden)
        # feats: result from two conv(cell conv and pathway conv), stacked together; dimension: (batch, 2, hidden)
        if h.shape[1] == 1:
            return self.conv_norm[layer * len(self.edges) + 1](h.squeeze(1))
        elif self.pwagg == 'sum':
            return h[:, 0, :] + h[:, 1, :]
        else:
            h1 = h[:, 0, :]
            h2 = h[:, 1, :]

            if self.sa:
                h1 = F.leaky_relu(h1)
                h2 = F.leaky_relu(h2)

            h1 = self.conv_norm[layer * len(self.edges) + 1](h1)
            h2 = self.conv_norm[layer * len(self.edges) + 2](h2)

        if self.pwagg == 'attention':
            feats = torch.stack([h1, h2], 1)
            att = torch.transpose(F.softmax(torch.matmul(feats, self.att_linears[layer](h0).unsqueeze(-1)), 1), 1, 2)
            feats = torch.matmul(att, feats)
            return feats.squeeze(1)
        elif self.pwagg == 'one_gate':
            att = torch.sigmoid(self.att_linears[layer](torch.cat([h0, h1, h2], 1)))
            return att * h1 + (1 - att) * h2
        elif self.pwagg == 'two_gate':
            att1 = torch.sigmoid(self.att_linears[layer * 2](torch.cat([h0, h1], 1)))
            att2 = torch.sigmoid(self.att_linears[layer * 2 + 1](torch.cat([h0, h2], 1)))
            return att1 * h1 + att2 * h2
        elif self.pwagg == 'alpha':
            # if layer == self.conv_layer_size-1:

            self.sample_h1, self.sample_h2 = h1, h2
            # print(self.sample_h1)
            return (1 - self.pwalpha) * h1 + self.pwalpha * h2
        elif self.pwagg == 'cat':
            return self.att_linears[layer](torch.cat([h1, h2], 1))
        elif self.pwagg == 'hadama':
            self.sample_h1, self.sample_h2 = h1, h2
            return 0.6 * (h1*h2)

    def conv(self, graph, layer, h, hist):
        h0 = hist[-1]
        h = self.conv_layers[layer](graph, h, mod_kwargs=dict(zip(self.edges, [{'edge_weight': F.dropout(
            graph.edges[self.edges[i]].data['weight'], p=self.edd, training=self.training)} for i in
                                                                               range(len(self.edges))])))
        # print("ok0")
        if self.mdd > 0:
            h = {'feature': F.dropout(self.conv_acts[layer * 2](self.attention_agg(layer, h0['feature'], h['feature'])),
                                      p=self.mdd, training=self.training),
                 'cell': F.dropout(
                     self.conv_acts[layer * 2 + 1](self.conv_norm[layer * len(self.edges)](h['cell'].squeeze(1))),
                     p=self.mdd, training=self.training)}
            # print("ok1")
        else:
            h = {'feature': self.conv_acts[layer * 2](self.attention_agg(layer, h0['feature'], h['feature'])),
                 'cell': self.conv_acts[layer * 2 + 1](self.conv_norm[layer * len(self.edges)](h['cell'].squeeze(1)))}

        return h

    def forward(self, graph, test=False, gex_feature=None):
        # print("e", graph.edges['occur'].data['weight'])
        input1 = F.leaky_relu(self.embed_feat(graph.nodes['feature'].data['id'])) #feature_size * hid_feat
        input2 = F.leaky_relu(self.embed_cell(graph.nodes['cell'].data['id'])) # cell_size * hid_feat
        # if test:
        # print(input2.shape)
        # print(graph.nodes['cell'].data['id'].shape)
        # input3 = graph.edges['occur'].data['weight'].reshape(graph.nodes['cell'].data['id'].shape[0], self.feature_size)
        # print(input3.shape)
        # print(input1.shape)
        # print(self.batch_feats.shape)
        if not self.nbf:
            input2 += F.leaky_relu(F.dropout(self.extra_encoder(self.batch_feats), p=0.2, training=self.training))[
                      :input2.shape[0]]


        if self.ov:
            input1 += F.leaky_relu(self.ov_encoder(gex_feature))

        hfeat = input1
        hcell = input2
        for i in range(self.em_layer_size - 1, (self.em_layer_size - 1) * 2): #一层隐藏层
            hfeat = self.input_linears[i](hfeat)
            hfeat = self.input_acts[i](hfeat) #经过2层全连接层和激活层
            # print(hfeat.shape)
            if self.nm != 'none':
                hfeat = self.input_norm[i](hfeat) #feat经过group normalize
            if self.mdd > 0:
                hfeat = F.dropout(hfeat, p=self.mdd, training=self.training)

        for i in range(self.em_layer_size - 1):
            hcell = self.input_linears[i](hcell)
            hcell = self.input_acts[i](hcell)
            if self.nm != 'none':
                hcell = self.input_norm[i](hcell)
            if self.mdd > 0:
                hcell = F.dropout(hcell, p=self.mdd, training=self.training)

        #cell feat均经过2层全连接层，激活层以及group normalize(不受batch影响)
        h = {'feature': hfeat, 'cell': hcell}
        hist = [h]

        for i in range(self.conv_layer_size):
            if i == 0 or self.res == 'none':
                pass
            elif self.res == 'res_add':
                if self.inres:
                    h = {'feature': h['feature'] + hist[0]['feature'],
                         'cell': h['cell'] + hist[0]['cell']}

                else:
                    h = {'feature': h['feature'] + hist[-2]['feature'],
                         'cell': h['cell'] + hist[-2]['cell']}

            elif self.res == 'res_cat':
                if self.inres:
                    h = {'feature': torch.cat([h['feature'], hist[0]['feature']], 1),
                         'cell': torch.cat([h['cell'], hist[0]['cell']], 1)}
                else:
                    h = {'feature': torch.cat([h['feature'], hist[-2]['feature']], 1),
                         'cell': torch.cat([h['cell'], hist[-2]['cell']], 1)}

            h = self.conv(graph, i, h, hist)
            # print("ok3")
            hist.append(h)
            # print("ok4")

        # revised by Yu Cheng
        if self.cis:
            h_cell = self.mlp(self.input3)
            if test:
                h_cell_test = self.mlp(self.input3_test)

            if not self.nrc:
                # print("ok5")
                h = torch.cat([i['cell'] for i in hist[1:]], 1)
                # print("ok5")
                # print("h", h.shape)
                #########################
                if not test:
                    h = torch.cat([h, h_cell], 1)
                else:
                    h = torch.cat([h, h_cell_test], 1)
                # print("h2", h.shape)
                # print(h_graph.shape)
            else:
                h = h['cell']
            # print("cell", h_cell_test.shape)
        # print("ok")

        #origin
        # if not self.nrc:
        #     h = torch.cat([i['cell'] for i in hist[1:]], 1)
        # else:
        #     h = h['cell']

        # revised by Yu Cheng
        # if not self.nrc:
        #     # print("ok5")
        #     h = torch.cat([i['cell'] for i in hist[1:]], 1)
        #     # print("ok5")
        #     # print("h", h.shape)
        #     #########################
        #     if not test:
        #         h = torch.cat([h, h_cell], 1)
        #     else:
        #         h = torch.cat([h, h_cell_test], 1)
        #     # print("h2", h.shape)
        #     # print(h_graph.shape)
        # else:
        #     h = h['cell']

        for i in range(self.ro_layer_size - 1):
            # print("here")
            h = self.readout_linears[i](h)
            # print("here")
            h = F.dropout(self.readout_acts[i](h), p=self.mdd, training=self.training)
        h = self.readout_linears[-1](h)



        if self.opr == 'relu':
            return F.relu(h)
        elif self.opr == 'leaky_relu':
            return F.leaky_relu(h)

        return h


# class udfEdgeWeightNorm(nn.Module):
#     def __init__(self, norm='both', eps=0.):
#         super(udfEdgeWeightNorm, self).__init__()
#         self._norm = norm
#         self._eps = eps
#
#     def forward(self, graph, edge_weight):
#         with graph.local_scope():
#             if isinstance(graph, DGLBlock):
#                 graph = block_to_graph(graph)
#             if len(edge_weight.shape) > 1:
#                 raise DGLError('Currently the normalization is only defined '
#                                'on scalar edge weight. Please customize the '
#                                'normalization for your high-dimensional weights.')
#             if self._norm == 'both' and th.any(edge_weight <= 0).item():
#                 raise DGLError('Non-positive edge weight detected with `norm="both"`. '
#                                'This leads to square root of zero or negative values.')
#
#             dev = graph.device
#             graph.srcdata['_src_out_w'] = torch.ones((graph.number_of_src_nodes())).float().to(dev)
#             graph.dstdata['_dst_in_w'] = torch.ones((graph.number_of_dst_nodes())).float().to(dev)
#             graph.edata['_edge_w'] = edge_weight
#
#             if self._norm in ['both', 'column', 'left']:
#                 reversed_g = reverse(graph)
#                 reversed_g.edata['_edge_w'] = edge_weight
#                 reversed_g.update_all(fn.copy_edge('_edge_w', 'm'), fn.sum('m', 'out_weight'))
#                 degs = reversed_g.dstdata['out_weight'] + self._eps
#                 norm = torch.pow(degs, -0.5)
#                 graph.srcdata['_src_out_w'] = norm
#
#             if self._norm in ['both', 'row', 'right']:
#                 graph.update_all(fn.copy_edge('_edge_w', 'm'), fn.sum('m', 'in_weight'))
#                 degs = graph.dstdata['in_weight'] + self._eps
#                 if self._norm == 'both':
#                     norm = torch.pow(degs, -0.5)
#                 else:
#                     norm = 1.0 / degs
#                 graph.dstdata['_dst_in_w'] = norm
#
#             graph.apply_edges(lambda e: {'_norm_edge_weights': e.src['_src_out_w'] * \
#                                                                e.dst['_dst_in_w'] * \
#                                                                e.data['_edge_w']})
#             return graph.edata['_norm_edge_weights']
#
