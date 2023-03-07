# -*- coding: utf-8 -*-
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd
import math
import re
# import argparse
import pickle as pk
import anndata as ad

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
# from tqdm import tqdm

# from collections import defaultdict
# import dgl.nn as dglnn
# import sys
# sys.path.append(r"/home/chengyu/Pycharm/Predict_Modality/DANCE")
from train.Network.autoencoder import AutoEncoder2En1De
from train.Network.gnn import WeightedGCN4
from train.args import gnn_parser

from train.Network.contrast_model import WithinEmbedContrast
from train.losses.barlow_twins import BarlowTwins
# suggestion: use same WeightedGCN4 as eval component
# to use, uncomment the following line


# from ..train.Network.autoencoder import AutoEncoder2En1De
# from train.Network.gnn import WeightedGCN4
# from .train.
# from .train.args import gnn_parser



def validate(model, gtest, test_labels, logger, test_size):
    model.eval()
    with torch.no_grad():
        logits = model(gtest, test=True)
        logits = logits[-test_size:]
        labels = test_labels
        loss = math.sqrt(F.mse_loss(logits, labels).item())
        # print('validation loss', loss)
        logger.write(f'validation loss:  {loss}\n')
        logger.flush()
        return loss

def run(g, cell_init_feats, cell_init_test, train_labels, output_size, feature_size,
        gtest, test_labels, test_size,
        batch_feats, device, logger, **kwargs):

    model = WeightedGCN4(cell_init_feats, cell_init_test, output_size, feature_size, batch_feats, **kwargs).to(device)
    # print("initizing")
    prefix = kwargs["prefix"]
    logger.write(str(model) + '\n')
    opt = torch.optim.AdamW(model.parameters(), lr=kwargs["learning_rate"], weight_decay=kwargs["weight_decay"])
    criterion = nn.MSELoss()
    val = []
    tr = []
    minval = 100
    minvep = -1
    # loop = tqdm(range(kwargs["epoch"]))
    for epoch in range(kwargs["epoch"]):
        logger.write(f'epoch:  {epoch}\n')
        model.train()
        logits = model(g)
        contras_model = WithinEmbedContrast(loss=BarlowTwins())
        # print(model.sample_h1)
        contras_loss = contras_model(model.sample_h1, model.sample_h2)
        # print(contras_loss.item())
        loss = criterion(logits, train_labels) + 0.1 * contras_loss
        running_loss = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.empty_cache()
        tr.append(math.sqrt(running_loss))
        # print("Epoch {} training loss: {}".format(epoch, tr[-1]))
        logger.write(f'training loss:  {tr[-1]}\n')
        logger.flush()
        # print("I am here 2")

        if True:  # epoch % 5 == 4:
            val.append(validate(model, gtest, test_labels, logger, test_size))
            print("Epoch {} training loss: {:.5f},  validation loss: {:.5f}".format(epoch, tr[-1], val[-1]))
            # logger.write(f'validation loss:  {tr[-1]}\n')
            # logger.flush()
        # loop.set_description(f'Epoch [{epoch}/{kwargs["epoch"]}]')
        # loop.set_postfix(loss=tr[-1])

        if epoch > 100 and val[-1] < minval:
            minval = val[-1]
            minvep = epoch
            if kwargs["save_best"]:
                torch.save(model, f'{kwargs["model_folder"]}/{prefix}_cell.pkl')

        if kwargs["early_stopping"] > 0 and min(val[-kwargs["early_stopping"]:]) > minval:
            logger.write('Early stopped.\n')
            break

    df = pd.DataFrame({'train': tr, 'val': val})
    df.to_csv(f'{kwargs["result_folder"]}/{prefix}_cell.csv', index=False)
    state = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch - 1}
    if kwargs["save_final"]:
        torch.save(state, f'{kwargs["model_folder"]}/{prefix}_cell.epoch{epoch}.ckpt')
    logger.write(f'epoch {minvep} minimal valid:  {minval} with training:  {tr[minvep]}\n')
    logger.close()

    return model



def pre_train():
    parser = gnn_parser()
    args = parser.parse_args()
    kwargs = vars(parser.parse_args())
    # print(kwargs["no_batch_features"])
    # print(kwargs["only_pathway"])f
    # print(args.overlap, args.no_pathway, args.only_pathway)
    PREFIX = args.prefix
    logger = open(f'{args.log_folder}/{PREFIX}_cell.log', 'w')
    logger.write(str(args)+'\n')

    device = torch.device('cpu')#cuda:0' if torch.cuda.is_available() else 'cpu')

    # torch.set_num_threads(args.cpu)
    subtask = args.subtask
    subtask_folder = args.data_folder + subtask + '/'
    subtask_filename = subtask_folder + subtask + '.censor_dataset.output_{}.h5ad'

    train_mod1 = ad.read_h5ad(subtask_filename.format('train_mod1'))
    train_mod2 = ad.read_h5ad(subtask_filename.format('train_mod2'))
    # print(train_mod2.var['feature_types'][0])

    # import pickle
    if args.batch_seperation: #不同的数据集可能划分训练集测试集的标准不同
        mask = pk.load(open(os.path.join(args.extra_files_folder, 'phase2_mask_sep.pkl'), 'rb'))[subtask]
    else:
        mask = pk.load(open(os.path.join(args.extra_files_folder, 'phase2_mask.pkl'), 'rb'))[subtask]

    # This will get passed to the method 切分数据集 维度：
    input_train_mod1 = train_mod1.X[mask['train']]
    input_train_mod2 = train_mod2.X[mask['train']]
    input_test_mod1 =  train_mod1.X[mask['test']]
    true_test_mod2 =  train_mod2.X[mask['test']]

    # print("mod1", input_train_mod1.shape[0],input_train_mod1.shape[1])
    # print("mod2", input_train_mod2.shape[0], input_train_mod2.shape[1])
    FEATURE_SIZE = input_train_mod1.shape[1]
    CELL_SIZE = input_train_mod1.shape[0] + input_test_mod1.shape[0]
    # CELL_SIZE_Train = input_train_mod1.shape[0]
    # CELL_SIZE_Test = input_test_mod1.shape[0]
    OUTPUT_SIZE = input_train_mod2.shape[1] #输出是feature的维度？
    TRAIN_SIZE = input_train_mod1.shape[0]
    TEST_SIZE = input_test_mod1.shape[0]
    # print(FEATURE_SIZE, CELL_SIZE, OUTPUT_SIZE)

    # Pathway Features
    # import pickle

    pww = args.pathway_weight
    npw = args.no_pathway
    uu = []
    vv = []
    ee = []
    if npw:
        pass
    elif pww == 'cos' and subtask == 'openproblems_bmmc_cite_phase2_rna':
        uu, vv, ee = pk.load(open(os.path.join(args.extra_files_folder, 'pw.pkl'), 'rb'))
        ee = [e.item() for e in ee]
    elif pww == 'cos' and subtask == 'openproblems_bmmc_multiome_phase2_rna':
        uu, vv, ee = pk.load(open(os.path.join(args.extra_files_folder, 'pw_multiome.pkl'), 'rb'))
        ee = [e.item() for e in ee]

    pwth = args.pathway_threshold
    if pwth > 0:
        nu = []
        nv = []
        ne = []

        for i in range(len(uu)):
            if ee[i] > pwth:
                ne.append(ee[i])
                nu.append(uu[i])
                nv.append(vv[i])
        uu, vv, ee = nu, nv, ne

    # Batch Features
    batch_feats = []
    if args.no_batch_features:
        batch_feats = None
    else:
        cells = []
        columns = ['cell_mean', 'cell_std', 'nonzero_25%', 'nonzero_50%', 'nonzero_75%', 'nonzero_max', 'nonzero_count',
                   'nonzero_mean', 'nonzero_std', 'batch']

        bcl = list(train_mod1.obs['batch'][mask['train']])
        for i, cell in enumerate(input_train_mod1):
            cell = cell.toarray()
            nz = cell[np.nonzero(cell)]
            cells.append([cell.mean(), cell.std(), np.percentile(nz, 25), np.percentile(nz, 50), np.percentile(nz, 75),
                          cell.max(), len(nz) / 1000, nz.mean(), nz.std(), bcl[i]])

        bcl = list(train_mod1.obs['batch'][mask['test']])
        for i, cell in enumerate(input_test_mod1):
            cell = cell.toarray()
            nz = cell[np.nonzero(cell)]
            cells.append([cell.mean(), cell.std(), np.percentile(nz, 25), np.percentile(nz, 50), np.percentile(nz, 75),
                          cell.max(), len(nz) / 1000, nz.mean(), nz.std(), bcl[i]])

        cell_features = pd.DataFrame(cells, columns=columns)
        batch_source = cell_features.groupby('batch').mean().reset_index()
        batch_list = batch_source.batch.tolist()
        batch_source = batch_source.drop('batch', axis=1).to_numpy().tolist()
        b2i = dict(zip(batch_list, range(len(batch_list))))
        batch_feats = []

        for b in train_mod1.obs['batch'][mask['train']]:
            batch_feats.append(batch_source[b2i[b]])

        for b in train_mod1.obs['batch'][mask['test']]:
            batch_feats.append(batch_source[b2i[b]])

        batch_feats = torch.tensor(batch_feats).float()

    # Graph construction
    if args.cell_init == 'none':
        cell_ids = torch.ones(CELL_SIZE).long()
    else:
        #记得改成GPU
        model = AutoEncoder2En1De(FEATURE_SIZE, OUTPUT_SIZE, 100)
        model.load_state_dict(torch.load(f'ensemble_models/{subtask}_auto_encoder_model.pth', map_location='cpu'))
        model.eval()
        with torch.no_grad():
            cell_ids = torch.cat([model.get_embedding(torch.from_numpy(input_train_mod1.toarray())).detach(),
                                  model.get_embedding(torch.from_numpy(input_test_mod1.toarray())).detach()], 0).float()

    def graph_construction(u, v, e, #uu, vv, ee, cell_ids,
                           train_size, feature_size,
                           test=False, **kwargs):
        # if args.only_pathway:
        only_pathway = kwargs["only_pathway"]
        inductive = kwargs["inductive"]
        no_pathway = kwargs["no_pathway"]
        # print(g)
        if only_pathway:
            graph_data = {
                ('feature', 'entail', 'cell'): (v, u),
                ('feature', 'pathway', 'feature'): (uu, vv),
            }
            graph = dgl.heterograph(graph_data)

            if inductive != 'trans':
                graph.nodes['cell'].data['id'] = cell_ids[:train_size] if not test else cell_ids
            else:
                graph.nodes['cell'].data['id'] = cell_ids

            graph.nodes['feature'].data['id'] = torch.arange(feature_size).long()
            graph.edges['entail'].data['weight'] = e
            graph.edges['pathway'].data['weight'] = torch.tensor(ee).float()

        elif no_pathway:
            if inductive == 'opt':
                graph_data = {
                    ('cell', 'occur', 'feature'): (u, v) if not test else (
                        u[:g.edges(etype='occur')[0].shape[0]], v[:g.edges(etype='occur')[0].shape[0]]),
                    ('feature', 'entail', 'cell'): (v, u),
                }

            else:
                graph_data = {
                    ('cell', 'occur', 'feature'): (u, v),
                    ('feature', 'entail', 'cell'): (v, u),
                }

            graph = dgl.heterograph(graph_data)

            if inductive != 'trans':
                graph.nodes['cell'].data['id'] = cell_ids[:TRAIN_SIZE] if not test else cell_ids
            else:
                graph.nodes['cell'].data['id'] = cell_ids
            graph.nodes['feature'].data['id'] = torch.arange(FEATURE_SIZE).long()
            graph.edges['entail'].data['weight'] = e
            graph.edges['occur'].data['weight'] = e[:graph.edges(etype='occur')[0].shape[0]] #为啥要这样写

        else:
            if inductive == 'opt':

                graph_data = {
                    ('cell', 'occur', 'feature'): (u, v) if not test else (
                        u[:g.edges(etype='occur')[0].shape[0]], v[:g.edges(etype='occur')[0].shape[0]]),
                    ('feature', 'entail', 'cell'): (v, u),
                    ('feature', 'pathway', 'feature'): (uu, vv),
                }
            else:
                graph_data = {
                    ('cell', 'occur', 'feature'): (u, v),
                    ('feature', 'entail', 'cell'): (v, u),
                    ('feature', 'pathway', 'feature'): (uu, vv),
                }
            graph = dgl.heterograph(graph_data)
            # print(graph)
            if inductive != 'trans':
                graph.nodes['cell'].data['id'] = cell_ids[:TRAIN_SIZE] if not test else cell_ids
            else:
                graph.nodes['cell'].data['id'] = cell_ids

            graph.nodes['feature'].data['id'] = torch.arange(FEATURE_SIZE).long()
            graph.edges['entail'].data['weight'] = e
            graph.edges['occur'].data['weight'] = e[:graph.edges(etype='occur')[0].shape[0]]
            graph.edges['pathway'].data['weight'] = torch.tensor(ee).float()

        return graph

    #normal
    if args.inductive != 'trans':
        #不为0的元素的行数 Num*1, [0,0,0,1,1,.....]
        u = torch.from_numpy(
            np.concatenate([np.array(t.nonzero()[0] + i) for i, t in enumerate(input_train_mod1)], axis=0))
        # 不为0的元素的列数 Num*1, [0,0,0,1,1,.....]
        v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1], axis=0))
        # 边的权重，转为稀疏矩阵
        e = torch.from_numpy(input_train_mod1.tocsr().data).float()
        cell_init_feats =  torch.from_numpy(input_train_mod1.toarray())
        # cell_init = e.reshape()
        # print(e.shape)

        # print("coming")
        g = graph_construction(u, v, e, TRAIN_SIZE, FEATURE_SIZE, **kwargs)

        u = torch.from_numpy(np.concatenate([np.array(t.nonzero()[0] + i) for i, t in enumerate(input_train_mod1)] + \
                                            [np.array(t.nonzero()[0] + i + TRAIN_SIZE) for i, t in
                                             enumerate(input_test_mod1)], axis=0))
        v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1] + \
                                            [np.array(t.nonzero()[1]) for t in input_test_mod1], axis=0))
        e = torch.from_numpy(
            np.concatenate([input_train_mod1.tocsr().data, input_test_mod1.tocsr().data], axis=0)).float()
        # e = torch.from_numpy(input_test_mod1.tocsr().data).float()

        gtest = graph_construction(u, v, e, TRAIN_SIZE, FEATURE_SIZE, test=True, **kwargs) #这里记得改一下
        cell_init_test =torch.from_numpy(np.concatenate([input_train_mod1.toarray(), input_test_mod1.toarray()], axis=0))
    else:
        u = torch.from_numpy(np.concatenate([np.array(t.nonzero()[0] + i) for i, t in enumerate(input_train_mod1)] + \
                                            [np.array(t.nonzero()[0] + i + TRAIN_SIZE) for i, t in
                                             enumerate(input_test_mod1)], axis=0))
        v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1] + \
                                            [np.array(t.nonzero()[1]) for t in input_test_mod1], axis=0))
        e = torch.from_numpy(
            np.concatenate([input_train_mod1.tocsr().data, input_test_mod1.tocsr().data], axis=0)).float()
        g = graph_construction(u, v, e)

        gtest = g

    if args.overlap:
        f = open('ADT_GEX.txt', 'r')
        gex_list = []
        for line in f:
            rs = re.search(':  (.*) contained in both Protein and RNA gene dataset', line)
            if rs is not None:
                gex_list.append(rs[1])

            rs = re.search('but not in RNA gene dataset \((.*)\)', line)
            if rs is not None and rs[1].find('not exist') == -1 and rs[1] != 'HLA-A,HLA-B,HLA-C' and rs[1] != 'CD1a':
                gex_list.append(rs[1])

        gex_list += ['HLA-A', 'HLA-B', 'HLA-C']
        gex_feature = torch.zeros(FEATURE_SIZE, 2)
        feature_list = list(train_mod1.var['feature_types'].index)
        for gex in gex_list:
            if gex in feature_list:
                ind = feature_list.index(gex)
                gex_feature[ind][0] = 1

        gex_list = pd.read_csv('ATAC_GEX_Overlap.csv').overlap_GEX.unique().tolist()
        for gex in gex_list:
            if gex in feature_list:
                gex_feature[feature_list.index(gex)][1] = 1

    # data loader
    train_labels = torch.from_numpy(input_train_mod2.toarray())
    test_labels = torch.from_numpy(true_test_mod2.toarray())
    BATCH_SIZE = args.batch_size

    # print("Batch_size", BATCH_SIZE)

    # device = args.device
    g = g.to(device)
    # print("here")
    gtest = gtest.to(device)
    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)
    if args.overlap:
        gex_feature = gex_feature.to(device)
    if not args.no_batch_features:
        batch_feats = batch_feats.to(device)


    # model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # start_epoch = checkpoint['epoch'] + 1


    run(g, cell_init_feats, cell_init_test, train_labels, OUTPUT_SIZE, FEATURE_SIZE,
        gtest, test_labels, TEST_SIZE,
        batch_feats, device, logger, **kwargs)
    
# model

if __name__ == '__main__':
    pre_train()

