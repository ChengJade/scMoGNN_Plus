import anndata as ad
import pickle
import numpy as np
from collections import defaultdict
import random

import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_folder', default = '/home/chengyu/Pycharm/Predict_Modality/output/datasets/predict_modality')
parser.add_argument('-ef', '--extra_files_folder', default = '/home/chengyu/Pycharm/Predict_Modality/DANCE/pretrain/data/')
parser.add_argument('-i', default=0)
args = parser.parse_args()

def load_pw():
    with open(args.extra_files_folder + '/h.all.v7.4.entrez.gmt') as gmt:
        gene_list = gmt.read().split()
    # print("gen_list", gene_list)
    gene_sets_entrez = defaultdict(list)

    indicator = 0
    for ele in gene_list:
        if not ele.isnumeric() and indicator == 1:
            indicator = 0
            continue
        if not ele.isnumeric() and indicator == 0: #既不是数字，indicator=0
            indicator = 1
            gene_set_name = ele
        else:
            gene_sets_entrez[gene_set_name].append(ele) #将每个组的基因的名字append,

    with open(args.extra_files_folder + '/h.all.v7.4.symbols.gmt') as gmt:
        gene_list = gmt.read().split()
    # print(gene_list)
    gene_sets_symbols = defaultdict(list)

    for ele in gene_list:
        if ele in gene_sets_entrez:
            gene_set_name = ele
        elif not ele.startswith( 'http://' ):
            gene_sets_symbols[gene_set_name].append(ele)
    # print(len([i[1] for i in gene_sets_symbols.items()])) #50组不同类型的基因，每个组有各自的基因
    return [i[1] for i in gene_sets_symbols.items()]

# def graph_construct(train_mod1, test_mod1):
def graph_construct(train_mod1):
    counter = 0
    total = 0
    # input_train_mod1 = train_mod1.X
    input_train_mod1 = train_mod1.X
    # input_test_mod1 = test_mod1.X
    feature_index = train_mod1.var['feature_types'].index.tolist() #有多少个feature node
    # feature_index_test = test_mod1.var['feature_types'].index.tolist()
    # f = feature_index_test + feature_index

    # print(feature_index, len(feature_index))
    # print(feature_index_test, len(feature_index_test))
    # print(f, len(f))
    new_pw = [] #统一下标，将外部知识和mod1对应
    for i in pw: #先遍历第一维度，是不同的set
        new_pw.append([]) #每一个新的set就会多一个列表
        for j in i: # 同一set下的基因，
            if j in feature_index: #如果这个基因在feature里面
                new_pw[-1].append(feature_index.index(j)) #返回这个featureNode的下标 index


    # cos similarity weight
    uu=[]
    vv=[]
    ee=[]
    for i in new_pw:#每一个组下面的
        for j in i: #具体的feature_node
            for k in i:
                if j!=k:
                    uu.append(j)
                    vv.append(k)
                    sj = np.sqrt(np.dot(input_train_mod1[:,j].toarray().T, input_train_mod1[:,j].toarray()).item())
                    sk = np.sqrt(np.dot(input_train_mod1[:,k].toarray().T, input_train_mod1[:,k].toarray()).item())
                    jk = np.dot(input_train_mod1[:,j].toarray().T, input_train_mod1[:,k].toarray())
                    cossim = jk/sj/sk
                    ee.append(cossim) #算权重，关系
                    
    return uu, vv, ee

print("Loading pw")
pw = load_pw()

print("Generating 'pw.pkl'")
# Generate pw.pkl
subtask = 'openproblems_bmmc_cite_phase2_rna'
subtask_folder = args.data_folder + '/' + subtask + '/'
subtask_filename = subtask_folder + subtask + '.censor_dataset.output_{}.h5ad'
uu, vv, ee = graph_construct(ad.read_h5ad(subtask_filename.format('train_mod1')))
#revised by Yu Cheng
root_dir = args.extra_files_folder + str(args.i)
if not os.path.exists(root_dir):
    os.mkdir(root_dir)
# pickle.dump([uu,vv,ee], open(args.extra_files_folder + '/' + str(args.i) + '/pw.pkl', 'wb'))
pickle.dump([uu,vv,ee], open(root_dir + '/pw.pkl', 'wb'))

print("Generating 'pw_multiome.pkl'")
subtask = 'openproblems_bmmc_multiome_phase2_rna'
subtask_folder = args.data_folder + '/' + subtask + '/'
subtask_filename = subtask_folder + subtask + '.censor_dataset.output_{}.h5ad'
uu, vv, ee = graph_construct(ad.read_h5ad(subtask_filename.format('train_mod1')))
# root_dir = args.extra_files_folder + '/' + str(args.i)
# if not os.path.exists(root_dir):
#     os.mkdir(root_dir)
# pickle.dump([uu,vv,ee], open(args.extra_files_folder + str(args.i) + '/pw_multiome.pkl', 'wb'))
pickle.dump([uu,vv,ee], open(root_dir + '/pw_multiome.pkl', 'wb'))

print("Generating 'phase2_mask.pkl'")
subtasks = ['openproblems_bmmc_cite_phase2_rna', 'openproblems_bmmc_cite_phase2_mod2', 'openproblems_bmmc_multiome_phase2_rna', 'openproblems_bmmc_multiome_phase2_mod2']
task_names = ['gex2adt', 'adt2gex', 'gex2atac', 'atac2gex']
mask = {}

for ts in range(4):
    subtask = subtasks[ts]
    mask[subtask] = {}
    subtask_folder = args.data_folder + '/' + subtask + '/'
    subtask_filename = subtask_folder + subtask + '.censor_dataset.output_{}.h5ad'
    train_mod1 = ad.read_h5ad(subtask_filename.format('train_mod1'))
    l = list(range(train_mod1.X.shape[0])) #cell的个数
    random.shuffle(l)
    train_size = int(train_mod1.X.shape[0] * 0.85) #手动切分数据集
    valid_size = train_mod1.X.shape[0] - train_size
    mask[subtask]['train'] = l[:train_size]
    mask[subtask]['test'] = l[-valid_size:]

import pickle
# pickle.dump(mask, open(args.extra_files_folder + str(args.i) +'/phase2_mask.pkl','wb')) #多个任务同时存储
pickle.dump(mask, open(root_dir +'/phase2_mask.pkl','wb'))

print("Generating 'phase2_mask_sep.pkl'")
subtask = 'openproblems_bmmc_cite_phase2_rna' #gex->adt
subtask_folder = args.data_folder + '/' + subtask + '/'
subtask_filename = subtask_folder + subtask + '.censor_dataset.output_{}.h5ad'

train_mod1 = ad.read_h5ad(subtask_filename.format('train_mod1'))

def get_index(batch):
    index = []
    for i in train_mod1[train_mod1.obs['batch']==batch].obs['batch'].index:
        index.append(list(train_mod1.obs['batch'].index).index(i))
    return index

s3d1 = get_index('s3d1')
s3d7 = get_index('s3d7')
s1d2 = get_index('s1d2')

test = s3d7+s1d2
train = [i for i in range(train_mod1.X.shape[0]) if i not in (test + s3d1)]

gex2adt = {}
gex2adt['test'] = test
gex2adt['train'] = train

mask = {}
mask['openproblems_bmmc_cite_phase2_rna'] = gex2adt
pickle.dump(mask, open(root_dir +'/phase2_mask_sep.pkl', 'wb'))
# pickle.dump(mask, open(args.extra_files_folder + str(args.i) +'/phase2_mask_sep.pkl', 'wb'))