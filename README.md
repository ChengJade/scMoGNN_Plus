# scMoGNN_Plus
scMoGNN_Plus is a GNN-based model, which can predict the modality of single cells (see https://openproblems.bio/ to learn more about the task of modality prediction).


1 Dependencies
======
```
Pytorch
Numpy
Pandas
Scanpy
Sklearn
```

2 Dataset
======
I have uploaded the required dataset in [here](https://drive.google.com/drive/folders/1ZYi9CQ-C7Qg9eL6CirZGmXB7Wj4GFqb8?usp=share_link)

3 Generate extra files
======
This file will generate the gene-gene relations and randomly split the training dataset for training and test.

If you want to get the RMSE loss by just running main.py 1 time, you can directly run generate_extra_files.py and remember to change the default file path to store the data, I usually store it at ./Pretrain/#subtask

And if you want to get the average RMSE loss by runing main.py 5 or more times, you can run generate.sh
``` bash
$ sh ./generate.sh
```

4 Train
=======

Subtask1: GEX2ATAC
-----
For example
```
python main.py bf_alpha_conv4_mean_fullbatch_1000_phase2_inductive_gex2atac 
-pww cos -res res_cat -inres -pwagg alpha -pwalpha=0.5 -bs=60000 
-nm group -ac gelu -em=2 -ro=1 -conv=4 -agg mean -sf 
-lr=0.01 -wd=1e-5 -hid=48 -edd=0.4 -mdd=0.2 -e=1000 -sb -i normal 
-t openproblems_bmmc_multiome_phase2_rna 
-m ./Pretrain/output_pretrain/GEX2ATAC/1 -r ./Pretrain/output_pretrain/GEX2ATAC/1
-l ./Pretrain/output_pretrain/GEX2ATAC/1 -d ./output/datasets/predict_modality/ 
-ef ./Pretrain/output_pretrain/GEX2ATAC/1 -cis
```

Subtask2: GEX2ADT
-----
For example
```
python main.py bf_alpha_conv4_mean_fullbatch_2000_phase2_inductive_gex2adt 
-pww cos -res res_cat -inres -pwagg alpha -pwalpha=0.5 -bs=60000 
-nm group -ac gelu -em=2 -ro=1 -conv=4 -agg mean -sf 
-lr=0.01 -wd=1e-5 -hid=48 -edd=0.4 -mdd=0.2 -e=2000 -sb -i normal 
-t  openproblems_bmmc_cite_phase2_rna 
-m ./Pretrain/output_pretrain/GEX2ADT/1 -r ./Pretrain/output_pretrain/GEX2ADT/1
-l ./Pretrain/output_pretrain/GEX2ADT/1 -d ./output/datasets/predict_modality/ 
-ef ./Pretrain/output_pretrain/GEX2ADT/1 -cis -bas
```

Subtask3: ADT2GEX
-----
For example
```
python main.py bf_alpha_conv4_mean_fullbatch_2000_phase2_inductive_adt2gex
-pww cos -res res_cat -inres -pwagg alpha -pwalpha=0.5 -bs=60000 
-nm group -ac gelu -em=2 -ro=1 -conv=4 -agg mean -sf 
-lr=0.01 -wd=1e-5 -hid=48 -edd=0.4 -mdd=0.2 -e=2000 -sb -i normal 
-t  openproblems_bmmc_cite_phase2_rna 
-m ./Pretrain/output_pretrain/ADT2GEX/1 -r ./Pretrain/output_pretrain/ADT2GEX/1
-l ./Pretrain/output_pretrain/ADT2GEX/1 -d ./output/datasets/predict_modality/ 
-ef ./Pretrain/output_pretrain/ADT2GEX/1 -cis -bas
```

Subtask4: ATAC2GEX
-----
For example
```
python main.py bf_alpha_conv4_mean_fullbatch_2000_phase2_inductive_atac2gex
-pww cos -res res_cat -inres -pwagg alpha -pwalpha=0.5 -bs=60000 
-nm group -ac gelu -em=2 -ro=1 -conv=4 -agg mean -sf 
-lr=0.01 -wd=1e-5 -hid=48 -edd=0.4 -mdd=0.2 -e=2000 -sb -i normal 
-t  openproblems_bmmc_multiome_phase2_mod2 
-m ./Pretrain/output_pretrain/ATAC2GEX/1 -r ./Pretrain/output_pretrain/ATAC2GEX/1
-l ./Pretrain/output_pretrain/ATAC2GEX/1 -d ./output/datasets/predict_modality/ 
-ef ./Pretrain/output_pretrain/ATAC2GEX/1 -cis
```

Note
======
In .train/Network/gnn.py, the uploaded file still adopts the dglnn.SAGEConv method since it would allows CPU overloaded and my GPU is too small to store a whole graph, so I recommend to run on the GPU(required >12G).
If you have a good computing resource, you can use the modified sageconv (from train.Network.sageconv import SAGEConv) to update the edge information.

And for convenience, I adopt the model saving method of scMoGNN (torch.save(model)) which would save the whole model. Actually It is not recommended.
The more recommended method is torch.save(model.state_dict(), path)

The saved training file is too big to upload.

As for the spliting method of the dataset, I follow the code provided by scMoGNN. Noticeably, this model uses the train file to split, but it is right to use provided test dataset to evaluate.
I will upload the new spliting method after.

At last, appreciating the code from [scMoGNN](https://github.com/openproblems-bio/neurips2021_multimodal_topmethods/tree/main/src/predict_modality/methods/DANCE)