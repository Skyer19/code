import copy
import gc
import json

import tempfile
import os

# 检查 tempfile 模块使用的临时文件目录
temp_dir = tempfile.gettempdir()
print("Updated temp directory:", temp_dir)

import sys
import importlib

new_path = '/data/mr423/project/ageClassification/'
if new_path not in sys.path:
    sys.path.insert(0, new_path)

# relaod the scgpt files
import scgpt
print("scgpt location: ", scgpt.__file__)
importlib.reload(scgpt)


from pathlib import Path
import shutil
import sys
import time
from typing import List, Tuple, Dict, Union, Optional
import warnings
import pandas as pd
# from . import asyn
import torch
from anndata import AnnData
import scanpy as sc
import scvi
import seaborn as sns
import numpy as np
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from sklearn.metrics import confusion_matrix

sys.path.insert(0, "../")
import scgpt as scg

from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed

sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

os.environ["WANDB_MODE"]= "offline"


######################################################################
# Settings for wandb mentior
######################################################################

hyperparameter_defaults = dict(
    seed=0,
    do_train=True,
    load_model="/data/mr423/project/pre_trained_model/scGPT_human",
    n_bins=101,

    epochs=15, # 2 !!!!!!!!!!!!  test only
    lr=0.0001,
    batch_size=128,   # 128 !!!!!!!!!!!!  test only

    layer_size=512, # 128
    nlayers=12,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead=8,  # number of heads in nn.MultiheadAttention
    
    dropout=0.0,  # dropout probability

    use_fast_transformer=True,
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision
    freeze = True, #freeze
)

run = wandb.init(
    config=hyperparameter_defaults,
    project="age_clusters",
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
)
config = wandb.config
print(config)

set_seed(config.seed)


######################################################################
# Settings for input and preprocessing
######################################################################
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_value = "auto"  # for masked values, now it should always be auto

max_seq_len = 3001
n_bins = config.n_bins

# input/output representation
input_style = "binned"  # "normed_raw", "log1p", or "binned"
input_emb_style = "category"  # "category" or "continuous" or "scaling"
cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"


# settings for training
CLS = True  # celltype classification objective



######################################################################
# Settings for optimizer
######################################################################
lr = config.lr
batch_size = config.batch_size
eval_batch_size = config.batch_size
epochs = config.epochs
early_stop = 10



######################################################################
# Settings for the model
######################################################################
use_fast_transformer = config.use_fast_transformer
fast_transformer_backend = "flash"  # "linear" or "flash"


embsize = config.layer_size  # embedding dimension
d_hid = config.layer_size  # dimension of the feedforward network in TransformerEncoder
nlayers = config.nlayers  # number of TransformerEncoderLayer in TransformerEncoder
nhead = config.nhead  # number of heads in nn.MultiheadAttention
dropout = config.dropout  # dropout probability



######################################################################
# Validate the settings
######################################################################
# assert input_style in ["normed_raw", "log1p", "binned"]
# assert output_style in ["normed_raw", "log1p", "binned"]

assert input_emb_style in ["category", "continuous", "scaling"]
if input_style == "binned":
    if input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif input_style == "log1p" or input_style == "normed_raw":
    if input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw input."
        )

if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins  # for padding gene expr values
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins



######################################################################
# Settings for the running recording
######################################################################
dataset_name = 'biobank-age-cluster'
save_dir = Path(f"/data/mr423/project/code/ageClassification/record/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)

print(f"save to {save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")



######################################################################
# Data loading
######################################################################
adata = sc.read("/data/mr423/project/data/3-OLINK_data_train_withOutlier_all.h5ad")
adata_test = sc.read("/data/mr423/project/data/3-OLINK_data_test_withOutlier_all.h5ad")

print(adata.shape)
print(adata_test.shape)

adata.obs["batch_id"]  = adata.obs["str_batch"] = "0"
adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "1" 

adata.var.set_index(adata.var["gene_name"], inplace=True)
adata_test.var.set_index(adata.var["gene_name"], inplace=True)

data_is_raw = False

adata_test_raw = adata_test.copy()
adata = adata.concatenate(adata_test, batch_key="str_batch")

# make the batch category column
batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels


ageGroup_id_labels = adata.obs["Age_Group"].astype("category").cat.codes.values
# ageGroup_types = adata.obs["Age_Group"].unique()
adata.obs["ageGroup_id"] = ageGroup_id_labels


n_cls = len(np.unique(ageGroup_id_labels))

id2type = dict(enumerate(adata.obs["Age_Group"].astype("category").cat.categories))
print(id2type)

adata.var["gene_name"] = adata.var.index.tolist()



######################################################################
# The pre-trained model
######################################################################
if config.load_model is not None:
    model_dir = config.load_model

    model_file = "/data/mr423/project/code/ageClassification/trained_model/trained_all_model/model.pt"
    vocab_file = "/data/mr423/project/code/ageClassification/trained_model/trained_all_model/vocab.json"


    vocab = GeneVocab.from_file(vocab_file)
    shutil.copy(vocab_file, save_dir / "vocab.json")
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]


    print("**** actual model parameters ****")
    print(f'layer_size = embsize: {embsize} = d_hid: {d_hid}, n_layers: {nlayers}, nhead: {nhead}')
    print("**** actual model parameters ****\n")

# set up the preprocessor, use the args to config the workflow
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=False,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=3000,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)


adata_test = adata[adata.obs["str_batch"] == "1"]
adata = adata[adata.obs["str_batch"] == "0"]

# preprocessor(adata, batch_key=None)
preprocessor(adata_test, batch_key=None)

######################################################################
# Split the data to train and test
######################################################################
input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
    "normed_raw": "X_normed",
    "log1p": "X_normed",
    "binned": "X_binned",
}[input_style]

# all_counts = (
#     adata.layers[input_layer_key].A
#     if issparse(adata.layers[input_layer_key])
#     else adata.layers[input_layer_key]
# )
genes = adata_test.var["gene_name"].tolist()

ageGroup_labels = adata.obs["ageGroup_id"].tolist()  # make sure count from 0
ageGroup_labels = np.array(ageGroup_labels)


# (
#     train_data,
#     valid_data,
#     train_ageGroup,
#     valid_ageGroup,
# ) = train_test_split(
#     all_counts, ageGroup_labels, test_size=0.2, shuffle=True
# )


if config.load_model is None:
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)




# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    if num_workers == 0:
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

    dataset = SeqDataset(data_pt)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader



######################################################################
# Load the model
######################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ntokens = len(vocab)  # size of vocabulary

print("\n\n **** load model parameters ****")
print(f'lr = {lr}, batch_size = {batch_size}, epochs = {epochs}')
print(f'ntokens = {ntokens}, layer_size = embsize: {embsize} = d_hid: {d_hid}, n_layers: {nlayers}, nhead: {nhead}')
print("**** load model parameters ****\n")

model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=3,
    n_cls=n_cls,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=False,
    do_dab=False,
    use_batch_labels=False,
    num_batch_labels=False,
    domain_spec_batchnorm=False,
    input_emb_style=input_emb_style,
    n_input_bins=n_input_bins,
    cell_emb_style=cell_emb_style,
    explicit_zero_prob=False,
    use_fast_transformer=use_fast_transformer,
    fast_transformer_backend=fast_transformer_backend,
    pre_norm=config.pre_norm,
)
if config.load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file))
        logger.info(f"Loading ALL model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        logger.info(f"Loading SOME model params from {model_file}")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        # for k, v in pretrained_dict.items():
        #     logger.info(f"Loading params {k} with shape {v.shape}")

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

# 初始化模型之后，修改参数的冻结状态
for name, param in model.named_parameters():
    param.requires_grad = True

print("-"*20)
print("-"*20)


model.to(device)
print(model)
wandb.watch(model)


######################################################################
# Loss function
######################################################################

criterion_cls = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
scaler = torch.cuda.amp.GradScaler(enabled=True)




all_counts = (
    adata_test.layers[input_layer_key].A
    if issparse(adata_test.layers[input_layer_key])
    else adata_test.layers[input_layer_key]
)

# print(adata.layers[input_layer_key])

ageGroup_labels = adata_test.obs["ageGroup_id"].tolist()
ageGroup_labels = np.array(ageGroup_labels)


tokenized_test = tokenize_and_pad_batch(
    all_counts,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=False,
)

tensor_age_test = torch.from_numpy(ageGroup_labels).long()


test_data_pt = {
    "gene_ids": tokenized_test["genes"],
    "values": tokenized_test["values"],
    "ageGroup": tensor_age_test,
}

test_loader = DataLoader(
    dataset=SeqDataset(test_data_pt),
    batch_size=eval_batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=min(len(os.sched_getaffinity(0)), eval_batch_size // 2),
    pin_memory=True,
)



def test(model: nn.Module):
    

    model.eval()

    total_loss = 0.0
    total_num = 0

    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            ageGroup = batch_data["ageGroup"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    CLS=True, 
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=False,
                    #generative_training = False,
                )
                    
                loss = criterion_cls(output_dict["classified_output"], ageGroup)
            
            total_loss += loss.item() * len(input_gene_ids)
            total_num += len(input_gene_ids)
                   
            preds = output_dict["classified_output"].argmax(dim=1).cpu().numpy()
            labels = ageGroup.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
    
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1_weight = f1_score(all_labels, all_preds, average='weighted')
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)
   
    logger.info("-" * 89)
    logger.info("-" * 89)
    
    logger.info(
    f"| test | total_num {total_num} | precision {precision:5.4f} | " 
    f" recall {recall:5.4f} | f1_weight {f1_weight:5.4f} | f1_micro {f1_micro:5.4f} | f1_macro {f1_macro:5.4f} | " 
    f" accuracy {accuracy:5.4f}")
    logger.info("-" * 89)    
    
    

    # 保存数据到 CSV 文件
    all_preds_list = all_preds.tolist()
    all_labels_list = all_labels.tolist()

    all_preds_save_dir = str(save_dir) + "/all_preds.txt"
    all_labels_save_dir = str(save_dir) + "/all_targets.txt"

    with open(all_preds_save_dir, "w") as file:
        file.write(str(all_preds_list))

    with open(all_labels_save_dir, "w") as file:
        file.write(str(all_labels_list))

    all_preds_df = pd.DataFrame(all_preds_list, columns=['Predictions'])
    all_labels_df = pd.DataFrame(all_labels_list, columns=['Target'])

    combined_df = pd.concat([all_preds_df, all_labels_df], axis=1)
    combined_df['Predictions'] = combined_df['Predictions'].round(1)
    combined_df['Target'] = combined_df['Target'].round(1)

    combined_df.to_csv(str(save_dir) + "/output.csv", index=False)

    

# %%
test(model, adata_test)

# %%

df = pd.read_csv(str(save_dir) + "/output.csv")

y_true = df['Target']
y_pred = df['Predictions']


# 定义评估指标函数
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1_weight = f1_score(y_true, y_pred, average='weighted')
f1_micro = f1_score(y_true, y_pred, average='micro')
f1_macro = f1_score(y_true, y_pred, average='macro')
accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

# 输出结果
print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'f1_weight: {f1_weight}')
print(f'f1_micro: {f1_micro}')
print(f'f1_macro: {f1_macro}')
print(f'accuracy: {accuracy:.2f}')
print(f'conf_matrix: {conf_matrix}')

wandb.finish()

