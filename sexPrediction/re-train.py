import tempfile
import os

# 检查 tempfile 模块使用的临时文件目录
temp_dir = tempfile.gettempdir()
print("Updated temp directory:", temp_dir)

import sys
import importlib

# new_path = '../sexPrediction/'
# if new_path not in sys.path:
#     sys.path.insert(0, new_path)

print(sys.path)

# relaod the scgpt files
import scgpt
print("scgpt location: ", scgpt.__file__)
importlib.reload(scgpt)



import copy
import gc
import json

import tempfile
import os
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

# os.environ["WANDB_MODE"]= "offline"



######################################################################
# Settings for wandb mentior
######################################################################

hyperparameter_defaults = dict(
    seed=0,
    do_train=True,
    load_model="../../pre_trained_model/scGPT_human",
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
    project="gender_pred",
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
dataset_name = 'biobank-gender_pred'
save_dir = Path(f"./record/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)

print(f"save to {save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")



######################################################################
# Data loading
######################################################################
adata = sc.read("../../data/3-OLINK_data_train_withOutlier_all.h5ad")
adata_test = sc.read("../../data/3-OLINK_data_test_withOutlier_all.h5ad")

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


gender_id_labels = adata.obs["sex"].astype("category").cat.codes.values
# ageGroup_types = adata.obs["Age_Group"].unique()
adata.obs["gender_id"] = gender_id_labels


n_cls = len(np.unique(gender_id_labels))

id2type = dict(enumerate(adata.obs["sex"].astype("category").cat.categories))
print(id2type)

adata.var["gene_name"] = adata.var.index.tolist()


######################################################################
# The pre-trained model
######################################################################
if config.load_model is not None:
    model_dir = config.load_model
    # model_config_file = model_dir + "/args.json"
    # model_file = model_dir + "/best_model.pt"
    # vocab_file = model_dir + "/vocab.json"

    model_file = "./trained_model/model.pt"
    vocab_file = "./trained_model/vocab.json"    

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

    # # model
    # with open(model_config_file, "r") as f:
    #     model_configs = json.load(f)
    # logger.info(
    #     f"Resume model from {model_file}, the model args will override the "
    #     f"config {model_config_file}."
    # )
    # embsize = model_configs["embsize"]
    # nhead = model_configs["nheads"]
    # d_hid = model_configs["d_hid"]
    # nlayers = model_configs["nlayers"]
    # n_layers_cls = model_configs["n_layers_cls"]

    # print("\n**** parameters from the pre-trained model ****")
    # print(f'layer_size = embsize: {model_configs["embsize"]} = d_hid: {model_configs["d_hid"]}, n_layers: {model_configs["nlayers"]}, nhead: {model_configs["nheads"]}')
    # print("**** parameters from the pre-trained model ****\n")

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

preprocessor(adata, batch_key=None)
preprocessor(adata_test, batch_key=None)



######################################################################
# Split the data to train and test
######################################################################
input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
    "normed_raw": "X_normed",
    "log1p": "X_normed",
    "binned": "X_binned",
}[input_style]
all_counts = (
    adata.layers[input_layer_key].A
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
genes = adata.var["gene_name"].tolist()

gender_labels = adata.obs["gender_id"].tolist()  # make sure count from 0
gender_labels = np.array(gender_labels)


(
    train_data,
    valid_data,
    train_gender,
    valid_gender,
) = train_test_split(
    all_counts, gender_labels, test_size=0.2, shuffle=True
)


if config.load_model is None:
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)



######################################################################
# Tokenize the data
######################################################################
tokenized_train = tokenize_and_pad_batch(
    train_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=False,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=False,
)
logger.info(
    f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
)
logger.info(
    f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
)


def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )

    input_values_train, input_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )

    tensor_gender_train = torch.from_numpy(train_gender).long()
    tensor_gender_valid = torch.from_numpy(valid_gender).long()


    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "gender": tensor_gender_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "gender": tensor_gender_valid,
    }

    return train_data_pt, valid_data_pt


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

pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

print("-"*20)
print("-"*20)

# 修改修改修改修改修改修改修改修改修改修改修改修改修改修改修改修改

# 列出要解冻的层的名称
layers_to_unfreeze = [
    # "transformer_encoder.layers.0.self_attn.Wqkv.weight",
    # "transformer_encoder.layers.0.self_attn.Wqkv.bias",
    # "transformer_encoder.layers.0.self_attn.out_proj.weight",
    # "transformer_encoder.layers.0.self_attn.out_proj.bias",
    # "transformer_encoder.layers.0.linear1.weight",
    # "transformer_encoder.layers.0.linear1.bias",
    # "transformer_encoder.layers.0.linear2.weight",
    # "transformer_encoder.layers.0.linear2.bias",
    # "transformer_encoder.layers.0.norm1.weight",
    # "transformer_encoder.layers.0.norm1.bias",
    # "transformer_encoder.layers.0.norm2.weight",
    # "transformer_encoder.layers.0.norm2.bias",
    # "transformer_encoder.layers.1.self_attn.Wqkv.weight",
    # "transformer_encoder.layers.1.self_attn.Wqkv.bias",
    # "transformer_encoder.layers.1.self_attn.out_proj.weight",
    # "transformer_encoder.layers.1.self_attn.out_proj.bias",
    # "transformer_encoder.layers.1.linear1.weight",
    # "transformer_encoder.layers.1.linear1.bias",
    # "transformer_encoder.layers.1.linear2.weight",
    # "transformer_encoder.layers.1.linear2.bias",
    # "transformer_encoder.layers.1.norm1.weight",
    # "transformer_encoder.layers.1.norm1.bias",
    # "transformer_encoder.layers.1.norm2.weight",
    # "transformer_encoder.layers.1.norm2.bias",
    # "transformer_encoder.layers.2.self_attn.Wqkv.weight",
    # "transformer_encoder.layers.2.self_attn.Wqkv.bias",
    # "transformer_encoder.layers.2.self_attn.out_proj.weight",
    # "transformer_encoder.layers.2.self_attn.out_proj.bias",
    # "transformer_encoder.layers.2.linear1.weight",
    # "transformer_encoder.layers.2.linear1.bias",
    # "transformer_encoder.layers.2.linear2.weight",
    # "transformer_encoder.layers.2.linear2.bias",
    # "transformer_encoder.layers.2.norm1.weight",
    # "transformer_encoder.layers.2.norm1.bias",
    # "transformer_encoder.layers.2.norm2.weight",
    # "transformer_encoder.layers.2.norm2.bias",
    # "transformer_encoder.layers.3.self_attn.Wqkv.weight",
    # "transformer_encoder.layers.3.self_attn.Wqkv.bias",
    # "transformer_encoder.layers.3.self_attn.out_proj.weight",
    # "transformer_encoder.layers.3.self_attn.out_proj.bias",
    # "transformer_encoder.layers.3.linear1.weight",
    # "transformer_encoder.layers.3.linear1.bias",
    # "transformer_encoder.layers.3.linear2.weight",
    # "transformer_encoder.layers.3.linear2.bias",
    # "transformer_encoder.layers.3.norm1.weight",
    # "transformer_encoder.layers.3.norm1.bias",
    # "transformer_encoder.layers.3.norm2.weight",
    # "transformer_encoder.layers.3.norm2.bias",
    "transformer_encoder.layers.4.self_attn.Wqkv.weight",
    "transformer_encoder.layers.4.self_attn.Wqkv.bias",
    "transformer_encoder.layers.4.self_attn.out_proj.weight",
    "transformer_encoder.layers.4.self_attn.out_proj.bias",
    "transformer_encoder.layers.4.linear1.weight",
    "transformer_encoder.layers.4.linear1.bias",
    "transformer_encoder.layers.4.linear2.weight",
    "transformer_encoder.layers.4.linear2.bias",
    "transformer_encoder.layers.4.norm1.weight",
    "transformer_encoder.layers.4.norm1.bias",
    "transformer_encoder.layers.4.norm2.weight",
    "transformer_encoder.layers.4.norm2.bias",
    "transformer_encoder.layers.5.self_attn.Wqkv.weight",
    "transformer_encoder.layers.5.self_attn.Wqkv.bias",
    "transformer_encoder.layers.5.self_attn.out_proj.weight",
    "transformer_encoder.layers.5.self_attn.out_proj.bias",
    "transformer_encoder.layers.5.linear1.weight",
    "transformer_encoder.layers.5.linear1.bias",
    "transformer_encoder.layers.5.linear2.weight",
    "transformer_encoder.layers.5.linear2.bias",
    "transformer_encoder.layers.5.norm1.weight",
    "transformer_encoder.layers.5.norm1.bias",
    "transformer_encoder.layers.5.norm2.weight",
    "transformer_encoder.layers.5.norm2.bias",
    "transformer_encoder.layers.6.self_attn.Wqkv.weight",
    "transformer_encoder.layers.6.self_attn.Wqkv.bias",
    "transformer_encoder.layers.6.self_attn.out_proj.weight",
    "transformer_encoder.layers.6.self_attn.out_proj.bias",
    "transformer_encoder.layers.6.linear1.weight",
    "transformer_encoder.layers.6.linear1.bias",
    "transformer_encoder.layers.6.linear2.weight",
    "transformer_encoder.layers.6.linear2.bias",
    "transformer_encoder.layers.6.norm1.weight",
    "transformer_encoder.layers.6.norm1.bias",
    "transformer_encoder.layers.6.norm2.weight",
    "transformer_encoder.layers.6.norm2.bias",
    "transformer_encoder.layers.7.self_attn.Wqkv.weight",
    "transformer_encoder.layers.7.self_attn.Wqkv.bias",
    "transformer_encoder.layers.7.self_attn.out_proj.weight",
    "transformer_encoder.layers.7.self_attn.out_proj.bias",
    "transformer_encoder.layers.7.linear1.weight",
    "transformer_encoder.layers.7.linear1.bias",
    "transformer_encoder.layers.7.linear2.weight",
    "transformer_encoder.layers.7.linear2.bias",
    "transformer_encoder.layers.7.norm1.weight",
    "transformer_encoder.layers.7.norm1.bias",
    "transformer_encoder.layers.7.norm2.weight",
    "transformer_encoder.layers.7.norm2.bias",
    # "transformer_encoder.layers.8.self_attn.Wqkv.weight",
    # "transformer_encoder.layers.8.self_attn.Wqkv.bias",
    # "transformer_encoder.layers.8.self_attn.out_proj.weight",
    # "transformer_encoder.layers.8.self_attn.out_proj.bias",
    # "transformer_encoder.layers.8.linear1.weight",
    # "transformer_encoder.layers.8.linear1.bias",
    # "transformer_encoder.layers.8.linear2.weight",
    # "transformer_encoder.layers.8.linear2.bias",
    # "transformer_encoder.layers.8.norm1.weight",
    # "transformer_encoder.layers.8.norm1.bias",
    # "transformer_encoder.layers.8.norm2.weight",
    # "transformer_encoder.layers.8.norm2.bias",
    # "transformer_encoder.layers.9.self_attn.Wqkv.weight",
    # "transformer_encoder.layers.9.self_attn.Wqkv.bias",
    # "transformer_encoder.layers.9.self_attn.out_proj.weight",
    # "transformer_encoder.layers.9.self_attn.out_proj.bias",
    # "transformer_encoder.layers.9.linear1.weight",
    # "transformer_encoder.layers.9.linear1.bias",
    # "transformer_encoder.layers.9.linear2.weight",
    # "transformer_encoder.layers.9.linear2.bias",
    # "transformer_encoder.layers.9.norm1.weight",
    # "transformer_encoder.layers.9.norm1.bias",
    # "transformer_encoder.layers.9.norm2.weight",
    # "transformer_encoder.layers.9.norm2.bias",
    # "transformer_encoder.layers.10.self_attn.Wqkv.weight",
    # "transformer_encoder.layers.10.self_attn.Wqkv.bias",
    # "transformer_encoder.layers.10.self_attn.out_proj.weight",
    # "transformer_encoder.layers.10.self_attn.out_proj.bias",
    # "transformer_encoder.layers.10.linear1.weight",
    # "transformer_encoder.layers.10.linear1.bias",
    # "transformer_encoder.layers.10.linear2.weight",
    # "transformer_encoder.layers.10.linear2.bias",
    # "transformer_encoder.layers.10.norm1.weight",
    # "transformer_encoder.layers.10.norm1.bias",
    # "transformer_encoder.layers.10.norm2.weight",
    # "transformer_encoder.layers.10.norm2.bias",
    # "transformer_encoder.layers.11.self_attn.Wqkv.weight",
    # "transformer_encoder.layers.11.self_attn.Wqkv.bias",
    # "transformer_encoder.layers.11.self_attn.out_proj.weight",
    # "transformer_encoder.layers.11.self_attn.out_proj.bias",
    # "transformer_encoder.layers.11.linear1.weight",
    # "transformer_encoder.layers.11.linear1.bias",
    # "transformer_encoder.layers.11.linear2.weight",
    # "transformer_encoder.layers.11.linear2.bias",
    # "transformer_encoder.layers.11.norm1.weight",
    # "transformer_encoder.layers.11.norm1.bias",
    # "transformer_encoder.layers.11.norm2.weight",
    # "transformer_encoder.layers.11.norm2.bias",
    "linear.weight",
    "linear.bias",
    "reg_decoder.fc1.weight",
    "reg_decoder.fc1.bias",
    "reg_decoder.bn1.weight",
    "reg_decoder.bn1.bias",
    "reg_decoder.fc2.weight",
    "reg_decoder.fc2.bias",
    "reg_decoder.bn2.weight",
    "reg_decoder.bn2.bias",
    "reg_decoder.fc4.weight",
    "reg_decoder.fc4.bias"
]

for name, param in model.named_parameters():
    print(f"name: {name}")

print("-"*20)
print("-"*20)
print("\n")

# 初始化模型之后，修改参数的冻结状态
for name, param in model.named_parameters():
    # 默认情况下冻结所有参数
    param.requires_grad = False
    # 解冻指定的层
    if name in layers_to_unfreeze:
        param.requires_grad = True
        print(f"Unfreezing layer: {name}")




# # Freeze all pre-decoder weights
# for name, para in model.named_parameters():
#     # print("-"*20)
#     print(f"name: {name}")

#     # if config.freeze and "encoder" in name and "transformer_encoder" not in name:
#     if config.freeze and "encoder" in name:
#         print(f"freezing weights for: {name}")
#         para.requires_grad = False

post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

logger.info(f"Total Pre freeze Params {(pre_freeze_param_count )}")
logger.info(f"Total Post freeze Params {(post_freeze_param_count )}")

wandb.log(
        {
            "info/pre_freeze_param_count": pre_freeze_param_count,
            "info/post_freeze_param_count": post_freeze_param_count,
        },
)

model.to(device)
print(model)
wandb.watch(model)


######################################################################
# Loss function
######################################################################

criterion_cls = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(
#     model.parameters(), lr=lr, eps=1e-4)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma=0.9)
# scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
scaler = torch.cuda.amp.GradScaler(enabled=True)



######################################################################
# Train the model
######################################################################
def train(model: nn.Module, loader: DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()

    total_loss = 0
    total_num = 0
    
    all_preds = []
    all_labels = []


    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        gender = batch_data["gender"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
                CLS=CLS,
                CCE=False,
                MVC=False,
                ECS=False,
                do_sample=False,
                #generative_training=False
            )

            loss = 0.0
            loss = criterion_cls(output_dict["classified_output"], gender)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(input_gene_ids)
        total_num += len(input_gene_ids)

        preds = output_dict["classified_output"].argmax(dim=1).cpu().numpy()
        labels = gender.cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)

    epoch_loss = total_loss / total_num
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)    
    accuracy = accuracy_score(all_labels, all_preds)


    logger.info("-" * 89)
    logger.info(f"Epoch {epoch}/{epochs}")
    logger.info(
    f"| train | total_num {total_num} | epoch loss {epoch_loss:5.4f} | precision {precision:5.4f} | " 
    f" recall {recall:5.4f} | f1 {f1:5.4f} |" 
    f" accuracy {accuracy:5.4f}")

    wandb.log(
        {
            "train/loss": epoch_loss,
            "train/precision": precision,
            "train/recall": recall,
            "train/f1": f1,
            "train/accuracy": accuracy,
            "epoch": epoch,
        },
    )



######################################################################
# Evaluate the model
######################################################################
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()

    total_loss = 0.0
    total_num = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            gender = batch_data["gender"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    CLS=CLS,  # evaluation does not need CLS or CCE
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=False,
                    #generative_training = False,
                )

                loss = criterion_cls(output_dict["classified_output"], gender)
            
            total_loss += loss.item() * len(input_gene_ids)
            total_num += len(input_gene_ids)
            
            preds = output_dict["classified_output"].argmax(dim=1).cpu().numpy()
            labels = gender.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)


    # 定义评估指标函数
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)    
    accuracy = accuracy_score(all_labels, all_preds)
    
    val_loss = total_loss / total_num
    accuracy = accuracy_score(all_labels, all_preds)

     
    wandb.log(
        {
            "valid/loss": val_loss,
            "valid/precision": precision,
            "valid/recall": recall,
            "valid/f1": f1,
            "valid/accuracy": accuracy,
            "epoch": epoch,
        },
    )

    scheduler.step(val_loss)

    logger.info(
    f"| valid | total_num {total_num} | epoch loss {val_loss:5.4f} | precision {precision:5.4f} | " 
    f" recall {recall:5.4f} | f1 {f1:5.4f} | " 
    f" accuracy {accuracy:5.4f}")
    logger.info("-" * 89)


    return val_loss



######################################################################
# 训练和验证循环
######################################################################

best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None

logger.info("Apply the model on the age of the clusters \n")

for epoch in range(1, epochs + 1):

    train_data_pt, valid_data_pt = prepare_data()
    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
    )

    if config.do_train:
        train(model,loader=train_loader,)
   
    val_loss = evaluate(model,loader=valid_loader)

    # early stop
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        logger.info(f"Best model with score {best_val_loss:5.4f}")
        patience = 0
    else:
        patience += 1
        if patience >= early_stop:
            logger.info(f"Early stop at epoch {epoch}")
            break


# save the model into the save_dir
torch.save(best_model.state_dict(), save_dir / "model.pt")


######################################################################
# Test the model
######################################################################
def test(model: nn.Module, adata: DataLoader):
    
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    # print(adata.layers[input_layer_key])

    gender_labels = adata.obs["gender_id"].tolist()
    gender_labels = np.array(gender_labels)


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

    tensor_gender_test = torch.from_numpy(gender_labels).long()


    test_data_pt = {
        "gene_ids": tokenized_test["genes"],
        "values": tokenized_test["values"],
        "gender": tensor_gender_test,
    }

    test_loader = DataLoader(
        dataset=SeqDataset(test_data_pt),
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), eval_batch_size // 2),
        pin_memory=True,
    )


    model.eval()

    total_loss = 0.0
    total_num = 0

    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            gender = batch_data["gender"].to(device)

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
                    
                loss = criterion_cls(output_dict["classified_output"], gender)
            
            total_loss += loss.item() * len(input_gene_ids)
            total_num += len(input_gene_ids)
                   
            preds = output_dict["classified_output"].argmax(dim=1).cpu().numpy()
            labels = gender.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
    
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)    
    accuracy = accuracy_score(all_labels, all_preds)
   
    logger.info("-" * 89)
    logger.info("-" * 89)
    
    logger.info(
    f"| test | total_num {total_num} | precision {precision:5.4f} | " 
    f" recall {recall:5.4f} | f1 {f1:5.4f} | " 
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
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

# 输出结果
print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'f1: {f1}')
print(f'accuracy: {accuracy:.2f}')
print(f'conf_matrix: {conf_matrix}')



wandb.finish()

