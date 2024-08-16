import tempfile
import os

# 检查 tempfile 模块使用的临时文件目录
temp_dir = tempfile.gettempdir()
print("Updated temp directory:", temp_dir)

import sys
import importlib

new_path = '/data/mr423/project/code/'
if new_path not in sys.path:
    sys.path.insert(0, new_path)

# relaod the scgpt files
import scgpt
print(scgpt.__file__)
importlib.reload(scgpt)


import copy
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
import pandas as pd
# from . import asyn
import pickle
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from sklearn.metrics import confusion_matrix

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics

sc.set_figure_params(figsize=(6, 6))

os.environ["KMP_WARNINGS"] = "off"

set_seed(0)


######################################################################
# Settings for the model
######################################################################
eval_batch_size = 64
embsize = 512  # embedding dimension
d_hid = 512  # dimension of the feedforward network in TransformerEncoder
nlayers = 12  # number of TransformerEncoderLayer in TransformerEncoder
nhead = 8  # number of heads in nn.MultiheadAttention
dropout = 0  # dropout probability

save_dir = None

######################################################################
# Settings for input and preprocessing
######################################################################

pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]


include_zero_gene = False  # if True, include zero genes among hvgs in the training
max_seq_len = 3001
n_bins = 51

# input/output representation
input_style = "binned"  # "normed_raw", "log1p", or "binned"                           
# output_style = "normed_raw"  # "normed_raw", "log1p", or "binned"

######################################################################
# Settings for training
######################################################################
MLM = False  # whether to use masked language modeling, currently it is always on.
CLS = True  # celltype classification objective
ADV = False  # Adversarial training for batch correction
CCE = False  # Contrastive cell embedding objective
MVC = False  # Masked value prediction for cell embedding
ECS = False  # Elastic cell similarity objective
DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
input_emb_style = "category"  # "category" or "continuous" or "scaling"
cell_emb_style = "w-pool"  # "avg-pool" or "w-pool" or "cls"
adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"
ecs_threshold = 0.0

explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training

per_seq_batch_sample = False

fast_transformer = True
fast_transformer_backend = "flash"  # "linear" or "flash"


# Settings for the logging
do_eval_scib_metrics = True


# %% validate settings
assert input_style in ["normed_raw", "log1p", "binned"]
# assert output_style in ["normed_raw", "log1p", "binned"]
assert input_emb_style in ["category", "continuous", "scaling"]

if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins  # for padding gene expr values
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins


######################################################################
# Data loading
######################################################################
adata = sc.read("/data/mr423/project/data/3-OLINK_data_sub_train.h5ad")
adata_test = sc.read("/data/mr423/project/data/3-OLINK_data_sub_test.h5ad")


adata.obs["batch_id"]  = adata.obs["str_batch"] = "0"
adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "1" 

adata.var.set_index(adata.var["gene_name"], inplace=True)
adata_test.var.set_index(adata.var["gene_name"], inplace=True)

data_is_raw = False
filter_gene_by_counts = False
adata_test_raw = adata_test.copy()
adata = adata.concatenate(adata_test, batch_key="str_batch")

# make the batch category column
batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels

num_types = 1
# num_types = len(np.unique(celltype_id_labels))
# id2type = dict(enumerate(adata.obs["Age_Group"].astype("category").cat.categories))
print(num_types)

# adata.obs["celltype_id"] = celltype_id_labels
adata.var["gene_name"] = adata.var.index.tolist()


######################################################################
# The pre-trained model
######################################################################

model_file = save_dir + "/model.pt"
vocab_file = model_dir + "/vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    shutil.copy(vocab_file, save_dir / "vocab.json")
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    
    print(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]


    print("**** actual model parameters ****")
    print(f'layer_size = embsize: {embsize} = d_hid: {d_hid}, n_layers: {nlayers}, nhead: {nhead}')
    print("**** actual model parameters ****\n")


######################################################################
# set up the preprocessor, use the args to config the workflow
######################################################################
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=filter_gene_by_counts,  # step 1
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

print("input_layer_key: ", input_layer_key)

all_counts = (
    adata.layers[input_layer_key].A
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)


genes = adata.var["gene_name"].tolist()

age = adata.obs["age"].tolist()
age = np.array(age)
# print(age)

batch_ids = adata.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
batch_ids = np.array(batch_ids)

(
    train_data,
    valid_data,
    train_age,
    valid_age,
) = train_test_split(
    all_counts, age, test_size=0.1, shuffle=True
)


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
    intra_domain_shuffle: bool = False,
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

ntokens = len(vocab)  # size of gene vocabulary
model = TransformerModel(
    ntokens,    # size of gene vocabulary
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=3,
    n_cls=num_types if CLS else 1,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=MVC,
    do_dab=DAB,
    use_batch_labels=INPUT_BATCH_LABELS,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=config.DSBN,
    input_emb_style=input_emb_style,
    n_input_bins=n_input_bins,
    cell_emb_style=cell_emb_style,
    mvc_decoder_style=mvc_decoder_style,
    ecs_threshold=ecs_threshold,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=fast_transformer,
    fast_transformer_backend=fast_transformer_backend,
    pre_norm=config.pre_norm,
)
if config.load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file))
        print(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        # for k, v in pretrained_dict.items():
            # logger.info(f"Loading params {k} with shape {v.shape}")
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

# Freeze all pre-decoder weights
for name, para in model.named_parameters():
    # print("-"*20)
    # print(f"name: {name}")
    # if config.freeze and "encoder" in name and "transformer_encoder" not in name:
    if config.freeze and "encoder" in name:
        # print(f"freezing weights for: {name}")
        para.requires_grad = False

post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

print(f"Total Pre freeze Params {(pre_freeze_param_count )}")
print(f"Total Post freeze Params {(post_freeze_param_count )}")


model.to(device)

print(model)

criterion_cls = nn.SmoothL1Loss()

######################################################################
# Evaluate the model
######################################################################
def evaluate(model: nn.Module, loader: DataLoader, return_raw: bool = False) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_num = 0

    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            age = batch_data["age"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=True):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    # batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                    batch_labels=None,
                    CLS=CLS,  # evaluation does not need CLS or CCE
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=do_sample_in_train,
                    #generative_training = False,
                )
                
                output_values = output_dict["reg_output"]
                output_values = output_values.squeeze()

                # print("output : ",output_values.size())
                # print("ground : ",age.size())

                loss = criterion_cls(output_values, age)

            total_loss += loss.item() * len(input_gene_ids)     
            total_num += len(input_gene_ids)

            
            # 保存预测值和真实值以计算其他评估指标
            all_preds.append(output_values.cpu())
            all_targets.append(age.cpu())

    # 将所有批次的预测值和真实值连接起来
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    with open("all_preds.txt", "w") as file:
        file.write(str(all_preds))
        
    with open("all_targets.txt", "w") as file:
        file.write(str(all_targets))
    
    # print('all_preds :', all_preds)
    # print('all_targets :', all_targets)
    

    # 定义评估指标函数
    # 计算 MSE
    mse = torch.mean((all_preds - all_targets) ** 2)

    # 计算 MAE
    mae = torch.mean(torch.abs(all_preds - all_targets))

    # 计算 RMSE
    rmse = torch.sqrt(mse)

    # 计算 R2
    ss_res = torch.sum((all_targets - all_preds) ** 2)
    ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # 计算 MAPE
    mape = torch.mean(torch.abs((all_targets - all_preds) / all_targets)) * 100

    val_loss = total_loss / total_num

    print(
    f"| validation  | loss {val_loss:5.4f} | mse {mse:5.4f} | " 
    f" mae {mae:5.4f} | rmse {rmse:5.4f} | " 
    f" r2 {r2:5.4f} | mape {mape:5.4f}")
    print("-" * 89)




# %% inference
def test(model: nn.Module, adata: DataLoader) -> float:
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    age = adata.obs["age"].tolist()
    age = np.array(age)


    tokenized_test = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=include_zero_gene,
    )

    tensor_age_test = torch.from_numpy(age).float()


    test_data_pt = {
        "gene_ids": tokenized_test["genes"],
        "values": tokenized_test["values"],
        "age": tensor_age_test,
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
    evaluate(model, loader=test_loader, return_raw=True,)
