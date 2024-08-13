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
warnings.filterwarnings('ignore')

######################################################################
# Settings for wandb mentior
######################################################################

hyperparameter_defaults = dict(
    seed=0,
    do_train=True,
    load_model="../../pre_trained_model/scGPT_blood",
    mask_ratio=0.0,
    epochs=500,
    n_bins=51,
    MVC=False, # Masked value prediction for cell embedding
    ecs_thres=0.0, # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=0.0,
    lr=0.001,
    batch_size=256,
    layer_size=128,
    nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead=4,  # number of heads in nn.MultiheadAttention
    dropout=0.0,  # dropout probability
    schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
    save_eval_interval=5,
    fast_transformer=True,
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision
    include_zero_gene = False,
    freeze = True, #freeze
    DSBN = False,  # Domain-spec batchnorm
)

run = wandb.init(
    config=hyperparameter_defaults,
    project="age_prediction-test",
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
mask_ratio = config.mask_ratio
mask_value = "auto"  # for masked values, now it should always be auto

include_zero_gene = config.include_zero_gene  # if True, include zero genes among hvgs in the training
max_seq_len = 3001
n_bins = config.n_bins

# input/output representation
input_style = "binned"  # "normed_raw", "log1p", or "binned"
output_style = "binned"  # "normed_raw", "log1p", or "binned"

######################################################################
# Settings for training
######################################################################
MLM = False  # whether to use masked language modeling, currently it is always on.
CLS = True  # celltype classification objective
ADV = False  # Adversarial training for batch correction
CCE = False  # Contrastive cell embedding objective
MVC = config.MVC  # Masked value prediction for cell embedding
ECS = config.ecs_thres > 0  # Elastic cell similarity objective
DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
cell_emb_style = "avg-pool"  # "avg-pool" or "w-pool" or "cls"
adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"
ecs_threshold = config.ecs_thres
dab_weight = config.dab_weight

explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training

per_seq_batch_sample = False

######################################################################
# Settings for optimizer
######################################################################
lr = config.lr  # TODO: test learning rate ratio between two tasks
lr_ADV = 1e-3  # learning rate for discriminator, used when ADV is True
batch_size = config.batch_size
eval_batch_size = config.batch_size
epochs = config.epochs
schedule_interval = 1


######################################################################
# Settings for the model
######################################################################
fast_transformer = config.fast_transformer
fast_transformer_backend = "flash"  # "linear" or "flash"
embsize = config.layer_size  # embedding dimension
d_hid = config.layer_size  # dimension of the feedforward network in TransformerEncoder
nlayers = config.nlayers  # number of TransformerEncoderLayer in TransformerEncoder
nhead = config.nhead  # number of heads in nn.MultiheadAttention
dropout = config.dropout  # dropout probability

######################################################################
# Settings for the logging
######################################################################
log_interval = 100  # iterations
save_eval_interval = config.save_eval_interval  # epochs
do_eval_scib_metrics = True


# %% validate settings
assert input_style in ["normed_raw", "log1p", "binned"]
assert output_style in ["normed_raw", "log1p", "binned"]
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

if ADV and DAB:
    raise ValueError("ADV and DAB cannot be both True.")
DAB_separate_optim = True if DAB > 1 else False


######################################################################
# Settings for the running recording
######################################################################
dataset_name = 'biobank'
save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")


######################################################################
# Data loading
######################################################################
adata = sc.read("../../data/3-OLINK_data_sub_train.h5ad")
adata_test = sc.read("../../data/3-OLINK_data_sub_test.h5ad")

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

# celltype_id_labels = adata.obs["Age_Group"].astype("category").cat.codes.values
# celltypes = adata.obs["Age_Group"].unique()


num_types = 1
# num_types = len(np.unique(celltype_id_labels))
# id2type = dict(enumerate(adata.obs["Age_Group"].astype("category").cat.categories))
print(num_types)

# adata.obs["celltype_id"] = celltype_id_labels
adata.var["gene_name"] = adata.var.index.tolist()


######################################################################
# The pre-trained model
######################################################################
if config.load_model is not None:
    model_dir = config.load_model
    model_config_file = model_dir + "/args.json"
    model_file = model_dir + "/best_model.pt"
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
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]


######################################################################
# set up the preprocessor, use the args to config the workflow
######################################################################
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=filter_gene_by_counts,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
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
    include_zero_gene=include_zero_gene,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=include_zero_gene,
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

    tensor_age_train = torch.from_numpy(train_age).float()
    tensor_age_valid = torch.from_numpy(valid_age).float()

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "age": tensor_age_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "age": tensor_age_valid,
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

ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(
    ntokens,
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
        logger.info(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
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
# from torch.optim.lr_scheduler import ReduceLROnPlateau

criterion_cls = nn.MSELoss()
# criterion_cls = NonNegativeMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

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
    
    start_time = time.time()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        age = batch_data["age"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        
        optimizer.zero_grad()  # Clear previous gradients
        
        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                # batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                batch_labels=None,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
                do_sample=do_sample_in_train,
                #generative_training=False
            )
            
            loss = 0.0
            metrics_to_log = {}
                
            # loss = criterion_cls(apply_sigmoid(output_dict["cls_output"]), age)
            output_values = output_dict["reg_output"]
            loss = criterion_cls(output_values, age)

            # print("output : ",output_values)
            # print("ground : ",age)
            
            # print("output after apply_sigmoid: ",apply_sigmoid(output_dict["cls_output"]))

                  
            # print("train total_loss: ",loss)
           
            # metrics_to_log.update({"train/cls": loss.item()})

        loss.backward()
        optimizer.step()
        
        # wandb.log(metrics_to_log)
        # total_loss += loss.item()
        total_loss += loss.item() * len(input_gene_ids)
        total_num += len(input_gene_ids)
        
     

        if batch % log_interval == 0 and batch > 0:
                        
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
 
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {optimizer.param_groups[0]['lr']:05.4f} | ms/batch {ms_per_batch:5.4f} | "
                f"loss {cur_loss:5.2f} | ")
            
            start_time = time.time()
    epoch_loss = total_loss / total_num
    # logger.info(f"Epoch {epoch}/{epochs}, Epoch Loss: {epoch_loss:.4f}")
    print(f"Epoch {epoch}/{epochs}, Epoch Loss: {epoch_loss:.4f}")   
    metrics_to_log.update({"train/loss": epoch_loss})
    wandb.log(metrics_to_log)


######################################################################
# 定义评估指标函数
######################################################################
def mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred)).item()

def mean_squared_error(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2).item()

def root_mean_squared_error(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()

def r_squared(y_true, y_pred):
    y_mean = torch.mean(y_true)
    total_variance = torch.sum((y_true - y_mean) ** 2)
    residual_variance = torch.sum((y_true - y_pred) ** 2)
    return 1 - (residual_variance / total_variance).item()

def mean_absolute_percentage_error(y_true, y_pred):
    return torch.mean(torch.abs((y_true - y_pred) / y_true)).item() * 100

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
            with torch.cuda.amp.autocast(enabled=config.amp):
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
                loss = criterion_cls(output_values, age)

                # print("evaluate loss: ",loss)

            total_loss += loss.item() * len(input_gene_ids)     
            total_num += len(input_gene_ids)

            
            # 保存预测值和真实值以计算其他评估指标
            all_preds.append(output_values.cpu())
            all_targets.append(age.cpu())

    # 将所有批次的预测值和真实值连接起来
    all_preds = torch.cat(all_preds)

    # print('all_preds :', all_preds)
    all_targets = torch.cat(all_targets)
    # print('all_targets :', all_targets)
    
    # 计算其他评估指标
    total_mae = mean_absolute_error(all_targets, all_preds)
    total_rmse = root_mean_squared_error(all_targets, all_preds)
    total_r2 = r_squared(all_targets, all_preds)
    total_mape = mean_absolute_percentage_error(all_targets, all_preds)

    wandb.log(
        {
            "valid/mse": total_loss / total_num,
            "valid/mae": total_mae,
            "valid/rmse": total_rmse,
            "valid/r2": total_r2,
            "valid/mape": total_mape,
            "epoch": epoch,
        },
    )
    
    return total_loss / total_num, total_mae, total_rmse, total_r2, total_mape


    # wandb.log(
    #     {
    #         "valid/mse": total_loss / total_num,
    #         "epoch": epoch,
    #     },
    # )


    # return total_loss / total_num


######################################################################
# 训练和验证循环
######################################################################

best_val_loss = float("inf")
best_model = None


logger.info("Apply the model on the age of the prediction \n")

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)

    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=batch_size,
        shuffle=True,
        intra_domain_shuffle=True,
        drop_last=False,
    )

    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=eval_batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )


    if config.do_train:
        train(model,loader=train_loader,)


    val_loss, total_mae, total_rmse, total_r2, total_mape = evaluate(model,loader=valid_loader)
    
    # scheduler.step(val_loss)  # 更新学习率调度器

    
    elapsed = time.time() - epoch_start_time
    
    logger.info("-" * 89)
    logger.info(
    f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | mse {val_loss:5.4f} | " 
    f" mae {total_mae:5.4f} | rmse {total_rmse:5.4f} | " 
    f" r2 {total_r2:5.4f} | mape {total_mape:5.4f}")
    logger.info("-" * 89)

    scheduler.step()


    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        logger.info(f"Best model with score {best_val_loss:5.4f}")
    

# save the model into the save_dir
# torch.save(best_model.state_dict(), save_dir / "model.pt")
