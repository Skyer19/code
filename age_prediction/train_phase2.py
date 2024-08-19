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

# reload the scgpt files
import scgpt
print("scgpt location: ", scgpt.__file__)
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

# os.environ["KMP_WARNINGS"] = "off"

warnings.filterwarnings('ignore')

######################################################################
# Load the saved configurations from phase 1
######################################################################

# Load the configuration from wandb or set it up again
hyperparameter_defaults = dict(
    seed=0,
    do_train=True,
    load_model="/data/mr423/project/pre_trained_model/scGPT_human",  # This is the pre-trained model directory
    n_bins=101,

    ecs_thres=0.0,
    dab_weight=0.0,
    
    epochs=100,  # You can modify the number of epochs for the second phase if needed
    lr=0.001,
    batch_size=128,

    layer_size=128,
    nlayers=4,
    nhead=8,
    
    dropout=0.0,
    schedule_ratio=0.9,
    save_eval_interval=5,
    use_fast_transformer=True,
    pre_norm=False,
    amp=True,
    include_zero_gene=False,
    freeze=False,  # Set to False to allow training of all model parameters
    DSBN=False,
)

run = wandb.init(
    config=hyperparameter_defaults,
    project="age_pred_phase2",  # You can change the project name to distinguish from phase 1
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
)
config = wandb.config
print(config)

set_seed(config.seed)

######################################################################
# Load data, vocab, and model from phase 1
######################################################################
# Paths to the saved model and vocab from phase 1
model_phase1_file = "/data/mr423/project/code/save/saved_model_phase1.pt"  # replace with your actual path
vocab_file = "/data/mr423/project/code/save/biobank-your_phase1_timestamp/vocab.json"  # replace with actual path

# Load vocab
vocab = GeneVocab.from_file(vocab_file)
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)

# Load data
adata = sc.read("/data/mr423/project/data/3-OLINK_data_sub_train_new.h5ad")
adata_test = sc.read("/data/mr423/project/data/3-OLINK_data_sub_test_new.h5ad")

# Preprocess and split the data
preprocessor = Preprocessor(
    use_key="X",
    filter_gene_by_counts=False,
    filter_cell_by_counts=False,
    normalize_total=3000,
    result_normed_key="X_normed",
    log1p=False,
    result_log1p_key="X_log1p",
    subset_hvg=False,
    hvg_flavor="cell_ranger",
    binning=config.n_bins,
    result_binned_key="X_binned",
)

adata_test = adata[adata.obs["str_batch"] == "1"]
adata = adata[adata.obs["str_batch"] == "0"]

preprocessor(adata, batch_key=None)
preprocessor(adata_test, batch_key=None)

# Define model parameters based on phase 1
input_style = "binned"
max_seq_len = 3001

# Load the model
ntokens = len(vocab)
embsize = config.layer_size
d_hid = config.layer_size
nlayers = config.nlayers
nhead = config.nhead
dropout = config.dropout

model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=3,
    n_cls=1,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=config.n_bins,
    do_mvc=False,
    do_dab=config.DAB,
    use_batch_labels=config.INPUT_BATCH_LABELS,
    domain_spec_batchnorm=config.DSBN,
    input_emb_style="category",
    n_input_bins=config.n_bins + 2,
    cell_emb_style="w-pool",
    mvc_decoder_style="inner product",

    use_fast_transformer=config.use_fast_transformer,
    fast_transformer_backend="flash",
    pre_norm=config.pre_norm,
)

# Load the weights from phase 1
model.load_state_dict(torch.load(model_phase1_file))

# Unfreeze all parameters for fine-tuning
for name, param in model.named_parameters():
    param.requires_grad = True
    print(f"Unfreezing {name}")

model.to("cuda" if torch.cuda.is_available() else "cpu")

# Re-initialize the optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, eps=1e-8)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

criterion = nn.MSELoss()

# Data preparation functions can be reused from phase 1

# Tokenize the data
tokenized_train = tokenize_and_pad_batch(
    adata.layers["X_binned"].A if issparse(adata.layers["X_binned"]) else adata.layers["X_binned"],
    np.array(vocab(adata.var["gene_name"].tolist()), dtype=int),
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=config.n_bins,
    append_cls=True,
    include_zero_gene=config.include_zero_gene,
)

tokenized_valid = tokenize_and_pad_batch(
    adata_test.layers["X_binned"].A if issparse(adata_test.layers["X_binned"]) else adata_test.layers["X_binned"],
    np.array(vocab(adata_test.var["gene_name"].tolist()), dtype=int),
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=config.n_bins,
    append_cls=True,
    include_zero_gene=config.include_zero_gene,
)

train_data_pt = {
    "gene_ids": tokenized_train["genes"],
    "values": tokenized_train["values"],
    "age": torch.from_numpy(np.array(adata.obs["age"].tolist())).float(),
}

valid_data_pt = {
    "gene_ids": tokenized_valid["genes"],
    "values": tokenized_valid["values"],
    "age": torch.from_numpy(np.array(adata_test.obs["age"].tolist())).float(),
}

train_loader = prepare_dataloader(train_data_pt, batch_size=config.batch_size)
valid_loader = prepare_dataloader(valid_data_pt, batch_size=config.batch_size)

# Training function is reused from phase 1
def train(model: nn.Module, loader: DataLoader) -> None:
    model.train()
    total_loss = 0
    total_num = 0
    all_preds = []
    all_targets = []
    
    for batch_data in loader:
        input_gene_ids = batch_data["gene_ids"].to("cuda" if torch.cuda.is_available() else "cpu")
        input_values = batch_data["values"].to("cuda" if torch.cuda.is_available() else "cpu")
        age = batch_data["age"].to("cuda" if torch.cuda.is_available() else "cpu")

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
            )
            
            output_values = output_dict["reg_output"].squeeze()
            loss = criterion(output_values, age)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(input_gene_ids)
        total_num += len(input_gene_ids)
        all_preds.append(output_values.cpu())
        all_targets.append(age.cpu())

    epoch_loss = total_loss / total_num
    mse = torch.mean((torch.cat(all_preds) - torch.cat(all_targets)) ** 2)
    mae = torch.mean(torch.abs(torch.cat(all_preds) - torch.cat(all_targets)))
    rmse = torch.sqrt(mse)
    ss_res = torch.sum((torch.cat(all_targets) - torch.cat(all_preds)) ** 2)
    ss_tot = torch.sum((torch.cat(all_targets) - torch.mean(torch.cat(all_targets))) ** 2)
    r2 = 1 - ss_res / ss_tot
    mape = torch.mean(torch.abs((torch.cat(all_targets) - torch.cat(all_preds)) / torch.cat(all_targets))) * 100

    wandb.log({
        "train/loss": epoch_loss,
        "train/mse": mse,
        "train/mae": mae,
        "train/rmse": rmse,
        "train/r2": r2,
        "train/mape": mape,
        "train/r2/mae": r2/mae,
    })

# Evaluation function is reused from phase 1
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    total_num = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to("cuda" if torch.cuda.is_available() else "cpu")
            input_values = batch_data["values"].to("cuda" if torch.cuda.is_available() else "cpu")
            age = batch_data["age"].to("cuda" if torch.cuda.is_available() else "cpu")

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
                )
                
                output_values = output_dict["reg_output"].squeeze()
                loss = criterion(output_values, age)

                total_loss += loss.item() * len(input_gene_ids)     
                total_num += len(input_gene_ids)
                all_preds.append(output_values.cpu())
                all_targets.append(age.cpu())

    val_loss = total_loss / total_num
    mse = torch.mean((torch.cat(all_preds) - torch.cat(all_targets)) ** 2)
    mae = torch.mean(torch.abs(torch.cat(all_preds) - torch.cat(all_targets)))
    rmse = torch.sqrt(mse)
    ss_res = torch.sum((torch.cat(all_targets) - torch.cat(all_preds)) ** 2)
    ss_tot = torch.sum((torch.cat(all_targets) - torch.mean(torch.cat(all_targets))) ** 2)
    r2 = 1 - ss_res / ss_tot
    mape = torch.mean(torch.abs((torch.cat(all_targets) - torch.cat(all_preds)) / torch.cat(all_targets))) * 100

    wandb.log({
        "valid/loss": val_loss,
        "valid/mse": mse,
        "valid/mae": mae,
        "valid/rmse": rmse,
        "valid/r2": r2,
        "valid/mape": mape,
        "valid/r2/mae": r2/mae,
    })

    scheduler.step(val_loss)
    return val_loss

######################################################################
# Train and evaluate the model in phase 2
######################################################################

best_val_loss = float("inf")
best_model = None
patience = 0

for epoch in range(1, config.epochs + 1):
    train(model, train_loader)
    val_loss = evaluate(model, valid_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        patience = 0
    else:
        patience += 1
        if patience >= 10:
            print(f"Early stopping at epoch {epoch}")
            break

# Save the best model
torch.save(best_model.state_dict(), save_dir / "best_model_phase2.pt")
