import copy
import gc


import tempfile

# 检查 tempfile 模块使用的临时文件目录
temp_dir = tempfile.gettempdir()
print("Updated temp directory:", temp_dir)

import sys
import importlib

# new_path = '/data/mr423/project/code/maskedValuePrediction/'
# if new_path not in sys.path:
#     sys.path.insert(0, new_path)

# relaod the scgpt files
import scgpt
print("scgpt location: ", scgpt.__file__)
importlib.reload(scgpt)



import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings

import torch
from anndata import AnnData
import scanpy as sc
import scvi
import numpy as np
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

import transformers

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, eval_scib_metrics, load_pretrained

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

os.environ["WANDB_MODE"]= "offline"



######################################################################
# Settings for wandb mentior
######################################################################
hyperparameter_defaults = dict(
    seed=0,
    do_train=True, # Flag to indicate whether to do update model parameters during training
    load_model="/data/mr423/project/pre_trained_model/scGPT_human", # Path to pre-trained model
    GEPC=True,  # Gene expression modelling for cell objective

    mask_ratio=0.4, # Default mask ratio

    n_bins=101, # Default number of bins for value binning in data pre-processing
    epochs=30, # Default number of epochs for fine-tuning
    lr=1e-4, # Default learning rate for fine-tuning
    batch_size=64, # Default batch size for fine-tuning
    
    layer_size=128,
    nlayers=4,
    nhead=4, # if load model, batch_size, layer_size, nlayers, nhead will be ignored
    
    dropout=0.2, # Default dropout rate during model fine-tuning
    
    schedule_ratio=0.9,  # Default rate for learning rate decay

    # save_eval_interval=5, # Default model evaluation interval
    # log_interval=100, # Default log interval
    
    use_fast_transformer=True, # Default setting
    pre_norm=False, # Default setting
    amp=True,  # # Default setting: Automatic Mixed Precision
)
run = wandb.init(
    config=hyperparameter_defaults,
    project="masked_train-test",
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

mask_value = -1
pad_value = -2

# n_hvg = 1200  # number of highly variable genes
# max_seq_len = n_hvg + 1

max_seq_len = 3001
n_bins = config.n_bins

input_emb_style = "category"  # "category" or "continuous" or "scaling"
cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"

explicit_zero_prob = True  # whether explicit bernoulli for zeros


######################################################################
# Settings for optimizer
######################################################################
lr = config.lr
batch_size = config.batch_size
epochs = config.epochs
early_stop = 10



######################################################################
# Settings for the model
######################################################################
use_fast_transformer = config.use_fast_transformer

embsize = config.layer_size 
nhead = config.nhead
nlayers = config.nlayers  
d_hid = config.layer_size


######################################################################
# Validate the settings
######################################################################
# assert input_style in ["normed_raw", "log1p", "binned"]
# assert output_style in ["normed_raw", "log1p", "binned"]

assert input_emb_style in ["category", "continuous", "scaling"]


if input_emb_style == "category":
    mask_value = -1
    pad_value = -2  # for padding gene expr values
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins


######################################################################
# Settings for the running recording
######################################################################
dataset_name = 'masked-train'
save_dir = Path(f"/data/mr423/project/code/maskedValuePrediction/record/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")


######################################################################
# Data loading
######################################################################
# adata = scvi.data.pbmc_dataset()  # 11990 × 3346
# ori_batch_col = "batch"
# adata.obs["celltype"] = adata.obs["str_labels"].astype("category")
# adata.var = adata.var.set_index("gene_symbols")
# data_is_raw = True

# # make the batch category column
# adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
# batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
# adata.obs["batch_id"] = batch_id_labels
# adata.var["gene_name"] = adata.var.index.tolist()

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

adata.var["gene_name"] = adata.var.index.tolist()


######################################################################
# The pre-trained model
######################################################################
if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
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
        f"Resume model from {model_file}, the model args will be overriden by the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

    print("\n**** parameters from the pre-trained model ****")
    print(f'layer_size = embsize: {model_configs["embsize"]} = d_hid: {model_configs["d_hid"]}, n_layers: {model_configs["nlayers"]}, nhead: {model_configs["nheads"]}')
    print("**** parameters from the pre-trained model ****\n")

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
    log1p=False,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=config.n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)
# preprocessor(adata, batch_key="str_batch" if dataset_name != "heart_cell" else None)

adata_test = adata[adata.obs["str_batch"] == "1"]
adata = adata[adata.obs["str_batch"] == "0"]

preprocessor(adata, batch_key=None)
preprocessor(adata_test, batch_key=None)


######################################################################
# Split the data to train and test
######################################################################
input_layer_key = "X_binned"
all_counts = (
    adata.layers[input_layer_key].A
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
genes = adata.var["gene_name"].tolist()

(
    train_data, # gene level
    valid_data, # gene level
) = train_test_split(
    all_counts, test_size=0.2, shuffle=False
)

print("train_data: ",train_data)


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
    include_zero_gene=True,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=True,
)
logger.info(
    f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
)
logger.info(
    f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
)

print(tokenized_train)

'''
tokenized_train: tokenized_train['genes'], tokenized_train['values']
tokenized_valid: tokenized_valid['genes'], tokenized_valid['values']

['genes'] in id format
['values'] in bins format

'''

def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(
        f"random masking: ratio of masked values in train: ",
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
    )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )


    # if sort_seq_batch:
    #     train_sort_ids = np.argsort(train_batch_labels)
    #     input_gene_ids_train = input_gene_ids_train[train_sort_ids]
    #     input_values_train = input_values_train[train_sort_ids]
    #     target_values_train = target_values_train[train_sort_ids]
    #     tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]

    #     valid_sort_ids = np.argsort(valid_batch_labels)
    #     input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
    #     input_values_valid = input_values_valid[valid_sort_ids]
    #     target_values_valid = target_values_valid[valid_sort_ids]
    #     tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,  # masked value
        "target_values": target_values_train, # without masked value
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,  # masked value
        "target_values": target_values_valid,
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
    dataset = SeqDataset(data_pt)

    # if per_seq_batch_sample: ## True
    #     # find the indices of samples in each seq batch
    #     subsets = []
    #     batch_labels_array = data_pt["batch_labels"].numpy()
    #     for batch_label in np.unique(batch_labels_array):
    #         batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
    #         subsets.append(batch_indices)
    #     data_loader = DataLoader(
    #         dataset=dataset,
    #         batch_sampler=SubsetsBatchSampler(
    #             subsets,
    #             batch_size,
    #             intra_subset_shuffle=intra_domain_shuffle,
    #             inter_subset_shuffle=shuffle,
    #             drop_last=drop_last,
    #         ),
    #         num_workers=num_workers,
    #         pin_memory=True,
    #     )
    #     return data_loader

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
    vocab=vocab,
    dropout=config.dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=config.GEPC, # True
    # do_dab=True,
    do_dab=False,
    # use_batch_labels=True,
    # num_batch_labels=num_batch_types,
    use_batch_labels=False,
    num_batch_labels=None,
    # domain_spec_batchnorm=DSBN,
    domain_spec_batchnorm=False, # use batch norm, we can also choose not use the norm
    
    n_input_bins=n_input_bins,
    cell_emb_style=cell_emb_style,

    # ecs_threshold=config.ecs_thres, # 0.8
    explicit_zero_prob=explicit_zero_prob, # true
    use_fast_transformer=use_fast_transformer, # True
    pre_norm=config.pre_norm, # False
)
if config.load_model is not None:
    load_pretrained(model, torch.load(model_file), verbose=False)


param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
logger.info(f"Total freeze Params {(param_count )}")


model.to(device)
print(model)
wandb.watch(model)


train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=False)

train_loader = prepare_dataloader(
    train_data_pt,
    batch_size=batch_size,
    shuffle=False,
    intra_domain_shuffle=True,
    drop_last=False,
)
valid_loader = prepare_dataloader(
    valid_data_pt,
    batch_size=config.batch_size,
    shuffle=False,
    intra_domain_shuffle=False,
    drop_last=False,
)


######################################################################
# Loss function
######################################################################
criterion = masked_mse_loss
# criterion_dab = nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma=0.99)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

warmup_ratio_or_step = 0.1

# setup scheduler
if warmup_ratio_or_step > 0:
    total_num_batches = len(train_loader) * config.epochs
    warmup_steps = (
        int(total_num_batches * warmup_ratio_or_step)
        if warmup_ratio_or_step < 1
        else int(warmup_ratio_or_step)
    )
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_num_batches,
        last_epoch=-1,
    )
else:
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma=0.99)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

scaler = torch.cuda.amp.GradScaler(enabled=config.amp)


######################################################################
# Train the model
######################################################################
def train(model: nn.Module, loader: DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_num, total_mlm, total_gepc ,total_error = 0.0, 0.0, 0.0, 0.0, 0.0
    # log_interval = config.log_interval
    # start_time = time.time()

    # num_batches = len(loader)

    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)  # masked value
        target_values = batch_data["target_values"].to(device) # without masked value

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        
        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels = None,

                MVC=config.GEPC,
                # ECS=config.ecs_thres > 0,
                ECS = False,
                # do_sample 的作用是在模型预测时引入随机性，通过对模型输出的零概率部分进行采样
                do_sample = False,
            )

            masked_positions = input_values.eq(mask_value)  # the postions to predict
            

            # MLM: 使用 ExprDecoder， 直接提取输入嵌入中的核心信息
            loss = loss_mlm = criterion(
                output_dict["mlm_output"], target_values, masked_positions
            )
            # metrics_to_log = {"train/mse": loss_mlm.item()}

            if explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_zero_log_prob
                # metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})


            # GEPC: 使用 MVCDecoder， 能够捕捉细胞和基因嵌入之间的复杂交互信息
            if config.GEPC: 
                loss_gepc = criterion(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_gepc
                # metrics_to_log.update({"train/mvc": loss_gepc.item()})
            
            if config.GEPC and explicit_zero_prob:
                loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mvc_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_gepc_zero_log_prob
                # metrics_to_log.update(
                #     {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
                # )

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        # wandb.log(metrics_to_log)

        with torch.no_grad():
            # 用于计算模型预测值与实际目标值之间的相对误差，并且只在 mask 指定的部分进行计算
            mre = masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            )

        total_loss += loss.item()* len(input_gene_ids)
        total_num += len(input_gene_ids)

        total_mlm += loss_mlm.item()* len(input_gene_ids)
        total_gepc += loss_gepc.item()* len(input_gene_ids)
        total_error += mre.item()* len(input_gene_ids)

    epoch_loss = total_loss / total_num
    
    cur_mse = total_mlm / total_num
    cur_gepc = total_gepc / total_num
    cur_error = total_error / total_num

    logger.info(
        f"| epoch {epoch} | lr {lr:5.4f} | epoch_loss {epoch_loss:5.4f} | "
        f"mse {cur_mse:5.4f} |  gepc {cur_gepc:5.4f} |  error {cur_error:5.4f}  |"
    )

    wandb.log(
        {
            "train/loss": epoch_loss,
            "train/cur_mse": cur_mse,
            "train/cur_gepc": cur_gepc,
            "train/cur_error": cur_error,
            "epoch": epoch,
        },
    )



def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_num = 0
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    # batch_labels=batch_labels if DSBN else None,
                    batch_labels=None,

                )
                output_values = output_dict["mlm_output"]

                masked_positions = input_values.eq(mask_value)
                loss = criterion(output_values, target_values, masked_positions)

            total_loss += loss.item() * len(input_gene_ids)

            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item() * len(input_gene_ids)
            total_num += len(input_gene_ids)

    wandb.log(
        {
            "valid/mse": total_loss / total_num,
            "valid/mre": total_error / total_num,
            "epoch": epoch,
        },
    )

    return total_loss / total_num, total_error / total_num




'''
def eval_testdata(
    model: nn.Module,
    adata_t: AnnData,
    include_types: List[str] = ["cls"],
) -> Optional[Dict]:
    
    """evaluate the model on test dataset of adata_t"""
    model.eval()

    # copy adata_t to avoid reuse previously computed results stored in adata_t
    adata_t = adata_t.copy()

    all_counts = (
        adata_t.layers[input_layer_key].A
        if issparse(adata_t.layers[input_layer_key])
        else adata_t.layers[input_layer_key]
    )

    celltypes_labels = adata_t.obs["celltype"].tolist()
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata_t.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    # Evaluate cls cell embeddings
    if "cls" in include_types:
        logger.info("Evaluating cls cell embeddings")
        tokenized_all = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=max_seq_len,
            vocab=vocab,
            pad_token=pad_token,
            pad_value=pad_value,
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=True,
        )
        all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
        src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.amp):
            cell_embeddings = model.encode_batch(
                all_gene_ids,
                all_values.float(),
                src_key_padding_mask=src_key_padding_mask,
                batch_size=config.batch_size,
                # batch_labels=torch.from_numpy(batch_ids).long() if DSBN else None,
                batch_labels=None,

                time_step=0,
                return_np=True,
            )
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )

        adata_t.obsm["X_scGPT"] = cell_embeddings

        results = {}
        try:
            results = eval_scib_metrics(adata_t)
        except Exception as e:
            traceback.print_exc()
            logger.error(e)

        sc.pp.neighbors(adata_t, use_rep="X_scGPT")
        sc.tl.umap(adata_t, min_dist=0.3)
        fig = sc.pl.umap(
            adata_t,
            color=["str_batch"],
            title=[f"batch, avg_bio = {results.get('avg_bio', 0.0):.4f}"],
            frameon=False,
            return_fig=True,
            show=False,
        )

        results["batch_umap"] = fig

        sc.pp.neighbors(adata_t, use_rep="X_scGPT")
        sc.tl.umap(adata_t, min_dist=0.3)
        fig = sc.pl.umap(
            adata_t,
            color=["celltype"],
            title=[
                f"celltype, avg_bio = {results.get('avg_bio', 0.0):.4f}",
            ],
            frameon=False,
            return_fig=True,
            show=False,
        )

        results["celltype_umap"] = fig

    if len(include_types) == 1:
        return results
'''

best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None

for epoch in range(0, epochs):
# for epoch in range(1, 2):

    print(f"epoch: {epoch}")

    train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=False)

    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=batch_size,
        shuffle=False,
        intra_domain_shuffle=True,
        drop_last=False,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=config.batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )

    if config.do_train:
        train(model,loader=train_loader,)

    val_loss, val_mre = evaluate(model,loader=valid_loader,)
    scheduler.step(val_loss)


    logger.info("-" * 89)
    logger.info(
        f"| valid | epoch {epoch:3d} | valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
    )
    logger.info("-" * 89)

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

    # if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
    #     logger.info(f"Saving model to {save_dir}")
    #     torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")

    #     # eval on testdata
    #     results = eval_testdata(
    #         best_model,
    #         # adata_t=adata_sorted if per_seq_batch_sample else adata,
    #         adata_t= adata,
    #         include_types=["cls"],
    #     )
    #     results["batch_umap"].savefig(
    #         save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png", dpi=300
    #     )

    #     results["celltype_umap"].savefig(
    #         save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png", dpi=300
    #     )
    #     metrics_to_log = {"test/" + k: v for k, v in results.items()}
    #     metrics_to_log["test/batch_umap"] = wandb.Image(
    #         str(save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png"),
    #         caption=f"celltype avg_bio epoch {best_model_epoch}",
    #     )

    #     metrics_to_log["test/celltype_umap"] = wandb.Image(
    #         str(save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png"),
    #         caption=f"celltype avg_bio epoch {best_model_epoch}",
    #     )
    #     metrics_to_log["test/best_model_epoch"] = best_model_epoch
    #     wandb.log(metrics_to_log)
    #     wandb.log({"avg_bio": results.get("avg_bio", 0.0)})








