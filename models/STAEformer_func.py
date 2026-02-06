'dataset and data processing utilities'

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class StandardScaler:
    def __init__(self, mean, std, eps=1e-6):
        self.mean = float(mean)
        self.std = float(std)
        self.eps = eps
    def transform(self, x):
        return (x - self.mean) / (self.std + self.eps)
    def inverse_transform(self, x):
        return x * (self.std + self.eps) + self.mean

def hf_dataset_to_stae_array(hf_ds, tod_idx, dow_idx, steps_per_day, normalize_tod=True, dow_zero_based=True):
    """
    Build data: (T_total, N, 3) == [value, tod, dow]
    """
    num_nodes = len(hf_ds)
    row0 = hf_ds[0]

    target0 = np.asarray(row0["target"], dtype=np.float32)
    T_total = target0.shape[0]

    dyn0 = np.asarray(row0["feat_dynamic_real"], dtype=np.float32)  # (F_dyn, T_total)
    assert dyn0.shape[1] == T_total

    tod = dyn0[tod_idx].astype(np.float32)  # (T_total,)
    dow = dyn0[dow_idx].astype(np.float32)  # (T_total,)

    # Optional normalization / shifting if needed
    if normalize_tod:
        # if tod is integer bins 0..steps_per_day-1, normalize to [0,1)
        tod = tod / float(steps_per_day)

    if not dow_zero_based:
        # if dow is 1..7, shift to 0..6
        dow = dow - 1.0

    data = np.zeros((T_total, num_nodes, 3), dtype=np.float32)

    for n in range(num_nodes):
        y = np.asarray(hf_ds[n]["target"], dtype=np.float32)
        assert y.shape[0] == T_total
        data[:, n, 0] = y
        data[:, n, 1] = tod
        data[:, n, 2] = dow

    item_ids = [hf_ds[i]["item_id"] for i in range(num_nodes)]
    return data, item_ids

def make_window_index(T_total, T_in, T_out, train_end, val_end=None):
    """
    Returns train_index, val_index, test_index where each row is [x_start, x_end, y_end].
    If val_end is None, returns only train_index and empty val/test.
    """
    last_start = T_total - (T_in + T_out)
    starts = np.arange(0, last_start + 1)

    x_end = starts + T_in
    y_end = x_end + T_out
    all_index = np.stack([starts, x_end, y_end], axis=1)

    if val_end is None:
        train_mask = y_end <= train_end
        return all_index[train_mask], np.zeros((0,3), dtype=np.int64), np.zeros((0,3), dtype=np.int64)

    train_mask = y_end <= train_end
    val_mask   = (y_end > train_end) & (y_end <= val_end)
    test_mask  = y_end > val_end
    return all_index[train_mask], all_index[val_mask], all_index[test_mask]

class STAEWindowDataset(Dataset):
    def __init__(self, data, index, scaler=None):
        self.data = data
        self.index = index.astype(np.int64)
        self.scaler = scaler

    def __len__(self):
        return self.index.shape[0]

    def __getitem__(self, i):
        x_start, x_end, y_end = self.index[i]
        x = self.data[x_start:x_end]        # (T_in, N, 3)
        y = self.data[x_end:y_end, :, :1]   # (T_out, N, 1)

        if self.scaler is not None:
            x = x.copy()
            x[..., 0] = self.scaler.transform(x[..., 0])

        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

def fit_scaler_on_train_x(data, train_index):
    # match authors: scaler is fit on x_train[...,0]
    xs = []
    for x_start, x_end, _ in train_index:
        xs.append(data[x_start:x_end, :, 0])  # (T_in, N)
    xs = np.concatenate(xs, axis=0)  # (num_samples*T_in, N)
    return StandardScaler(xs.mean(), xs.std())

'layers and model class of STAEformer'

import torch.nn as nn
import torch

class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out
    
class STAEformer(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]

        # ---- TOD ----
        if self.tod_embedding_dim > 0:
            tod_raw = tod
            if tod_raw.dtype.is_floating_point and tod_raw.min() >= 0 and tod_raw.max() <= 1.0:
                tod_idx = (tod_raw * self.steps_per_day).long()
            else:
                tod_idx = tod_raw.long()

            # minutes-of-day heuristic
            if tod_idx.max() >= self.steps_per_day and tod_idx.max() <= 2000:
                tod_idx = (tod_idx.float() / (1440.0 / self.steps_per_day)).long()

            tod_idx = tod_idx.clamp(0, self.steps_per_day - 1)
            tod_emb = self.tod_embedding(tod_idx)
            features.append(tod_emb)
        # ---- DOW ----
        if self.dow_embedding_dim > 0:
            dow_idx = dow.long()
            if dow_idx.min() >= 1 and dow_idx.max() <= 7:
                dow_idx = dow_idx - 1
            dow_idx = dow_idx.clamp(0, 6)
            dow_emb = self.dow_embedding(dow_idx)
            features.append(dow_emb)

        # if self.dow_embedding_dim > 0:
        #     dow_emb = self.dow_embedding(
        #         dow.long()
        #     )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
        #     features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
        # (batch_size, in_steps, num_nodes, model_dim)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out

import torch.nn.functional as F
from torch.distributions import NegativeBinomial

eps = 1e-4

def nb_nll_loss(raw_out, y_true):
    """
    raw_out: (B, T_out, N, 2)  raw model outputs
    y_true:  (B, T_out, N)     integer counts
    """
    raw_mu = raw_out[..., 0]
    raw_r  = raw_out[..., 1]

    # enforce positivity
    mu = F.softplus(raw_mu) + eps  # mean
    r  = F.softplus(raw_r)  + eps  # dispersion

    # convert mean+dispersion -> probs for PyTorch NB
    probs = r / (r + mu)

    dist = NegativeBinomial(total_count=r, probs=probs)

    # log_prob shape: (B, T_out, N)
    log_prob = dist.log_prob(y_true)

    # NLL: minus log-likelihood
    return -log_prob.mean()

class STAEformer_NB(nn.Module):
    def __init__(self, stae_config):
        super().__init__()
        # force output_dim = 2
        self.base = STAEformer(output_dim=2, **stae_config)
        self.eps = 1e-4

    def forward(self, x):
        """
        Returns (mu, r) for convenience.
        x: (B, in_steps, N, input_channels)
        """
        raw_out = self.base(x)   # (B, out_steps, N, 2)
        raw_mu = raw_out[..., 0]
        raw_r  = raw_out[..., 1]

        mu = F.softplus(raw_mu) + self.eps
        r  = F.softplus(raw_r)  + self.eps
        return mu, r

    def nll(self, x, y_true):
        """
        Directly compute NLL given inputs and ground truth.
        y_true: (B, out_steps, N)
        """
        mu, r = self.forward(x)
        probs = r / (r + mu)
        dist = NegativeBinomial(total_count=r, probs=probs)
        return -dist.log_prob(y_true).mean()
    
from pathlib import Path

def save_STAEformer_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None, config=None, scaler=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": config or {},
    }

    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if epoch is not None:
        ckpt["epoch"] = epoch

    # Save your StandardScaler as plain data
    if scaler is not None:
        scaler_payload = {}
        # common conventions:
        if hasattr(scaler, "mean"):
            scaler_payload["mean"] = scaler.mean
        if hasattr(scaler, "std"):
            scaler_payload["std"] = scaler.std
        if hasattr(scaler, "scale_"):
            scaler_payload["scale_"] = scaler.scale_
        if hasattr(scaler, "mean_"):
            scaler_payload["mean_"] = scaler.mean_
        if hasattr(scaler, "var_"):
            scaler_payload["var_"] = scaler.var_

        # store even if names differ; at least you will see what was saved
        ckpt["data_scaler"] = scaler_payload

    torch.save(ckpt, path)


def load_STAEformer_checkpoint(path, train_dataset, config,
                               device="cpu", optimizer=None, scheduler=None, strict=True):

    ckpt = torch.load(path, map_location=device,weights_only=False)

    # config = ckpt.get("config", {})
    # model = model_class(**config).to(device)
    lr_           = config["lr"]
    num_epochs_   = config["num_epochs"]
    weight_decay_ = config["weight_decay"]

    T_in  = config.get("context_length", config.get("input_length"))
    T_out = config.get("prediction_length")
    if T_in is None or T_out is None:
        raise ValueError("Provide config['context_length' or 'input_length'] and config['prediction_length'].")

    steps_per_day = config.get("steps_per_day", 96)
    batch_size    = config.get("batch_size", 64)

    tod_idx = config["tod_idx"]
    dow_idx = config["dow_idx"]

    # --- Build (T_total, N, 3) data array ---
    data, item_ids = hf_dataset_to_stae_array(
        train_dataset,
        tod_idx=tod_idx,
        dow_idx=dow_idx,
        steps_per_day=steps_per_day,
        normalize_tod=config.get("normalize_tod", True),
        dow_zero_based=config.get("dow_zero_based", True),
    )
    T_total, num_nodes, C = data.shape
    assert C == 3

    # --- STAEformer config: for (value,tod,dow) input, set input_dim=1 ---
    stae_config = dict(
        num_nodes=num_nodes,
        in_steps=T_in,
        out_steps=T_out,
        steps_per_day=steps_per_day,
        input_dim=1,  # value is channel 0; tod/dow read from channels 1 and 2
        input_embedding_dim=config.get("input_embedding_dim", 24),
        tod_embedding_dim=config.get("tod_embedding_dim", 24),
        dow_embedding_dim=config.get("dow_embedding_dim", 24),
        spatial_embedding_dim=config.get("spatial_embedding_dim", 0),
        adaptive_embedding_dim=config.get("adaptive_embedding_dim", 80),

        feed_forward_dim=config.get("feed_forward_dim", 256),
        num_heads=config.get("num_heads", 4),
        num_layers=config.get("num_layers", 3),
        dropout=config.get("dropout", 0.1),
        use_mixed_proj=config.get("use_mixed_proj", True),
    )

    model = STAEformer_NB(stae_config)

    model.load_state_dict(ckpt["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    epoch = ckpt.get("epoch", None)

    # Reconstruct StandardScaler (adjust to your class signature)
    scaler = None
    if "data_scaler" in ckpt and ckpt["data_scaler"]:
        sp = ckpt["data_scaler"]
        if "mean" in sp and "std" in sp:
            scaler = StandardScaler(sp["mean"], sp["std"])

    return model, epoch, scaler

def negative_binomial_nll(y, mu, theta, eps=1e-8, reduction="mean"):
    """
    y:      observed counts, (B, T_out, N, C)  (non-negative ints, but we'll cast to float)
    mu:     mean > 0,       same shape as y
    theta:  dispersion > 0, same shape as y

    Returns scalar if reduction="mean"/"sum", else elementwise NLL.
    """
    y = y.to(mu.dtype)

    # safety
    mu = mu.clamp(min=eps)
    theta = theta.clamp(min=eps)

    # log NB pmf
    # p = theta / (theta + mu)  (not needed explicitly)
    log_prob = (
        torch.lgamma(y + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y + 1.0)
        + theta * (torch.log(theta) - torch.log(theta + mu))
        + y * (torch.log(mu) - torch.log(theta + mu))
    )
    nll = -log_prob

    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    else:
        return nll  # no reduction

'training function for STAEformer_NB model'
from torch.optim import AdamW
import numpy as np
from tqdm.auto import tqdm

def train_STAEformer_from_hf_dataset(train_dataset, config, device, accelerator,
                                     training_=True, path_ =None):
    """
    Train STAEformer_NB directly from HF Dataset(rows=nodes).

    Required config keys:
      - lr, num_epochs, weight_decay
      - context_length (T_in) OR input_length
      - prediction_length (T_out)
      - steps_per_day
      - tod_idx, dow_idx  (column indices into feat_dynamic_real)
      - train_end (time index cutoff, exclusive)   OR train_ratio
    Optional:
      - val_end OR val_ratio
      - normalize_tod (bool)
      - dow_zero_based (bool)
      - batch_size
      - embedding/transformer hyperparams
    """

    lr_           = config["lr"]
    num_epochs_   = config["num_epochs"]
    weight_decay_ = config["weight_decay"]

    T_in  = config.get("context_length", config.get("input_length"))
    T_out = config.get("prediction_length")
    if T_in is None or T_out is None:
        raise ValueError("Provide config['context_length' or 'input_length'] and config['prediction_length'].")

    steps_per_day = config.get("steps_per_day", 96)
    batch_size    = config.get("batch_size", 64)

    tod_idx = config["tod_idx"]
    dow_idx = config["dow_idx"]

    # --- Build (T_total, N, 3) data array ---
    data, item_ids = hf_dataset_to_stae_array(
        train_dataset,
        tod_idx=tod_idx,
        dow_idx=dow_idx,
        steps_per_day=steps_per_day,
        normalize_tod=config.get("normalize_tod", True),
        dow_zero_based=config.get("dow_zero_based", True),
    )
    T_total, num_nodes, C = data.shape
    assert C == 3

    # --- Determine split cutoffs (train_end / val_end) ---
    train_end = None
    if "train_end" in config:
        train_end = int(config["train_end"])
    # else:
    #     train_ratio = float(config.get("train_ratio", 0.7))
    #     train_end = int(train_ratio * T_total)

    val_end = None
    if "val_end" in config:
        val_end = int(config["val_end"]) + train_end

    # --- Window indices ---
    train_index, val_index, test_index = make_window_index(
        T_total=T_total, T_in=T_in, T_out=T_out, train_end=train_end, val_end=val_end
    )

    if len(train_index) == 0:
        raise ValueError("No training windows created. Check T_in/T_out and train_end split.")

    # --- Fit scaler on training X target channel ---
    scaler = fit_scaler_on_train_x(data, train_index)

    # --- Build datasets/loaders ---
    train_ds = STAEWindowDataset(data, train_index, scaler=scaler)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # --- STAEformer config: for (value,tod,dow) input, set input_dim=1 ---
    stae_config = dict(
        num_nodes=num_nodes,
        in_steps=T_in,
        out_steps=T_out,
        steps_per_day=steps_per_day,
        input_dim=1,  # value is channel 0; tod/dow read from channels 1 and 2

        input_embedding_dim=config.get("input_embedding_dim", 24),
        tod_embedding_dim=config.get("tod_embedding_dim", 24),
        dow_embedding_dim=config.get("dow_embedding_dim", 24),
        spatial_embedding_dim=config.get("spatial_embedding_dim", 0),
        adaptive_embedding_dim=config.get("adaptive_embedding_dim", 80),

        feed_forward_dim=config.get("feed_forward_dim", 256),
        num_heads=config.get("num_heads", 4),
        num_layers=config.get("num_layers", 3),
        dropout=config.get("dropout", 0.1),
        use_mixed_proj=config.get("use_mixed_proj", True),
    )

    model = STAEformer_NB(stae_config).to(device)

    optimizer = AdamW(model.parameters(), lr=lr_, betas=(0.9, 0.95), weight_decay=weight_decay_)

    # --- accelerator prepare ---
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    model.train()
    loss_history, avg_loss_print = [], []

    for epoch in tqdm(range(num_epochs_)):
        for idx, (X, Y) in enumerate(train_loader):
            optimizer.zero_grad()

            # X: (B, T_in, N, 3), Y: (B, T_out, N, 1)
            mu, theta = model(X)  # (B, T_out, N, 1)

            loss = negative_binomial_nll(Y, mu, theta, reduction="mean")

            accelerator.backward(loss)
            optimizer.step()

            loss_history.append(loss.item())
            if idx % 100 == 0:
                avg_loss_print.append(np.mean(loss_history))
                loss_history = []

    if training_:
      save_STAEformer_checkpoint(
        f"{path_}/checkpoints/staeformer.pt",
        model,
        optimizer=optimizer,
        # scheduler=scheduler,
        epoch=epoch,
        config=config,
        scaler=scaler,
      )

    return model, avg_loss_print, scaler, item_ids

'inference function for STAEformer_NB model'

from torch.utils.data import DataLoader
from torch.distributions import NegativeBinomial
import torch

def infer_STAEformer_from_hf_dataset(
    model,
    dataset,
    config,
    device,
    scaler=None,
    n_samples=100,
    batch_size=128,
    split: str = "test",   # NEW: {"train","val","test","all"}
):
    """
    Inference for STAEformer_NB using an HF dataset.

    NEW:
      split controls which windows to run inference on:
        - "train": windows whose targets lie in the training region
        - "val":   windows whose targets lie in the validation region
        - "test":  windows whose targets lie in the test region (default; previous behavior)
        - "all":   all possible windows

    Returns dict with:
      mu:      (num_windows, T_out, N, 1)
      theta:   (num_windows, T_out, N, 1)
      y_true:  (num_windows, T_out, N, 1)
      samples: (num_windows, n_samples, T_out, N, 1)  if n_samples>0
      index:   list/array of windows [x_start, x_end, y_end]
      item_ids
      split
    """
    model.eval()

    T_in  = config.get("context_length", config.get("input_length"))
    T_out = config.get("prediction_length")
    if T_in is None or T_out is None:
        raise ValueError("Provide config['context_length' or 'input_length'] and config['prediction_length'].")

    steps_per_day = config.get("steps_per_day", 96)
    tod_idx = config["tod_idx"]
    dow_idx = config["dow_idx"]

    # Build data array (T_total, N, 3)
    data, item_ids = hf_dataset_to_stae_array(
        dataset,
        tod_idx=tod_idx,
        dow_idx=dow_idx,
        steps_per_day=steps_per_day,
        normalize_tod=config.get("normalize_tod", False),
        dow_zero_based=config.get("dow_zero_based", True),
    )
    T_total, N, C = data.shape
    if C != 3:
        raise ValueError(f"Expected 3 channels (value,tod,dow). Got C={C}.")

    # --- reconstruct split cutoffs exactly as you did in training ---
    # train_end = number of time steps in train region
    # val_end   = number of time steps in (train+val) region
    train_end = int(config["train_end"]) if "train_end" in config and config["train_end"] is not None else None

    val_end = None
    if "val_end" in config and config["val_end"] is not None:
        # Your original code assumes val_end is a length (not absolute) and adds train_end.
        if train_end is None:
            raise ValueError("config['val_end'] provided but config['train_end'] is missing; cannot reconstruct splits.")
        val_end = int(config["val_end"]) + train_end

    # Build window indices
    train_index, val_index, test_index = make_window_index(
        T_total=T_total, T_in=T_in, T_out=T_out, train_end=train_end, val_end=val_end
    )

    split = split.lower()
    if split == "train":
        use_index = train_index
    elif split == "val":
        use_index = val_index
    elif split == "test":
        use_index = test_index
    elif split == "all":
        # all possible windows: [x_start, x_end, y_end] for y_end in [T_in+T_out .. T_total]
        # x_end = y_end - T_out; x_start = x_end - T_in
        use_index = [(y_end - T_out - T_in, y_end - T_out, y_end) for y_end in range(T_in + T_out, T_total + 1)]
    else:
        raise ValueError("split must be one of {'train','val','test','all'}")

    if len(use_index) == 0:
        raise ValueError(f"No windows created for split='{split}'. Check cutoffs and T_in/T_out.")

    ds = STAEWindowDataset(data, use_index, scaler=scaler)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_mu, all_theta, all_y = [], [], []
    all_samples = [] if n_samples and n_samples > 0 else None

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)  # (B, T_in, N, 3)
            Y = Y.to(device)  # (B, T_out, N, 1)

            mu, theta = model(X)  # (B, T_out, N, 1)

            all_mu.append(mu.cpu())
            all_theta.append(theta.cpu())
            all_y.append(Y.cpu())

            if n_samples and n_samples > 0:
                mu_s = mu.squeeze(-1).clamp(min=1e-6)        # (B, T_out, N)
                th_s = theta.squeeze(-1).clamp(min=1e-6)     # (B, T_out, N)
                probs = th_s / (th_s + mu_s)                 # (B, T_out, N)

                dist = NegativeBinomial(total_count=th_s, probs=probs)
                s = dist.sample((n_samples,)).unsqueeze(-1)  # (n_samples, B, T_out, N, 1)
                s = s.permute(1, 0, 2, 3, 4).cpu()           # (B, n_samples, T_out, N, 1)
                all_samples.append(s)

    mu = torch.cat(all_mu, dim=0)         # (num_windows, T_out, N, 1)
    theta = torch.cat(all_theta, dim=0)   # (num_windows, T_out, N, 1)
    y_true = torch.cat(all_y, dim=0)      # (num_windows, T_out, N, 1)

    out = dict(
        mu=mu,
        theta=theta,
        y_true=y_true,
        index=use_index,
        item_ids=item_ids,
        split=split,
    )

    if n_samples and n_samples > 0:
        out["samples"] = torch.cat(all_samples, dim=0)  # (num_windows, n_samples, T_out, N, 1)

    return out
