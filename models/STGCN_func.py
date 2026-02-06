'auxilary function for creating STGCN static graph'
# Based on among stations distances - Steps:
# * compute pairwise distance between all station pairs in meters
# * convert distances to similaries:

# > w_ij = 1/d_ij, if d_ij <= 1000m; otherwise w_ij=0

# * collecy the w_ij values into(NxN) adjacency matrix

import numpy as np
import torch

def pairwise_haversine(latlon):
    """
    latlon: array of shape (N, 2) with columns [lat, lon] in degrees.
    Returns:
        dist: (N, N) matrix of distances in meters.
    """
    lat = np.deg2rad(latlon[:, 0])[:, None]  # (N, 1)
    lon = np.deg2rad(latlon[:, 1])[:, None]  # (N, 1)

    dlat = lat - lat.T                        # (N, N)
    dlon = lon - lon.T                        # (N, N)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371000.0  # Earth radius in meters
    dist = R * c
    return dist


def build_distance_adjacency(latlon, threshold=1000.0, eps=1e-6):
    """
    dist: (N, N) numpy array of pairwise distances in meters.
    threshold: max distance to consider an edge.

    reference: following geography proximity defined in https://www.sciencedirect.com/science/article/pii/S0968090X22001668
    Joint demand prediction for multimodal systems: A multi-task multi-relational spatiotemporal graph neural network approach, 2022
    """
    dist = pairwise_haversine(latlon)  # (N, N)

    # mask for edges we want to keep: 0 < d <= threshold
    mask = (dist > 0) & (dist <= threshold)

    # compute sigma_d as std of these distances
    valid_dists = dist[mask]
    sigma_d = valid_dists.std() if valid_dists.size > 0 else 1.0
    sigma_d = max(sigma_d, eps)

    A = np.zeros_like(dist, dtype=np.float32)
    A[mask] = np.exp(- (dist[mask] / sigma_d) ** 2)

    # (optional) enforce symmetry
    A = 0.5 * (A + A.T)

    return A, sigma_d

# 'usage example'
# df_demand = pd.read_csv('/content/drive/MyDrive/Paper 1 Revision Runs/intermediate_15min_updated.csv')
# valid_station_lst = df_demand['station_id'].unique()
# df_loc= pd.read_csv('station_coordination.csv')
# # filter df_loc for start_station_id within valid_station_lst
# df_loc = df_loc[df_loc['start_station_id'].isin(valid_station_lst)]
# # reorder rows by ascending order of station
# df_loc = df_loc.sort_values(by=['start_station_id'])
# # coords: np.array of shape (N, 2) with [lat, lon]
# latlon = np.array(df_loc[['start_lat', 'start_lng']].values)
# # print(latlon[:5])
# A_np, sigma_d = build_distance_adjacency(latlon, threshold=1000.0)
# A is in the order of station_id, in ascending order

# # then either:
# A = torch.from_numpy(A_np)  # (N, N) torch tensor
# model.set_adj(A)
# # or pass A each forward:
# mu, theta = model(X, A)

'dataset and dataloader for STGCN model'

import numpy as np
import torch
from torch.utils.data import Dataset

# borrow TST dataloader
from .TST_func import *

class STGCNWindowDataset(Dataset):
    """
    Generic STGCN dataset over ALL stations, using precomputed:
      - targets:  (T_total, N)
      - feat_dyn: (T_total, F_dyn)
      - static_real: (N, F_s) or None
    and a list of window start indices.

    Each __getitem__ returns a dict with:
      past_values:          (T_in, N)
      future_values:        (T_out, N)
      past_time_features:   (T_in, F_dyn)
      future_time_features: (T_out, F_dyn)
      past_observed_mask:   (T_in, N)
      future_observed_mask: (T_out, N)
      static_real_features: (N, F_s)   [if available]
    """

    def __init__(
        self,
        targets: torch.Tensor,        # (T_total, N)
        feat_dyn: torch.Tensor,       # (T_total, F_dyn)
        static_real: torch.Tensor,    # (N, F_s) or None
        context_length: int,
        prediction_length: int,
        start_indices: list[int],
    ):
        super().__init__()
        self.targets = targets
        self.feat_dyn = feat_dyn
        self.static_real = static_real
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.start_indices = list(start_indices)

        self.T_total, self.N = self.targets.shape
        self.F_dyn = self.feat_dyn.shape[-1]
        self.has_static_real = static_real is not None

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        s = self.start_indices[idx]
        e_past = s + self.context_length
        e_future = e_past + self.prediction_length

        past_values = self.targets[s:e_past, :]          # (T_in, N)
        future_values = self.targets[e_past:e_future, :] # (T_out, N)

        past_time_features = self.feat_dyn[s:e_past, :]        # (T_in, F_dyn)
        future_time_features = self.feat_dyn[e_past:e_future, :]  # (T_out, F_dyn)

        past_observed_mask = torch.ones_like(past_values)
        future_observed_mask = torch.ones_like(future_values)

        sample = {
            "past_values": past_values,
            "future_values": future_values,
            "past_time_features": past_time_features,
            "future_time_features": future_time_features,
            "past_observed_mask": past_observed_mask,
            "future_observed_mask": future_observed_mask,
        }

        if self.has_static_real:
            sample["static_real_features"] = self.static_real  # (N, F_s)

        return sample


from typing import Tuple


def create_stgcn_backtest_datasets(
    ts_dataset,
    context_length: int,
    prediction_length: int,
    testing_size: int,
    training_size: int,
) -> Tuple[STGCNWindowDataset, STGCNWindowDataset]:
    """
    Create STGCN train & test datasets for backtesting from transformed data.

    ts_dataset: transformed_data (e.g. Cached(...) from create_transformation)
    context_length:     T_in
    prediction_length:  T_out
    testing_size:       number of *test* windows (out-of-sample)
    training_size:      number of *train/validation* windows (in-sample),
                        taken just before the test segment.

    Returns
    -------
    stgcn_test_dataset, stgcn_train_dataset
    """

    # -------- 1. Materialize and compute common T_total across stations --------
    ts_list = list(ts_dataset)
    N = len(ts_list)

    lengths = []
    for ex in ts_list:
        vals = np.asarray(ex["values"], dtype=np.float32)
        lengths.append(len(vals))

    T_total = min(lengths)

    if len(set(lengths)) > 1:
        print(
            "[create_stgcn_backtest_datasets] Warning: unequal series lengths.",
            "Using last T_total =", T_total, "time steps for all stations.",
        )

    # -------- 2. Build dynamic features (time_features) from series 0 --------
    ex0 = ts_list[0]
    feat_dyn0 = np.asarray(ex0["time_features"], dtype=np.float32)
    if feat_dyn0.ndim == 1:
        feat_dyn0 = feat_dyn0[None, :]

    if feat_dyn0.shape[0] <= feat_dyn0.shape[1]:
        # (F_dyn, T0) -> (T0, F_dyn)
        feat_dyn0 = feat_dyn0.T

    feat_dyn0 = feat_dyn0[-T_total:, :]          # (T_total, F_dyn)
    feat_dyn = torch.from_numpy(feat_dyn0)       # (T_total, F_dyn)
    F_dyn = feat_dyn.shape[-1]

    # -------- 3. Stack targets from all stations -> (T_total, N) --------
    targets_list = []
    for i, ex in enumerate(ts_list):
        ti = np.asarray(ex["values"], dtype=np.float32)
        ti = ti[-T_total:]                       # last T_total
        if len(ti) != T_total:
            raise RuntimeError(
                f"Station {i} has length {len(ti)} after trimming; expected {T_total}."
            )
        targets_list.append(ti)

    targets = torch.from_numpy(np.stack(targets_list, axis=1))  # (T_total, N)

    # -------- 4. Static real features per station (optional) --------
    has_static_real = "static_real_features" in ex0
    static_real = None
    if has_static_real:
        sr0 = np.asarray(ex0["static_real_features"], dtype=np.float32)  # (F_s,)
        F_static_real = sr0.shape[0]

        static_list = []
        for i, ex in enumerate(ts_list):
            s_i = np.asarray(ex["static_real_features"], dtype=np.float32)
            assert s_i.shape[0] == F_static_real, \
                "All stations must share same static_real_features length"
            static_list.append(s_i)
        static_real = torch.from_numpy(np.stack(static_list, axis=0))    # (N, F_s)

    # -------- 5. Compute all possible window start indices --------
    L = context_length + prediction_length
    max_num_windows = T_total - L + 1
    if max_num_windows <= 0:
        raise RuntimeError(
            f"Series too short for context_length={context_length} and "
            f"prediction_length={prediction_length} (T_total={T_total})."
        )

    # All possible starts with stride 1
    all_starts = list(range(max_num_windows))

    # -------- 6. Define test and train window start indices --------
    if testing_size > max_num_windows:
        raise ValueError(
            f"testing_size={testing_size} > max_num_windows={max_num_windows}"
        )

    test_starts = all_starts[-testing_size:]   # last testing_size windows

    # Training windows are taken from the *pre-test* region
    pre_test_starts = all_starts[:-testing_size]  # everything before the test segment
    if training_size > len(pre_test_starts):
        raise ValueError(
            f"training_size={training_size} > available pre-test windows={len(pre_test_starts)}"
        )

    train_starts = pre_test_starts[-training_size:]  # last training_size pre-test windows

    # -------- 7. Create two datasets (test & train) --------
    stgcn_test_dataset = STGCNWindowDataset(
        targets=targets,
        feat_dyn=feat_dyn,
        static_real=static_real,
        context_length=context_length,
        prediction_length=prediction_length,
        start_indices=test_starts,
    )

    stgcn_train_dataset = STGCNWindowDataset(
        targets=targets,
        feat_dyn=feat_dyn,
        static_real=static_real,
        context_length=context_length,
        prediction_length=prediction_length,
        start_indices=train_starts,
    )

    return stgcn_test_dataset, stgcn_train_dataset

import numpy as np
import torch
from torch.utils.data import Dataset


class STGCNTrainDataset(Dataset):
    """
    STGCN training dataset that returns windows over *all* stations,
    using transformed data (after create_transformation).

    Each element of ts_dataset is expected to contain at least:
      - "values":         (T_total,)
      - "time_features":  (num_time_features, T_total) or (T_total, num_time_features)
      - optionally "static_real_features": (F_static_real,)
    """

    def __init__(
        self,
        ts_dataset,
        context_length: int,
        prediction_length: int,
        num_samples_per_epoch: int,
    ):
        super().__init__()

        # Make iterable indexable
        ts_list = list(ts_dataset)
        self.ds = ts_list
        self.N = len(self.ds)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples_per_epoch

        # ---------- 1. Time length + dynamic features ----------

        ex0 = self.ds[0]

        target0 = np.asarray(ex0["values"], dtype=np.float32)  # (T_total,)
        self.T_total = len(target0)

        feat_dyn0 = np.asarray(ex0["time_features"], dtype=np.float32)
        # Normalize to (T_total, F_dyn)
        if feat_dyn0.ndim == 1:
            feat_dyn0 = feat_dyn0[None, :]
        if feat_dyn0.shape[0] <= feat_dyn0.shape[1]:
            # assume (F_dyn, T_total) -> (T_total, F_dyn)
            feat_dyn0 = feat_dyn0.T

        self.feat_dyn = torch.from_numpy(feat_dyn0)  # (T_total, F_dyn)
        self.F_dyn = self.feat_dyn.shape[-1]

        # ---------- 2. Stack targets: (T_total, N) ----------

        targets = []
        for i in range(self.N):
            ti = np.asarray(self.ds[i]["values"], dtype=np.float32)
            assert len(ti) == self.T_total, "All series must have same length"
            targets.append(ti)
        self.targets = torch.from_numpy(np.stack(targets, axis=1))  # (T_total, N)

        # ---------- 3. Static real features per station (optional) ----------

        self.has_static_real = "static_real_features" in ex0
        if self.has_static_real:
            sr0 = np.asarray(ex0["static_real_features"], dtype=np.float32)  # (F_s,)
            self.F_static_real = sr0.shape[0]

            static_list = []
            for i in range(self.N):
                s_i = np.asarray(self.ds[i]["static_real_features"], dtype=np.float32)
                assert s_i.shape[0] == self.F_static_real, \
                    "All stations must share same static_real_features length"
                static_list.append(s_i)
            # (N, F_s)
            self.static_real = torch.from_numpy(np.stack(static_list, axis=0))
        else:
            self.F_static_real = 0
            self.static_real = None

        # ---------- 4. Window start limit ----------

        self.max_start = self.T_total - (context_length + prediction_length)
        assert self.max_start > 0, (
            f"Time series too short for context_length={context_length} "
            f"+ prediction_length={prediction_length}"
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Sample random window
        s = np.random.randint(0, self.max_start + 1)
        e_past = s + self.context_length
        e_future = e_past + self.prediction_length

        past_values = self.targets[s:e_past, :]           # (T_in, N)
        future_values = self.targets[e_past:e_future, :]  # (T_out, N)

        past_time_features = self.feat_dyn[s:e_past, :]           # (T_in, F_dyn)
        future_time_features = self.feat_dyn[e_past:e_future, :]  # (T_out, F_dyn)

        past_observed_mask = torch.ones_like(past_values)
        future_observed_mask = torch.ones_like(future_values)

        sample = {
            "past_values": past_values,
            "future_values": future_values,
            "past_time_features": past_time_features,
            "future_time_features": future_time_features,
            "past_observed_mask": past_observed_mask,
            "future_observed_mask": future_observed_mask,
        }

        # Add static_real_features as (N, F_s); DataLoader will collate to (B, N, F_s)
        if self.has_static_real:
            sample["static_real_features"] = self.static_real

        return sample

def create_transformation_stgcn(freq: str, config: PretrainedConfig) -> Transformation:
    remove_field_names = []
    if config['num_static_real_features'] == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config['num_dynamic_real_features'] == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config['num_static_categorical_features'] == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)
    if len(remove_field_names) > 0:
      print('field name removed',remove_field_names)

    # a bit like torchvision.transforms.Compose
    return Chain(
        # step 1: remove static/dynamic fields if not specified
        [RemoveFields(field_names=remove_field_names)]
        # step 2: convert the data to NumPy (potentially not needed)
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                )
            ]
            if config['num_static_categorical_features'] > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                )
            ]
            if config['num_static_real_features'] > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.TARGET,
                # we expect an extra dim for the multivariate case:
                expected_ndim=1 if config['input_size'] == 1 else 2,
            ),

            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),

            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                # time_features=time_features_from_frequency_str(freq),
                time_features=time_features_from_frequency_str(freq)[1:3],
                pred_length=config['prediction_length'],
            ),

            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config['prediction_length'],
                log_scale=True,
            ),
            # step 6: vertically stack all the temporal features into the key FEAT_TIME
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    # [FieldName.FEAT_DYNAMIC_REAL_TEMP, FieldName.FEAT_DYNAMIC_REAL_PREP, FieldName.FEAT_DYNAMIC_REAL_WINDS]
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if config['num_dynamic_real_features'] > 0
                    else []
                ),
            ),
            # step 7: rename to match HuggingFace names
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )


from gluonts.dataset.common import Cached
from torch.utils.data import DataLoader

def create_stgcn_train_dataloader(
    config,
    freq,
    data,                     # original raw dataset
    batch_size: int,
    num_batches_per_epoch: int,
    cache_data: bool = True,
    **kwargs,
) -> DataLoader:
    """
    STGCN version of create_train_dataloader.

    - Uses create_transformation(freq, config) to add time features etc.
    - Builds one graph-style dataset over *all* stations.
    """

    # 1) Same transformation pipeline as transformer
    transformation = create_transformation_stgcn(freq, config)
    transformed_data = transformation.apply(data, is_train=True)

    if cache_data:
        transformed_data = Cached(transformed_data)

    # 2) Build STGCN dataset
    context_length = config['context_length']
    prediction_length = config['prediction_length']
    num_samples_per_epoch = batch_size * num_batches_per_epoch

    stgcn_dataset = STGCNTrainDataset(
        ts_dataset=transformed_data,
        context_length=context_length,
        prediction_length=prediction_length,
        num_samples_per_epoch=num_samples_per_epoch,
    )

    # 3) Wrap into PyTorch DataLoader
    train_dataloader = DataLoader(
        stgcn_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return train_dataloader


import numpy as np
import torch
from torch.utils.data import Dataset


class STGCNBacktestDataset(Dataset):
    def __init__(
        self,
        ts_dataset,
        context_length: int,
        prediction_length: int,
        testing_size: int,
    ):
        super().__init__()

        ts_list = list(ts_dataset)
        self.ds = ts_list
        self.N = len(self.ds)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.testing_size = testing_size

        # ---------- 1. Compute per-station lengths and choose common T_total ----------
        lengths = []
        for ex in self.ds:
            vals = np.asarray(ex["values"], dtype=np.float32)
            lengths.append(len(vals))

        # Common length = minimum across stations
        self.T_total = min(lengths)

        if len(set(lengths)) > 1:
            print(
                "[STGCNBacktestDataset] Warning: station series have unequal lengths.",
                "Using last T_total =", self.T_total, "time steps for all stations.",
            )

        # ---------- 2. Dynamic (time) features from series 0 ----------
        ex0 = self.ds[0]

        feat_dyn0 = np.asarray(ex0["time_features"], dtype=np.float32)
        if feat_dyn0.ndim == 1:
            feat_dyn0 = feat_dyn0[None, :]  # (1, T0)

        # normalize to (T0, F_dyn)
        if feat_dyn0.shape[0] <= feat_dyn0.shape[1]:
            # (F_dyn, T0) -> (T0, F_dyn)
            feat_dyn0 = feat_dyn0.T
        # else: already (T0, F_dyn)

        # keep ONLY the last T_total steps
        feat_dyn0 = feat_dyn0[-self.T_total:, :]      # (T_total, F_dyn)

        self.feat_dyn = torch.from_numpy(feat_dyn0)
        self.F_dyn = self.feat_dyn.shape[-1]

        # ---------- 3. Stack targets from all stations -> (T_total, N) ----------
        targets = []
        for i, ex in enumerate(self.ds):
            ti = np.asarray(ex["values"], dtype=np.float32)
            # take the LAST T_total entries to align with feat_dyn0
            ti = ti[-self.T_total:]
            if len(ti) != self.T_total:
                raise RuntimeError(
                    f"Station {i} still has mismatching length {len(ti)} after trimming."
                )
            targets.append(ti)

        self.targets = torch.from_numpy(np.stack(targets, axis=1))  # (T_total, N)

        # ---------- 4. Static real features per station (optional) ----------
        self.has_static_real = "static_real_features" in ex0
        if self.has_static_real:
            sr0 = np.asarray(ex0["static_real_features"], dtype=np.float32)  # (F_s,)
            self.F_static_real = sr0.shape[0]

            static_list = []
            for i, ex in enumerate(self.ds):
                s_i = np.asarray(ex["static_real_features"], dtype=np.float32)
                assert s_i.shape[0] == self.F_static_real, \
                    "All stations must share same static_real_features length"
                static_list.append(s_i)
            self.static_real = torch.from_numpy(np.stack(static_list, axis=0))  # (N, F_s)
        else:
            self.F_static_real = 0
            self.static_real = None

        # ---------- 5. Determine backtest window start indices ----------
        L = context_length + prediction_length
        max_num_windows = self.T_total - L + 1
        assert self.testing_size <= max_num_windows, (
            f"testing_size={self.testing_size} is too large for "
            f"T_total={self.T_total} and window length={L}"
        )

        self.start0 = self.T_total - L - (self.testing_size - 1)
        assert self.start0 >= 0
        self.start_indices = [self.start0 + i for i in range(self.testing_size)]

    def __len__(self):
        return self.testing_size

    def __getitem__(self, idx):
        s = self.start_indices[idx]
        e_past = s + self.context_length
        e_future = e_past + self.prediction_length

        past_values = self.targets[s:e_past, :]          # (T_in, N)
        future_values = self.targets[e_past:e_future, :] # (T_out, N)

        past_time_features = self.feat_dyn[s:e_past, :]       # (T_in, F_dyn)
        future_time_features = self.feat_dyn[e_past:e_future, :]  # (T_out, F_dyn)

        past_observed_mask = torch.ones_like(past_values)
        future_observed_mask = torch.ones_like(future_values)

        sample = {
            "past_values": past_values,
            "future_values": future_values,
            "past_time_features": past_time_features,
            "future_time_features": future_time_features,
            "past_observed_mask": past_observed_mask,
            "future_observed_mask": future_observed_mask,
        }

        if self.has_static_real:
            sample["static_real_features"] = self.static_real  # (N, F_s)

        return sample

from gluonts.dataset.common import Cached
from torch.utils.data import DataLoader

def create_stgcn_backtest_dataloader(
    config,
    freq,
    data,                     # raw dataset (same as transformer backtest)
    testing_size: int,
    training_size: int,
    batch_size: int,
    batch_size_train: int,
    cache_data: bool = True,
    **kwargs,
):
    """
    Backtest dataloader for STGCN.

    - Uses create_transformation_stgcn(freq, config) to get 'values', 'time_features',
      and static_real_features if configured.
    - Builds the last `testing_size` windows over ALL stations.
    - Returns a DataLoader yielding dicts compatible with your STGCN training/eval.
    """

    # 1) Same transformation pipeline as for transformer
    transformation = create_transformation_stgcn(freq, config)
    transformed_data = transformation.apply(data)

    if cache_data:
        transformed_data = Cached(transformed_data)

    # 2) Build backtest dataset
    context_length = config['context_length']
    prediction_length = config['prediction_length']

    stgcn_test_dataset, stgcn_train_dataset = create_stgcn_backtest_datasets(
    ts_dataset=transformed_data,
    context_length=context_length,
    prediction_length=prediction_length,
    testing_size=testing_size,
    training_size=training_size,
    )


    # 3) Wrap in a DataLoader
    test_data_loader = DataLoader(
        stgcn_test_dataset,
        batch_size=batch_size,
        shuffle=False,   # keep chronological order of backtest windows
        num_workers=2,
    )

    val_train_data_loader = DataLoader(
        stgcn_train_dataset,
        batch_size=batch_size_train,
        shuffle=False,
        num_workers=2
    )

    return test_data_loader, val_train_data_loader

'Layers and Model Class for STGCN'

'layers'

import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_adj(A: torch.Tensor) -> torch.Tensor:
    """
    Symmetric normalization: Â = D^(-1/2) (A + I) D^(-1/2)
    A: (N, N)
    """
    N = A.size(0)
    device = A.device
    A_hat = A + torch.eye(N, device=device)
    d = A_hat.sum(1)
    d_inv_sqrt = torch.pow(d + 1e-8, -0.5)
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt

class TemporalConvGLU(nn.Module):
    """
    Temporal convolution with GLU gating.
    Input:  x (B, C_in, N, T)
    Output: (B, C_out, N, T)   (time length preserved via padding)
    """
    def __init__(self, c_in, c_out, kernel_size=3):
        super().__init__()
        padding = kernel_size - 1
        self.conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=2 * c_out,
            kernel_size=(1, kernel_size),
            padding=(0, padding)
        )
        self.c_out = c_out

    def forward(self, x):
        # x: (B, C_in, N, T)
        conv_out = self.conv(x)  # (B, 2*C_out, N, T + padding)
        # keep only last T steps to preserve length
        T = x.size(-1)
        conv_out = conv_out[..., -T:]  # (B, 2*C_out, N, T)
        P, Q = torch.split(conv_out, self.c_out, dim=1)
        return P * torch.sigmoid(Q)  # GLU


class SpatialGCN(nn.Module):
    """
    Simple first-order GCN over nodes.
    Input:  x (B, C_in, N, T)
    Output: (B, C_out, N, T)
    """
    def __init__(self, c_in, c_out):
        super().__init__()
        self.theta = nn.Linear(c_in, c_out, bias=False)

    def forward(self, x, A_norm):
        # x: (B, C_in, N, T)
        B, C_in, N, T = x.shape

        # apply graph propagation for each time step
        # reshape to (B*T, N, C_in)
        x_t = x.permute(0, 3, 2, 1).contiguous().view(B * T, N, C_in)
        # graph diffusion: (B*T, N, C_in)
        x_diff = torch.einsum("nm, bmc -> bnc", A_norm, x_t)
        # linear transform on features
        x_out = self.theta(x_diff)  # (B*T, N, C_out)
        # back to (B, C_out, N, T)
        x_out = x_out.view(B, T, N, -1).permute(0, 3, 2, 1).contiguous()
        return x_out


class STConvBlock(nn.Module):
    """
    One Spatio-Temporal block:
    T-Conv(GLU) -> Spatial GCN -> T-Conv(GLU) -> Dropout + Residual + Norm
    """
    def __init__(self, c_in, c_hidden, c_out, kernel_size=3, dropout=0.3):
        super().__init__()
        self.temp1 = TemporalConvGLU(c_in, c_hidden, kernel_size)
        self.spat = SpatialGCN(c_hidden, c_hidden)
        self.temp2 = TemporalConvGLU(c_hidden, c_out, kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(c_out)

    def forward(self, x, A_norm):
        # x: (B, C_in, N, T)
        residual = x

        x = self.temp1(x)                    # (B, c_hidden, N, T)
        x = F.relu(self.spat(x, A_norm))     # (B, c_hidden, N, T)
        x = self.temp2(x)                    # (B, c_out, N, T)
        x = self.dropout(x)

        # project residual if needed
        if residual.size(1) != x.size(1):
            # 1x1 conv to match channels
            proj = nn.Conv2d(residual.size(1), x.size(1), kernel_size=1).to(x.device)
            residual = proj(residual)

        x = x + residual                     # residual connection
        # LayerNorm over channels & nodes; reshape to apply
        B, C, N, T = x.shape
        x = x.permute(0, 3, 2, 1).contiguous()  # (B, T, N, C)
        x = self.norm(x)
        x = x.permute(0, 3, 2, 1).contiguous()  # (B, C, N, T)
        return x

class STGCN_NB(nn.Module):
    """
    STGCN with Negative Binomial output head.

    Expects:
        X: (B, T_in, N, F_in)
    Returns:
        mu:    (B, T_out, N, nb_channels)
        theta: (B, T_out, N, nb_channels)   # dispersion > 0
    """
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        A: torch.Tensor = None,
        nb_channels: int = 1,        # how many NB outputs per node (typically 1)
        horizon: int = 1,
        eps: float = 1e-6,
        kernel_size: int = 24,  # how many timesteps before the model reads for prediction
        #  hyperparameters
        hidden_channels: int= 32, # 16, 32, 64
        num_blocks: int = 2, # 1,2,3
        dropout: float = 0.3, # 0.1, 0.2, 0.3
        # (optimizer hyperparam) - learning rate
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.horizon = horizon
        self.nb_channels = nb_channels
        self.eps = eps

        # same backbone as STGCN, but final channels = 2 * nb_channels
        self.input_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=(1, 1)
        )

        blocks = []
        c_in = hidden_channels
        for _ in range(num_blocks):
            blocks.append(
                STConvBlock(
                    c_in=c_in,
                    c_hidden=hidden_channels,
                    c_out=hidden_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )
            c_in = hidden_channels
        self.blocks = nn.ModuleList(blocks)

        # output projection: hidden -> 2 * nb_channels  (mu + theta)
        self.output_proj = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=2 * nb_channels,
            kernel_size=(1, 1)
        )

        if A is not None:
            self.register_buffer("A_norm", normalize_adj(A))
        else:
            self.A_norm = None

    def set_adj(self, A: torch.Tensor):
        self.A_norm = normalize_adj(A.to(next(self.parameters()).device))

    def forward(self, X, A: torch.Tensor = None):
        """
        X: (B, T_in, N, F_in)
        A: optional adjacency (N, N)
        Returns:
            mu:    (B, T_out, N, nb_channels)
            theta: (B, T_out, N, nb_channels)
        """
        # (B, F_in, N, T_in)
        X = X.permute(0, 3, 2, 1).contiguous()

        # adjacency
        if A is not None:
            A_norm = normalize_adj(A.to(X.device))
        else:
            if self.A_norm is None:
                raise ValueError("Adjacency matrix A not set.")
            A_norm = self.A_norm

        # backbone
        x = self.input_proj(X)  # (B, hidden, N, T)

        for block in self.blocks:
            x = block(x, A_norm)  # (B, hidden, N, T)

        x = self.output_proj(x)  # (B, 2*nb_channels, N, T)

        # choose last T_out steps
        T = x.size(-1)
        T_out = self.horizon
        if T_out <= T:
            x = x[..., -T_out:]  # (B, 2*nb_channels, N, T_out)
        else:
            pad = T_out - T
            x = F.pad(x, (pad, 0))

        # back to (B, T_out, N, 2*nb_channels)
        x = x.permute(0, 3, 2, 1).contiguous()

        # split into raw_mu and raw_theta
        raw_mu, raw_theta = torch.split(x, self.nb_channels, dim=-1)

        # make them positive
        mu = F.softplus(raw_mu) + self.eps      # mean
        theta = F.softplus(raw_theta) + self.eps  # dispersion (shape)

        return mu, theta

def build_XY_from_batch(batch, device, config):
  # ---------- 1. Extract original fields ----------
    # time-varying covariates & target
    past_time_features   = batch["past_time_features"].float().to(device)   # (B, T_in, F_t)
    future_time_features = batch["future_time_features"].float().to(device) # (B, T_out, F_t)
    past_values          = batch["past_values"].float().to(device)          # (B, T_in, N)
    future_values        = batch["future_values"].float().to(device)        # (B, T_out, N)

    B, T_in, N = past_values.shape
    # T_out = future_values.size(1)

    # static features (may be absent)
    if config['num_static_categorical_features'] > 0:
        static_cat = batch["static_categorical_features"].float().to(device)  # (B, F_sc)
    else:
        static_cat = None

    if config['num_static_real_features'] > 0:
        static_real = batch["static_real_features"].float().to(device)       # (B, F_sr)
    else:
        static_real = None

    # ---------- 2. Build X = [static, past_time_features, past_values, future_time_summary] ----------
    feat_list = []

    # (a) static features -> broadcast to (B, T_in, N, F_static)
    if static_cat is not None:
        # (B, 1, 1, F_sc) -> (B, T_in, N, F_sc)
        sc = static_cat.unsqueeze(1).unsqueeze(2).expand(-1, T_in, N, -1)
        feat_list.append(sc)

    if static_real is not None:
        # static_real is currently (B, 1, 1, N, F_s) or similar
        # → squeeze to (B, N, F_s)
        static_real = static_real.squeeze(1).squeeze(1)   # (B, N, F_s)
        # (B, N, F_s) -> (B, T_in, N, F_s)
        sr = static_real.unsqueeze(1).expand(-1, T_in, -1, -1)
        feat_list.append(sr)

    # (b) past time features -> broadcast over nodes
    # past_time_features: (B, T_in, F_t) -> (B, T_in, N, F_t)
    ptf = past_time_features.unsqueeze(2).expand(-1, -1, N, -1)
    feat_list.append(ptf)

    # (c) past values as a feature
    # past_values: (B, T_in, N) -> (B, T_in, N, 1)
    pv = past_values.unsqueeze(-1)
    feat_list.append(pv)

    # (d) summary of future_time_features as "future-aware" static covariate
    # future_time_features: (B, T_out, F_t) -> summary (e.g., mean) (B, F_t)
    # then broadcast to (B, T_in, N, F_t)
    if future_time_features is not None:
        fut_summary = future_time_features.mean(dim=1)          # (B, F_t)
        fut_summary = fut_summary.unsqueeze(1).unsqueeze(2).expand(-1, T_in, N, -1)
        feat_list.append(fut_summary)

    # concatenate along feature dim -> X: (B, T_in, N, F_in)
    X = torch.cat(feat_list, dim=-1)

    # Y is just future_values with a channel dim: (B, T_out, N, 1)
    Y = future_values.unsqueeze(-1)
    return X, Y

'training and inference functions of STGCN_NB'
def infer_stgcn_dims_from_dataloader(train_dataloader, config):
    # Get one batch (no accelerator here yet)
    batch = next(iter(train_dataloader))

    # Past / future values: (B, T_in, N) and (B, T_out, N)
    print(batch["past_values"].shape)
    B, T_in, N = batch["past_values"].shape
    print(batch["future_values"].shape)
    T_out = batch["future_values"].shape[1]

    # Static categorical
    if config["num_static_categorical_features"]> 0:
        F_sc = batch["static_categorical_features"].shape[-1]
    else:
        F_sc = 0

    # Static real
    if config["num_static_real_features"] > 0:
        F_sr = batch["static_real_features"].shape[-1]
    else:
        F_sr = 0

    # Time features
    F_pt = batch["past_time_features"].shape[-1]    # past_time_features: (B, T_in, F_pt)
    F_ft = batch["future_time_features"].shape[-1]  # future_time_features: (B, T_out, F_ft)

    # One channel for past_values
    F_in = F_sc + F_sr + F_pt + 1 + F_ft

    # print(f"F_in: {F_in}")
    # print(f'F_sc: {F_sc}')
    # print(f'F_sr: {F_sr}')
    # print(f'F_pt: {F_pt}')
    # print(f'F_ft: {F_ft}')
    # print(f'auxilary channel: 1')

    return {
        "F_in": F_in,
        "num_nodes": N,
        "T_in": T_in,
        "T_out": T_out,
    }

'util: NLL loss'

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

def train_STGCN(train_dataloader, config, device, accelerator):
    'train the model'
    lr_ = config['lr']
    num_epochs_ = config['num_epochs']
    weight_decay_ = config['weight_decay']

    dims = infer_stgcn_dims_from_dataloader(train_dataloader, config)

    F_in      = dims["F_in"]
    num_nodes = dims["num_nodes"]
    T_out     = dims["T_out"]

    # print('debugging: ', dims)

    model = STGCN_NB(
      num_nodes=num_nodes,
      in_channels=F_in,
      hidden_channels=config['hidden_channels'],
      nb_channels=1,
      horizon=config['prediction_length'],
      num_blocks=config['num_STGCN_blocks'],
      kernel_size=max(config['lags_sequence']),
      dropout=config['dropout'],
      A=config['A_static'],   # or set later via set_adj
    )

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr_, betas=(0.9, 0.95), weight_decay=weight_decay_)
    # compile model
    model, optimizer, train_dataloader = accelerator.prepare(
    model,
    optimizer,
    train_dataloader,
    )

    A = config['A_static'].to(device)

    model.train()
    loss_history = []
    avg_loss_print = []
    for epoch in tqdm(range(num_epochs_)):
        for idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            X,Y = build_XY_from_batch(batch, device,config)
            # === Forward pass ===
            # STGCN_NB is assumed to return mu, theta
            mu, theta = model(X, A)                # (B, T_out, N, 1) each

            # === Loss ===
            loss = negative_binomial_nll(Y, mu, theta, reduction="mean")

            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()

            loss_history.append(loss.item())

            if idx % 100 == 0:
                avg_loss_print.append(np.mean(loss_history))
                loss_history = []

    return model, avg_loss_print

def stgcn_nb_inference(
    model,
    X,
    A,
    n_samples: int = 0,
    clamp_eps: float = 1e-8,
    return_params: bool = True,
    as_long: bool = True,
):
    """
    Run STGCN_NB and optionally sample from the NB predictive distribution.

    Args
    ----
    model:      STGCN_NB instance (already on correct device, in eval() mode)
    X:          (B, T_in, N, F_in)  input tensor
    A:          adjacency matrix, e.g. (N, N)
    n_samples:  if > 0, draw this many samples per (B, T_out, N, C)
    clamp_eps:  small value to avoid numerical issues in mu/theta
    return_params:
                if True, also return (mu, theta)
    as_long:    if True, cast samples to torch.long (integer counts)

    Returns
    -------
    If n_samples <= 0:
        - mu, theta

    If n_samples > 0 and return_params:
        - samples, mu, theta
          where samples: (B, n_samples, T_out, N, C)

    If n_samples > 0 and not return_params:
        - samples
    """
    # 1) Forward pass through STGCN_NB
    mu, theta = model(X, A)     # shapes: (B, T_out, N, C)

    # No sampling requested → just return parameters
    if n_samples <= 0:
        return mu, theta

    # 2) Sample from NB(mu, theta)
    # Parameterization: mean = mu, shape = theta
    # p = theta / (theta + mu)
    mu_clamped = mu.clamp(min=clamp_eps)
    theta_clamped = theta.clamp(min=clamp_eps)

    p = theta_clamped / (theta_clamped + mu_clamped)

    # torch.distributions.NegativeBinomial expects total_count=theta, probs=p
    nb_dist = torch.distributions.NegativeBinomial(
        total_count=theta_clamped,
        probs=p,
    )

    # samples_raw: (n_samples, B, T_out, N, C)
    samples_raw = nb_dist.sample((n_samples,))

    if as_long:
        samples_raw = samples_raw.long()

    # Reorder to (B, n_samples, T_out, N, C) – often more convenient
    samples = samples_raw.permute(1, 0, 2, 3, 4).contiguous()

    if return_params:
        return samples, mu, theta
    else:
        return samples
