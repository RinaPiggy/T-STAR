'dataloader and dataset creation functions for TST model'

'''
This file contains utility functions for creating dataloaders for training and backtesting
the Time Series Transformer model using the Hugging Face Transformers library.
'''

# basic libraries
import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from tqdm import tqdm
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# dataloaders and datasets
from datasets import Dataset as HF_Dataset  # the class we need
from datasets import DatasetDict

# modeling libraries
import torch
from typing import Optional, Iterable
from functools import lru_cache
from functools import partial
from typing import Iterable
from gluonts.itertools import Cached, Cyclic
from gluonts.dataset.loader import as_stacked_batches
from gluonts.transform.sampler import InstanceSampler
from gluonts.time_feature import (
    time_features_from_frequency_str,
    TimeFeature,
    get_lags_for_frequency,
    get_seasonality
)
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    RenameFields,
)
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
from transformers import PretrainedConfig
from evaluate import load
from General_Utils import *

def create_transformation(freq: str, config: PretrainedConfig) -> Transformation:
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
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
            if config.num_static_categorical_features > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                )
            ]
            if config.num_static_real_features > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.TARGET,
                # we expect an extra dim for the multivariate case:
                expected_ndim=1 if config.input_size == 1 else 2,
            ),
            # step 3: handle the NaN's by filling in the target with zero
            # and return the mask (which is in the observed values)
            # true for observed values, false for nan's
            # the decoder uses this mask (no loss is incurred for unobserved values)
            # see loss_weights inside the xxxForPrediction model
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # step 4: add temporal features based on freq of the dataset
            # month of year in the case when freq="M"
            # these serve as positional encodings
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                # time_features=time_features_from_frequency_str(freq),
                time_features=time_features_from_frequency_str(freq)[1:3],
                pred_length=config.prediction_length,
            ),
            # step 5: add another temporal feature (just a single number)
            # tells the model where in the life the value of the time series is
            # sort of running counter
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            # step 6: vertically stack all the temporal features into the key FEAT_TIME
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    # [FieldName.FEAT_DYNAMIC_REAL_TEMP, FieldName.FEAT_DYNAMIC_REAL_PREP, FieldName.FEAT_DYNAMIC_REAL_WINDS]
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if config.num_dynamic_real_features > 0
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

def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
    train_sampler: Optional[InstanceSampler] = None,
    validation_sampler: Optional[InstanceSampler] = None,
) -> Transformation:
    assert mode in ["train", "validation", "test"]

    instance_sampler = {
        "train": train_sampler
        or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=config.prediction_length
        ),
        "validation": validation_sampler
        or ValidationSplitSampler(min_future=config.prediction_length),
        "test": TestSplitSampler(),
    }[mode]

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )

def create_train_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    num_batches_per_epoch: int,
    shuffle_buffer_length: Optional[int] = None,
    cache_data: bool = True,
    **kwargs,
) -> Iterable:
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=True)

    if cache_data:
        transformed_data = Cached(transformed_data)

    # we initialize a Training instance
    instance_splitter = create_instance_splitter(config, "train")

    # the instance splitter will sample a window of
    # context length + lags + prediction length (from the 235 possible transformed time series)
    # randomly from within the target time series and return an iterator.
    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(stream)

    return as_stacked_batches(
        training_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )

def create_rolling_instance_splitter(config, datah):
    """
    Creates an instance sampler that samples instances from the test data in a rolling fashion.

    Args:
        config (PretrainedConfig): Model configuration.
        data (Dataset): Test dataset.
        prediction_length (int): Length of the prediction horizon.
        context_length (int): Length of the context window.

    Returns:
        InstanceSampler: Rolling instance sampler.

    Update Log 2024.08.07:
        * change prediction_length into future_length
        * change context_length into past_length
        * remove prediction_length, context_length from function inputs
    """

    past_length=config.context_length + max(config.lags_sequence)
    future_length=config.prediction_length

    def rolling_instance_sampler(data, is_train=False):
        # Convert the TransformedDataset to a list
        data_list = list(data)

        # Initialize an empty list to store the sampled instances
        sampled_instances = []

        original_len = len(data_list[0]["values"])
        # Iterate over each time series
        for time_series in data_list:
            # Get the time series data
            time_series_data = time_series["values"]
            id_ = time_series["static_categorical_features"]
            temp = []
            assert len(time_series_data) == original_len, f"time series length mismatch {len(time_series_data)}-{original_len}"

            # Iterate over the time steps in a rolling fashion
            for i in range(len(time_series_data) - past_length - future_length + 1):
                # Sample the context window and prediction horizon
                context_window = time_series_data[i:i + past_length]
                prediction_horizon = time_series_data[i + past_length:i + past_length + future_length]
                past_time_features_ = torch.from_numpy(time_series["time_features"][:, i:i + past_length]).float()
                future_time_features_ = torch.from_numpy(time_series["time_features"][:,i + past_length:i + past_length + future_length]).float()

                # Create a new instance with the context window and prediction horizon
                instance = {
                    "past_time_features": past_time_features_.permute(1, 0),
                    "past_values": torch.from_numpy(context_window),
                    "past_observed_mask": torch.from_numpy(time_series["observed_mask"][i:i + past_length]),
                    "static_categorical_features": torch.LongTensor(id_),
                    "future_time_features": future_time_features_.permute(1, 0),
                    "future_values": torch.from_numpy(prediction_horizon),
                    "future_observed_mask": torch.from_numpy(time_series["observed_mask"][i + past_length:i + past_length + future_length]),
                }

                del context_window
                del prediction_horizon
                del past_time_features_
                del future_time_features_

                # if static_real_features exists
                if config.num_static_real_features > 0:
                    instance["static_real_features"] = torch.from_numpy(time_series["static_real_features"])

                # Add the instance to the list of sampled instances
                temp.append(instance)
            sampled_instances.append(temp)
            del time_series_data
            del temp

        return sampled_instances
    return rolling_instance_sampler

'NEW: BY HUGGING FACE CHAT'

# use PyTorch dataloader instead
from torch.utils.data import DataLoader, Dataset

# Create a custom dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, instances, exist_static_real_=False):
        self.instances = instances
        self.exist_static_real_ = exist_static_real_

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        temp = {
            "past_time_features": instance["past_time_features"],
            "past_values": instance["past_values"],
            "past_observed_mask": instance["past_observed_mask"],
            "static_categorical_features": instance["static_categorical_features"],
            "future_time_features": instance["future_time_features"],
            "future_values": instance["future_values"],
            "future_observed_mask": instance["future_observed_mask"],
        }
        if self.exist_static_real_:
            temp["static_real_features"] = instance["static_real_features"]

        return temp


def create_backtest_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    testing_size: int,
    batch_size: int,
    batch_size_train: int,
    # instance_size: int,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        print('yes, static real exists') # debugger print
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data)

    for item in transformed_data:
        print(item.keys())
        print('check: id', item['static_categorical_features'])
        break  # Print only the first item

    # Create a rolling instance splitter
    instance_sampler = create_rolling_instance_splitter(config,
                                                        transformed_data)

    # Apply the transformations in train mode
    all_instances = instance_sampler(transformed_data, is_train=True)

    print('next')

    def get_training_testing_instances(all_instances, cut_pt):
        training_instances = []
        testing_instances = []
        print(len(all_instances))
        print(len(all_instances[0]))
        for ts_instances in all_instances:
            training_instances.extend(ts_instances[:-cut_pt])
            testing_instances.extend(ts_instances[-cut_pt:])

        return training_instances, testing_instances

    # testing data instance size: testing_size
    training_instances, testing_instances = get_training_testing_instances(all_instances, testing_size)
    print('length training instances', len(training_instances))
    print('length testing instances', len(testing_instances))

    del all_instances

    print('next')
    # check whether static real features are present
    exist_static_real_ = config.num_static_real_features > 0
    print('2nd check - exist static real: ', exist_static_real_) # debugger print
    # # Create a dataset instance
    test_ins_dataset = TimeSeriesDataset(testing_instances, exist_static_real_=exist_static_real_)
    train_ins_dataset = TimeSeriesDataset(training_instances, exist_static_real_=exist_static_real_)

    del training_instances
    del testing_instances

    # # Create a data loader instance
    test_data_loader = DataLoader(test_ins_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    train_data_loader = DataLoader(train_ins_dataset, batch_size=batch_size_train, shuffle=False, num_workers=2)

    return test_data_loader,train_data_loader

# split datasets function
def data_split_by_dates(df, date_col, training_end, validation_end):
  df[date_col] = pd.to_datetime(df[date_col])
  df['dayofweek'] = df[date_col].dt.dayofweek
  df_train = df[df[date_col] <= training_end]
  df_val = df[df[date_col] < validation_end]
  df_test = df[df[date_col] > validation_end]
  print(len(df_train[date_col].unique()))
  print(len(df_test[date_col].unique()))
  return df_train, df_test, df


def to_dataset(df_, model_type):
  # Group by station_id
  grouped = df_.groupby('station_id')

  # Prepare lists to hold the data
  start_dates = []
  targets = []
  item_ids = []
  station_id_cardinality = []
  # Populate the lists with data
  cardinality_ = 0

  # Create the DatasetDict
  if model_type == 'univariate':
    for station_id, group in grouped:
        group = group.sort_values(by=['date', 'hour'])  # Ensure time order
        start_dates.append(group['date'].iloc[0])
        targets.append(group['count'].tolist())
        item_ids.append(str(station_id))
        station_id_cardinality.append([cardinality_])
        cardinality_ += 1

    data_temp = {
        'start': start_dates,
        'target': targets,
        'feat_static_cat': station_id_cardinality,
        'item_id': item_ids
    }

    dataset_ = HF_Dataset.from_dict(data_temp)

  elif model_type == 'multivariate':
    feat_dynamic_reals = []

    for station_id, group in grouped:
        group = group.sort_values(by=['date', 'hour'])  # Ensure time order
        start_dates.append(group['date'].iloc[0])
        targets.append(group['count'].tolist())
        # --- multivariate ---
        # temp = np.array(group[['temperature_2m', 'wind_speed_10m', 'precipitation']].values.tolist()).T
        temp = np.array(group[['temperature_2m', 'wind_speed_10m', 'precipitation','holiday_binary']].values.tolist()).T
        temp = temp.astype(np.float32)
        feat_dynamic_reals.append(temp)
        # --------------------
        item_ids.append(str(station_id))
        station_id_cardinality.append([cardinality_])
        cardinality_ += 1

    data_temp = {
        'start': start_dates,
        'target': targets,
        'feat_dynamic_real': feat_dynamic_reals,
        'feat_static_cat': station_id_cardinality,
        'item_id': item_ids
    }

    dataset_ = HF_Dataset.from_dict(data_temp)
  else:
    raise ValueError("Invalid model_type. Must be 'univariate' or 'multivariate'.")

  # save item_ids order list
  # np.save('/content/drive/MyDrive/PhD/Module I/July/pretrained/item_ids.npy', np.array(item_ids))

  return dataset_

def to_dataset_stage2(df_, feature_type, pred_type):
  '''
  df_: original dataframe will all features
  feature_type: name of model, corresponding to different sources fusion of input
  pred_type: type of predictor, 'pickup' or 'dropoff'
  '''
  # Group by station_id
  grouped = df_.groupby('station_id')

  # Prepare lists to hold the data
  start_dates = []
  targets = []
  item_ids = []
  station_id_cardinality = []
  # Populate the lists with data
  cardinality_ = 0

  # Create the DatasetDict
  if feature_type == 'type_0':

    if pred_type == 'pickup':
      target_col = 'pickup_count'
      # target_col = 'delta_pickup'
    else:
      target_col = 'dropoff_count'
      # target_col = 'delta_dropoff'

    for station_id, group in grouped:
        group = group.sort_values(by=['date', 'hour'])  # Ensure time order
        start_dates.append(group['date'].iloc[0])
        targets.append(group[target_col].tolist())
        item_ids.append(str(station_id))
        station_id_cardinality.append([cardinality_])
        cardinality_ += 1

    dataset_ = HF_Dataset.from_dict({
        'start': start_dates,
        'target': targets,
        'feat_static_cat': station_id_cardinality,
        'item_id': item_ids
    })

  else:
    target_col, dynamic_features, static_features = set_config_type(feature_type,pred_type)
    feat_dynamic_reals = []
    feat_static_real = []

    for station_id, group in grouped:
        group = group.sort_values(by=['date', 'hour'])  # Ensure time order
        start_dates.append(group['date'].iloc[0])
        targets.append(group[target_col].tolist())
        # ----- conditional feature appending -------
        if len(dynamic_features) > 0:
          temp = np.array(group[dynamic_features].values.tolist()).T
          temp = temp.astype(np.float32)
          feat_dynamic_reals.append(temp)
        if len(static_features) > 0:
          temp = np.array(group[static_features].values.tolist())[0]
          temp = temp.astype(np.float32)
          feat_static_real.append(temp)
        # --------------------
        item_ids.append(str(station_id))
        station_id_cardinality.append([cardinality_])
        cardinality_ += 1
    if len(feat_static_real) > 0:
      dataset_ = HF_Dataset.from_dict({
          'start': start_dates,
          'target': targets,
          'feat_dynamic_real': feat_dynamic_reals,
          'feat_static_real': feat_static_real,
          'feat_static_cat': station_id_cardinality,
          'item_id': item_ids
      })
    else:
      dataset_ = HF_Dataset.from_dict({
          'start': start_dates,
          'target': targets,
          'feat_dynamic_real': feat_dynamic_reals,
          'feat_static_cat': station_id_cardinality,
          'item_id': item_ids
      })

  return dataset_

def get_dataset(df_,feature_type,training_end, validation_end,freq):
    df_train, df_test, df_full = data_split_by_dates(df_, 'date', training_end, validation_end)

    train_dataset = to_dataset(df_train,feature_type)
    test_dataset = to_dataset(df_test,feature_type)
    full_datset = to_dataset(df_full,feature_type)

    # Create DatasetDict
    uni_pickup_dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset,
        'full': full_datset
    })
    train_dataset = uni_pickup_dataset_dict["train"]
    test_dataset = uni_pickup_dataset_dict["test"]
    full_dataset = uni_pickup_dataset_dict["full"]

    train_dataset.set_transform(partial(transform_start_field, freq=freq))
    test_dataset.set_transform(partial(transform_start_field, freq=freq))
    full_dataset.set_transform(partial(transform_start_field, freq=freq))

    return train_dataset,test_dataset,full_dataset,uni_pickup_dataset_dict

# stage 2 dataset config:
def get_dataset_stage2(df_, feature_type, pred_type, training_end, validation_end,freq):
    df_train, df_test, df_full = data_split_by_dates(df_, 'date', training_end, validation_end)

    train_dataset = to_dataset_stage2(df_train, feature_type, pred_type)
    test_dataset = to_dataset_stage2(df_test, feature_type, pred_type)
    full_datset = to_dataset_stage2(df_full, feature_type, pred_type)

    # Create DatasetDict
    uni_pickup_dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset,
        'full': full_datset
    })
    train_dataset = uni_pickup_dataset_dict["train"]
    test_dataset = uni_pickup_dataset_dict["test"]
    full_dataset = uni_pickup_dataset_dict["full"]

    train_dataset.set_transform(partial(transform_start_field, freq=freq))
    test_dataset.set_transform(partial(transform_start_field, freq=freq))
    full_dataset.set_transform(partial(transform_start_field, freq=freq))

    return train_dataset,test_dataset,full_dataset,uni_pickup_dataset_dict


@lru_cache(10_000)
def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)

def transform_start_field(batch, freq):
    batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
    return batch

def dataloader_debugger_print(dataloader):
    batch = next(iter(dataloader))
    print(len(batch))
    print(batch.keys())
    counter_batch = 0

    for k, v in batch.items():
        print(k, v.shape, v.type())

    for batch in dataloader:
      counter_batch += 1
    print(counter_batch)

'''
This file contains the functions to train, debug, and inference generation of the model. The functions are as follows:
1. forward_debugger: This function is used to debug the forward pass of the model. It prints the loss value after the forward pass.
2. train_model: This function is used to train the model. It takes the model, train_dataloader, config,
    and training_config as input and returns the trained model and the average loss per batch during training.
3. training_loss_plot: This function is used to plot the training loss. It takes the average loss per batch during training as input.
4. model_inference: This function is used to generate forecasts using the trained model.
    It takes the model, test_dataloader, config, and device as input and returns the actual values and forecasts.
'''

# basic libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from torch.optim import AdamW


def forward_debugger(model, config, train_dataloader):
    'debug forward pass of the model'
    batch = next(iter(train_dataloader))
    # perform forward pass
    outputs = model(
        past_values=batch["past_values"],
        past_time_features=batch["past_time_features"].float(),
        past_observed_mask=batch["past_observed_mask"],
        static_categorical_features=batch["static_categorical_features"]
        if config.num_static_categorical_features > 0
        else None,
        static_real_features=batch["static_real_features"]
        if config.num_static_real_features > 0
        else None,
        future_values=batch["future_values"],
        future_time_features=batch["future_time_features"].float(),
        future_observed_mask=batch["future_observed_mask"],
        output_hidden_states=True,
    )
    print("Loss:", outputs.loss.item())

def train_model(model,train_dataloader,config, training_config,device,accelerator):
    'train the model'
    lr_ = training_config['lr']
    num_epochs_ = training_config['num_epochs']
    weight_decay_ = training_config['weight_decay']

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr_, betas=(0.9, 0.95), weight_decay=weight_decay_)

    # compile model
    model, optimizer, train_dataloader = accelerator.prepare(
    model,
    optimizer,
    train_dataloader,
    )

    print("accelerator.device:", accelerator.device)
    print("model device:", next(model.parameters()).device)

    device = accelerator.device

    model.train()
    loss_history = []
    avg_loss_print = []
    for epoch in tqdm(range(num_epochs_)):
        for idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(
                static_categorical_features=batch["static_categorical_features"].to(device)
                if config.num_static_categorical_features > 0
                else None,
                static_real_features=batch["static_real_features"].to(device)
                if config.num_static_real_features > 0
                else None,
                past_time_features=batch["past_time_features"].float().to(device),
                past_values=batch["past_values"].to(device),
                future_time_features=batch["future_time_features"].float().to(device),
                future_values=batch["future_values"].to(device),
                past_observed_mask=batch["past_observed_mask"].to(device),
                future_observed_mask=batch["future_observed_mask"].to(device),
            )
            loss = outputs.loss

            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()

            loss_history.append(loss.item())

            if idx % 100 == 0:
                avg_loss_print.append(np.mean(loss_history))
                loss_history = []

    return model, avg_loss_print
# usage
# model, avg_loss_print = train_model(model,train_dataloader,config, training_config, device, accelerator)

def training_loss_plot(avg_loss_print):
    'plot training loss'
    # view training
    x = range(len(avg_loss_print))
    plt.figure(figsize=(10, 5))
    plt.plot(x, avg_loss_print, label="train")
    plt.title("Loss", fontsize=15)
    plt.legend(loc="upper right")
    plt.xlabel("iteration")
    plt.ylabel("Negative Log Likelihood Loss (avg per 150 iterations)")
    plt.show()

def model_inference(model, test_dataloader, config, device):
    'generate forecasts using the trained model'
    model.eval()
    forecasts = []
    actuals = [] # Optional: Note that in actual testing environment, we would not have access to the actuals

    for batch in test_dataloader:
        # my
        actuals.append(batch["future_values"].cpu().numpy()) # Optional
        # original
        outputs = model.generate(
            static_categorical_features=batch["static_categorical_features"].to(device)
            if config.num_static_categorical_features > 0
            else None,
            static_real_features=batch["static_real_features"].to(device)
            if config.num_static_real_features > 0
            else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
            )
        forecasts.append(outputs.sequences.cpu().numpy())

    return np.array(actuals), np.array(forecasts)

# usage
# actuals, forecasts = model_inference(model, test_dataloader, config, device)

