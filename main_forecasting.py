"""
This main file use the final run of T-STAR as a case study example.
For the benchmark model executions, 
users can replace base model correspondingly.
"""

from models.TST_func import *
from General_Utils import *
import os
from datetime import timedelta

seed_sequence = [42, 3407, 99, 107, 256] # for instance

import random
import numpy as np
import torch

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)              # CPU RNG
    torch.cuda.manual_seed_all(seed)     # all CUDA devices (safe even if you use 1)

# ---------- configuration of experiment: T-STAR Stage 1 ---------#
is_pretrained = False #False

predictor_type = 'dropoff'
# predictor_type = 'pickup' #'dropoff'
feature_type = 'multivariate' #'univariate'

num_dynamic_real_features_ = 4

# 'number of days in training & validation set'
training_days = 70
validation_days = 0

# frequency of data stream
freq = '1H' #'15min'

# one-step ahead prediction
prediction_length = 1

# --------------------- Begin ---------------------
if predictor_type == 'dropoff':
    # load data from the csv files
    df_demand = pd.read_csv('data/hourly_dropoff_complete.csv')
else:
    df_demand = pd.read_csv('data/hourly_pickup_complete.csv')

# split the date into train, test and validation* sets by dates
df_demand['date'] = pd.to_datetime(df_demand['date'])
earliest_date = df_demand['date'].min()
latest_date = df_demand['date'].max()

training_end = earliest_date + timedelta(days=training_days)
validation_end = training_end + timedelta(days=validation_days)

model_name = f"hourly_{predictor_type}_final"
file_save_path = f'results/T_STAR/{model_name}'

# create directory if not exist
os.makedirs(file_save_path, exist_ok=True)

# ----------------- model hyperparameter -----------------#
# types of lags to be used in Time Series Transformer
lags_sequence_ = [1,2,3,4,24]
# systematic time features DayofWeek, HourOfDay
time_features_ = time_features_from_frequency_str(freq)[:2]
# kernel size, recommend 4-12
context_length_ = 6
# tune done batch size if computation is too expensive
batch_size_ = 128
num_batches_per_epoch_ = 256
epochs = 100
decoder_layers_ = 2
weight_decay_ = 1e-2
attention_dropout_ = 0.1

# below are the tunable hyperparameters
learning_rate_ = 0.00015 # 0.0006, 0.00015
dropout_ = 0.2 # 0.1, 0.2
# model architecture
encoder_layers_ = 2 # n1, 2
d_model = 64 # dimension of model

# transformer learner config
model_config = {'batch_size': batch_size_,
            'num_batches_per_epoch': num_batches_per_epoch_,
            'num_epochs': epochs,
            'lr': learning_rate_,
            'weight_decay': weight_decay_,
            'dropout': dropout_,
            'context_length': context_length_,
            'lags_sequence': lags_sequence_,
            'time_features': time_features_,
            'encoder_layers': encoder_layers_,
            'decoder_layers': decoder_layers_,
            'd_model': d_model,
            'attention_dropout': attention_dropout_,
            }

train_dataset,test_dataset,full_dataset, uni_dataset_dict = get_dataset(df_demand,
                                                                        feature_type,
                                                                        training_end,
                                                                        validation_end,
                                                                        freq)

# model config of pickup demand predictor
config = TimeSeriesTransformerConfig(
    prediction_length=prediction_length,

    context_length=model_config['context_length'],  # context length:
    lags_sequence=model_config['lags_sequence'],  # lags coming from helper given the freq:
    num_time_features=len(model_config['time_features'])+ 1, # we'll add 2+1 time features (and "age", see further):
    num_static_categorical_features=1, # we have a single static categorical feature, namely time series ID:
    cardinality=[len(train_dataset)], # it has 235 possible values:
    embedding_dimension=[6], # the model will learn an embedding of size 6 for each of the 235 possible values:
    # --- multivariate --- #
    num_dynamic_real_features = num_dynamic_real_features_, # 3 real value dynamical features from weather + 1 binary feature about holiday
    # --------------------

    # transformer params:
    encoder_layers=model_config['encoder_layers'],
    decoder_layers=model_config['decoder_layers'],
    d_model=model_config['d_model'],
    dropout=model_config['dropout'],
    attention_dropout=model_config['attention_dropout'],
    distribution_output = 'negative_binomial', # *distribution_output : Could be either “student_t”, “normal” or “negative_binomial”.
    )

# dataloader for demand predictor
train_dataloader = create_train_dataloader(
    config=config,
    freq=freq,
    data=train_dataset,
    batch_size=batch_size_,
    num_batches_per_epoch=num_batches_per_epoch_,
)

dataloader_debugger_print(train_dataloader)

accelerator = Accelerator()
device = accelerator.device

# multi-runs
model_seed_lst = {}
for seed in seed_sequence:
  file_save_path_seed = f'{file_save_path}_{seed}'
  if is_pretrained == False:
      'create new model from config'
      seed_all(seed)
      model = TimeSeriesTransformerForPrediction(config)
      model, avg_loss_print = train_model(model,
                                          train_dataloader,
                                          config,
                                          model_config,
                                          device,accelerator)
      # save the model
      model.save_pretrained(file_save_path_seed, filename=model_name)
      # visualize the loss along the training process
      training_loss_plot(avg_loss_print)
      # save model to list for calling later
      model_seed_lst[seed] = model
  else:
      # if the model has been saved, we can load it back
      model = TimeSeriesTransformerForPrediction.from_pretrained(file_save_path_seed)
      model.to(device)
      model_seed_lst[seed] = model

training_batch_size_ = len(train_dataset[0]['target']) - (config.context_length + max(config.lags_sequence))

testing_batch_size_ = len(test_dataset[0]['target'])
print(testing_batch_size_)

test_dataloader, train_val_dataloader = create_backtest_dataloader(
    config=config,
    freq=freq,
    data=full_dataset,
    testing_size=testing_batch_size_,
    batch_size=testing_batch_size_, # so each time series will be its own batch -> convienient for eval
    batch_size_train=training_batch_size_,
)

for seed in tqdm(seed_sequence):
  file_save_path_seed = f'{file_save_path}_{seed}'
  model_seed = model_seed_lst[seed]

  test_actuals, test_forecasts = model_inference(model_seed, test_dataloader, config, device)
  np.save(f'{file_save_path_seed}/test_forecasts.npy', test_forecasts) # shape = (235, 504, 100, 1)

  train_actuals, train_forecasts = model_inference(model_seed, train_val_dataloader, config, device)

  np.save(f'{file_save_path_seed}/train_forecasts.npy', train_forecasts) #

# inference: testing set
test_actuals, test_forecasts = model_inference(model, test_dataloader, config, device)
test_forecasts_median = np.median(test_forecasts, 2)
test_forecasts_std = np.std(test_forecasts, 2)


# inference: training set
train_actuals, train_forecasts = model_inference(model, train_val_dataloader, config, device)
train_forecasts_median = np.median(train_forecasts, 2)
train_forecasts_std = np.std(train_forecasts, 2)

# save the forecasts

np.save(f'{file_save_path}/{model_name}_test_forecasts_dropoff.npy', test_forecasts) # shape = (235, 504, 100, 1)
np.save(f'{file_save_path}/{model_name}_train_forecasts_dropoff.npy', train_forecasts) # shape = (235, 1680, 100, 1)


# ... pipeline stage 1 predictions into stage 2 dataset... #

# stage 2 forecasting #

# is_pretrained = True
is_pretrained = False #True # False
# predictor_type = 'pickup' #'dropoff' #
predictor_type = 'dropoff' #'pickup' #
feature_type = 'type_6'

target_col, dynamic_features, static_features = set_config_type(feature_type, predictor_type)
num_dynamic_real_features_ = len(dynamic_features)
num_static_real_features_ = len(static_features)

# 'number of days in training & validation set'
training_days = 70
validation_days = 0
# frequency of data stream
freq = '15min'

prediction_length = 1
# --------------------- Begin ---------------------
df_demand = pd.read_csv('data/intermediate_15min_updated.csv')
training_days = 70
validation_days = 0

# find out data spliting timestamps
df_demand['date'] = pd.to_datetime(df_demand['date'])
earliest_date = df_demand['date'].min()
latest_date = df_demand['date'].max()
training_end = earliest_date + timedelta(days=training_days)
validation_end = training_end + timedelta(days=validation_days)

# remove duplicates in df_demand
df_demand = df_demand.drop_duplicates()

df_demand['delta_dropoff_lag'] = df_demand.groupby('station_id')['delta_dropoff'].shift(1)
df_demand['delta_pickup_lag'] = df_demand.groupby('station_id')['delta_pickup'].shift(1)

# drop rows with nan in 'delta_dropoff_lag'
df_demand = df_demand.dropna(subset=['delta_dropoff_lag'])

# drop rows with nan in 'delta_pickup_lag'
df_demand = df_demand.dropna(subset=['delta_pickup_lag'])

## match with contextual data
df_weather = pd.read_csv('data/weather_training_small.csv')
df_holiday = pd.read_csv('data/holiday_training_small.csv')

df_weather['date'] = pd.to_datetime(df_weather['date'])
df_weather['hour'] = df_weather['date'].dt.hour
df_weather['date'] = df_weather['date'].dt.date
df_weather = df_weather[['date', 'hour','station_id','temperature_2m', 'precipitation', 'wind_speed_10m']]

df_holiday['date'] = pd.to_datetime(df_holiday['date']).dt.date

df_demand['date'] = pd.to_datetime(df_demand['date']).dt.date
df_demand = pd.merge(df_demand, df_weather, on=['date', 'hour','station_id'], how='left')
df_demand = pd.merge(df_demand, df_holiday, on=['date'], how='left')

model_name = f"quarterly_{predictor_type}_final"
file_save_path = f'results/T_STAR/{model_name}'

# create directory if not exist
os.makedirs(file_save_path, exist_ok=True)

# from datasets import Dataset, DatasetDict (df_, feature_type, pred_type, training_end, validation_end,freq)

train_dataset,test_dataset,full_dataset, uni_dataset_dict = get_dataset_stage2(df_demand,
                                                                        feature_type,
                                                                        predictor_type,
                                                                        training_end,
                                                                        validation_end,
                                                                        freq)   

# define fixed hyperparameters
hyperparams = {
    "prediction_length": prediction_length,
    "num_dynamic_real_features": num_dynamic_real_features_,
    "num_static_real_features" : num_static_real_features_,
    "lags_sequence": [1,2,3,4,5,6,7,8,24],
    "time_features": time_features_from_frequency_str(freq)[1:3],
    "context_length": 6,
    "num_batches_per_epoch": 256,
    "batch_size":  128,
    "num_epochs": 100,
    "weight_decay": 1e-2,
    "decoder_layers": 2,
    "attention_dropout": 0.1,
    "embedding_dimension": [6],
    "is_pretrained": False,
    # tuned params
    "lr": 0.00027, # 0.00072, 0.00027,#
    "d_model": 64, # 16, 64,#
    "encoder_layers": 2, #3, 2,
    "dropout": 0.1 #0.2, 0.1,
}

# model config of pickup demand predictor
config = TimeSeriesTransformerConfig(
    prediction_length=hyperparams['prediction_length'],
    context_length=hyperparams['context_length'],  # context length:
    lags_sequence=hyperparams['lags_sequence'],  # lags coming from helper given the freq:
    num_time_features=len(hyperparams['time_features'])+ 1, # we'll add 2+1 time features (and "age", see further):
    num_static_categorical_features=1, # we have a single static categorical feature, namely time series ID:
    cardinality=[len(train_dataset)], # it has 235 possible values:
    embedding_dimension=[6], # the model will learn an embedding of size 6 for each of the 235 possible values:
    # --- multivariate --- #
    num_dynamic_real_features = hyperparams['num_dynamic_real_features'], # 3 real value dynamical features from weather + 1 binary feature about holiday
    num_static_real_features = hyperparams['num_static_real_features'], # 1
    # --------------------

    # transformer params:
    encoder_layers=hyperparams['encoder_layers'],
    decoder_layers=hyperparams['decoder_layers'],
    d_model=hyperparams['d_model'],
    dropout=hyperparams['dropout'],
    attention_dropout=hyperparams['attention_dropout'],
    distribution_output = 'negative_binomial', # *distribution_output : Could be either “student_t”, “normal” or “negative_binomial”.
    )

# dataloader for demand predictor
train_dataloader = create_train_dataloader(
    config=config,
    freq=freq,
    data=train_dataset,
    batch_size=hyperparams['batch_size'],
    num_batches_per_epoch=hyperparams['num_batches_per_epoch'],
)

accelerator = Accelerator()
device = accelerator.device

# multi-runs
model_seed_lst = {}
for seed in seed_sequence:
  file_save_path_seed = f'{file_save_path}_{seed}'
  if is_pretrained == False:
      'create new model from config'
      seed_all(seed)
      model = TimeSeriesTransformerForPrediction(config)
      model, avg_loss_print = train_model(model,
                                          train_dataloader,
                                          config,
                                          hyperparams,
                                          device,accelerator)
      # save the model
      model.save_pretrained(file_save_path_seed, filename=model_name)
      # visualize the loss along the training process
      training_loss_plot(avg_loss_print)
      # save model to list for calling later
      model_seed_lst[seed] = model
  else:
      # if the model has been saved, we can load it back
      model = TimeSeriesTransformerForPrediction.from_pretrained(file_save_path_seed)
      model.to(device)
      model_seed_lst[seed] = model


training_batch_size_ = len(train_dataset[0]['target']) - (config.context_length + max(config.lags_sequence))

testing_batch_size_ = len(test_dataset[0]['target'])

test_dataloader, train_val_dataloader = create_backtest_dataloader(
    config=config,
    freq=freq,
    data=full_dataset,
    testing_size=testing_batch_size_,
    batch_size=testing_batch_size_, # so each time series will be its own batch -> convienient for eval
    batch_size_train=training_batch_size_,
)

for seed in tqdm(seed_sequence):
  file_save_path_seed = f'{file_save_path}_{seed}'
  model_seed = model_seed_lst[seed]

  test_actuals, test_forecasts = model_inference(model_seed, test_dataloader, config, device)
  np.save(f'{file_save_path_seed}/test_forecasts.npy', test_forecasts) # shape = (235, 504, 100, 1)

  train_actuals, train_forecasts = model_inference(model_seed, train_val_dataloader, config, device)

  np.save(f'{file_save_path_seed}/train_forecasts.npy', train_forecasts) #