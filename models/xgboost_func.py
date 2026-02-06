'dataset and dataloader creation functions for XGBoost model'

import pandas as pd

def data_split(data_,
               station_id,
               config):
  data_ = data_[data_.station_id == station_id]
  if config['pred_freq'] == '1H':
    data_ = data_.sort_values(by=['date','hour'])
  else:
    data_ = data_.sort_values(by=['date','hour','quarter'])
  # data_train, data_test = data_split_by_dates(data_, 'date', config['training_end'])
  data_['date'] = pd.to_datetime(data_['date'])
  data_train = data_[data_['date'] <= config['training_end']]
  data_test = data_[(data_['date'] > config['training_end'])] # Added parentheses around the condition
  del data_

  target = config['target_col']
  feature_lst = config['feature_lst']
  X_train = data_train[feature_lst]
  X_test = data_test[feature_lst]
  y_train = data_train[target]
  y_test = data_test[target]
  # free up memory
  del data_train, data_test, feature_lst
  return X_train, X_test, y_train, y_test

def creat_data_loader(data, config):
  training_dataloader = {}
  testing_dataloader = {}
  station_list = data.station_id.unique()

  for station_id in station_list:
    station_key = f'station_{station_id}'
    dataset_train = {}
    dataset_test = {}
    X_train, X_test, y_train, y_test = data_split(data,
                                        station_id=station_id,
                                        config=config)
    dataset_train['X'] = X_train
    dataset_test['X'] = X_test
    dataset_train['y'] = y_train
    dataset_test['y'] = y_test
    # free up memory
    del X_train, X_test, y_train, y_test
    training_dataloader[station_key] = dataset_train
    testing_dataloader[station_key] = dataset_test
    del dataset_train, dataset_test
  return training_dataloader, testing_dataloader

'training function for XGBoost model'
import pickle

def save_model_xgb(model_path,xgb_model):
    file_name = f"{model_path}/xgboost.pkl"
    pickle.dump(xgb_model, open(file_name, "wb"))

def load_model_xgb(model_path):
    file_name = f"{model_path}/xgboost.pkl"
    with open(file_name, 'rb') as file:
        model = pickle.load(file)
    return model

import xgboost as xgb

def xgboost_fit(X_train, y_train, params, seed: int = 42):
    model = xgb.XGBRegressor(
        objective="count:poisson",
        colsample_bytree=params["colsample_bytree"],
        learning_rate=params["learning_rate"],
        max_depth=int(params["max_depth"]),
        n_estimators=int(params["n_estimators"]),
        random_state=seed,     # key line
    )
    model.fit(X_train.values, y_train.values)
    return model

'inference function for XGBoost model'
import numpy as np

def xgboost_inference(model, station_key, dataloader_):
    X_test = dataloader_[station_key]['X']
    y_test = dataloader_[station_key]['y']
    # dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(X_test.values)

    return y_pred, y_test # y_actual

def xgboost_inference_both(model, dataloader_):
  forecast_lst = []
  actual_lst = []

  for station_key, dataset in dataloader_.items():
    forecast, actual = xgboost_inference(model, station_key, dataloader_)

    forecast_lst.append(forecast)
    actual_lst.append(actual)

  forecast_lst = np.array(forecast_lst)
  actual_lst = np.array(actual_lst)
  return forecast_lst, actual_lst
