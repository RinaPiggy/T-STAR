'''
This file contains general utility functions for prediction experiment.
'''

# basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from tqdm import tqdm
from datetime import timedelta
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
from evaluate import load

'''
This file contains general utility functions for prediction experiment.
'''

# basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from tqdm import tqdm
from datetime import timedelta
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
from evaluate import load

def mean_squared_error(y_true, y_pred):
    # RMSE
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def metric_evaluation(train_dataset,test_dataset,forecast_median,freq, simple_=True):
    if simple_:
        rmse_metrics = []
        mae_metrics = []
        mean_forecast_error_metrics = []
        std_forecast_error_metrics = []

        for item_id in range(forecast_median.shape[0]):
            ground_truth = test_dataset[item_id]["target"][:]
            rmse = mean_squared_error(ground_truth, forecast_median[item_id])
            mae = mean_absolute_error(ground_truth, forecast_median[item_id])
            rmse_metrics.append(rmse)
            mae_metrics.append(mae)
            # NEW: Check bias and scale of forecasting errors
            mean_forecast_error = np.mean(forecast_median[item_id] - np.array(ground_truth))
            mean_forecast_error_metrics.append(mean_forecast_error)
            std_forecast_error = np.std(forecast_median[item_id] - np.array(ground_truth))
            std_forecast_error_metrics.append(std_forecast_error)
        return rmse_metrics,mae_metrics,mean_forecast_error_metrics,std_forecast_error_metrics
    else:
        mase_metric = load("evaluate-metric/mase")
        smape_metric = load("evaluate-metric/smape")

        mase_metrics = []
        smape_metrics = []

        prediction_length = forecast_median.shape[1]

        for item_id, ts in enumerate(test_dataset):
            ts_train = train_dataset[item_id]
            training_data = ts_train["target"]
            ground_truth = ts["target"][-prediction_length:]

            if item_id<1:
                print(f"full time series. shape {np.array(ts['target']).shape}")
                print(f"Training data. shape {np.array(training_data).shape}")
                print(f"Prediction. shape {forecast_median[item_id].shape}")
                print(f"Ground Truth. shape {np.array(ground_truth).shape}")

            mase = mase_metric.compute(
                predictions=forecast_median[item_id],
                references=np.array(ground_truth),
                training=np.array(training_data),
                periodicity=get_seasonality(freq),
            )
            mase_metrics.append(mase["mase"])

            smape = smape_metric.compute(
                predictions=forecast_median[item_id],
                references=np.array(ground_truth),
            )
            smape_metrics.append(smape["smape"])

        return mase_metrics,smape_metrics,rmse_metrics,mae_metrics,mean_forecast_error_metrics,std_forecast_error_metrics

# usage
# mase_metrics,smape_metrics,rmse_metrics,mae_metrics = metric_evaluation(train_dataset,test_dataset,forecast_median,freq)

def metric_validation(train_dataset,forecast_median,freq):
    mase_metric = load("evaluate-metric/mase")
    smape_metric = load("evaluate-metric/smape")
    mase_metrics = []
    smape_metrics = []
    prediction_length = forecast_median.shape[1]

    rmse_metrics = []
    mae_metrics = []
    mean_forecast_error_metrics = []
    std_forecast_error_metrics = []

    for item_id in range(forecast_median.shape[0]):
        ground_truth = train_dataset[item_id]["target"][-prediction_length:]
        rmse = mean_squared_error(ground_truth, forecast_median[item_id], squared=False)
        mae = mean_absolute_error(ground_truth, forecast_median[item_id])
        rmse_metrics.append(rmse)
        mae_metrics.append(mae)
        mase = mase_metric.compute(
            predictions=forecast_median[item_id],
            references=np.array(ground_truth),
            training=np.array(ground_truth),
            periodicity=get_seasonality(freq),
        )
        mase_metrics.append(mase["mase"])

        smape = smape_metric.compute(
            predictions=forecast_median[item_id],
            references=np.array(ground_truth),
        )
        smape_metrics.append(smape["smape"])

        # NEW: Check bias and scale of forecasting errors
        mean_forecast_error = np.mean(forecast_median[item_id] - np.array(ground_truth))
        mean_forecast_error_metrics.append(mean_forecast_error)
        std_forecast_error = np.std(forecast_median[item_id] - np.array(ground_truth))
        std_forecast_error_metrics.append(std_forecast_error)

    return mase_metrics,smape_metrics,rmse_metrics,mae_metrics,mean_forecast_error_metrics,std_forecast_error_metrics

def plot_pred_actual(ts_index, test_dataset, forecasts,freq,saving_path):
    fig, ax = plt.subplots()

    index = pd.period_range(
        start=test_dataset[ts_index][FieldName.START],
        periods=len(test_dataset[ts_index][FieldName.TARGET]),
        freq=freq,
    ).to_timestamp()

    # Major ticks every half year, minor ticks every month,
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())

    ax.plot(
        index,
        test_dataset[ts_index]["target"],
        label="actual",
    )

    plt.plot(
        index[:],
        np.median(forecasts, 2)[ts_index].squeeze(),
        label="median",
    )

    mean_ = np.mean(forecasts, 2)[ts_index].squeeze()
    std_ = np.std(forecasts, 2)[ts_index].squeeze()

    plt.fill_between(
        index[:],
        mean_ - std_,
        mean_ + std_,
        alpha=0.3,
        interpolate=True,
        label="+/- 1-std",
    )

    # plt.ylim(-1,5)
    plt.legend()
    plt.show()

    # save plot
    plt.savefig(f'{saving_path}/{ts_index}_testing_forecast.png')
    plt.close()

'type of input features'
def set_config_type(feature_type, pred_type):
  if pred_type == 'pickup':
    target_col = 'pickup_count'
    # target_col = 'delta_pickup'
  else:
    target_col = 'dropoff_count'
    # target_col = 'delta_dropoff'

  if feature_type == 'type_0':
    dynamic_features = []
    static_features = []
    # explanation: past obs. + seasonal features

  elif feature_type == 'type_1':
    static_features = []
    # explanation: past obs. + seasonal feature + stage_1_variation
    if pred_type == 'pickup':
      dynamic_features = ['hr_pred_pickup','hr_std_pickup']
    else:
      dynamic_features = ['hr_pred_dropoff','hr_std_dropoff']

  elif feature_type == 'type_3':
    static_features = ['capacity']
    if pred_type == 'pickup':
      dynamic_features = ['hr_pred_pickup', 'hr_std_pickup', 'delta_checkout']
    else:
      dynamic_features = ['hr_pred_dropoff', 'hr_std_dropoff', 'delta_checkin']
    # explanation: past obs. + seasonal feature + stage_1_variation + endogenous + exogenous_PT_variation

  elif feature_type == 'type_4':
    static_features = ['capacity']
    if pred_type == 'pickup':
      dynamic_features = ['hr_pred_pickup','hr_std_pickup','delta_pickup_lag']
    else:
      dynamic_features = ['hr_pred_dropoff','hr_std_dropoff','delta_dropoff_lag']

  elif feature_type == 'type_5':
    static_features = ['capacity']
    if pred_type == 'pickup':
      dynamic_features = ['hr_pred_pickup','hr_std_pickup','delta_pickup_lag','delta_dropoff_lag']
    else:
      dynamic_features = ['hr_pred_dropoff','hr_std_dropoff','delta_pickup_lag','delta_dropoff_lag']
    # explanation: past obs. + seasonal feature + stage_1_variation
    # + endogenous (pickup dropoff and capacity)

  elif feature_type == 'type_6':
    static_features = ['capacity']
    if pred_type == 'pickup':
      dynamic_features = ['hr_pred_pickup','hr_std_pickup','delta_checkout',
                          'delta_pickup_lag','delta_dropoff_lag']
    else:
      dynamic_features = ['hr_pred_dropoff','hr_std_dropoff','delta_checkin',
                          'delta_pickup_lag','delta_dropoff_lag']
    # explanation: past obs. + seasonal feature + stage_1_variation
    # + endogenous (pickup dropoff and capacity)

  elif feature_type == 'type_6_1':
    static_features = ['capacity']
    if pred_type == 'pickup':
      dynamic_features = ['delta_checkout',
                          'temperature_2m', 'wind_speed_10m', 'precipitation','holiday_binary']
    else:
      dynamic_features = ['delta_checkin',
                          'temperature_2m', 'wind_speed_10m', 'precipitation','holiday_binary']

  elif feature_type == 'global':
    static_features = []
    dynamic_features = ['temperature_2m', 'wind_speed_10m', 'precipitation','holiday_binary']

  elif feature_type == 'STAEformer':
    static_features = []
    dynamic_features = ['tod','dow']
  else:
    raise ValueError("Invalid feature_type. Must be 'type_0', 'type_1', 'type_2', 'type_3', or 'type_4'.")

  return target_col, dynamic_features, static_features

# example usage
# target_col, dynamic_features, static_features = set_config_type('type_1', 'pickup')

