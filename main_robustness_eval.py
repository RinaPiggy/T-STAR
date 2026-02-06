'utils'
import os
import numpy as np
import pandas as pd

def infer_steps_per_week(df, date_col="date"):
    df_sorted = df.sort_values(date_col)
    all_dates = np.sort(df_sorted[date_col].unique())

    if len(all_dates) < 2:
        raise ValueError(f"Not enough unique timestamps in df['{date_col}'].")

    dt = pd.Series(all_dates).diff().median()
    if pd.isna(dt) or dt <= pd.Timedelta(0):
        raise ValueError(f"Cannot infer a valid time step from df['{date_col}'].")

    one_week = pd.Timedelta("7D")
    steps_per_week = int(round(one_week / dt))
    return all_dates, dt, max(1, steps_per_week)


def make_expanding_week_folds(
    df_all,
    freq,
    schedule_weeks=None,
    date_col="date",
):
    """
    Expanding-window rolling forecast origin folds with an explicit (train_weeks, test_weeks) schedule.

    Example schedule for 13 weeks total:
      [(4,2), (6,2), (8,2), (10,2), (12,1)]
    """
    if schedule_weeks is None:
        schedule_weeks = [(4, 2), (6, 2), (8, 2), (10, 2), (12, 1)]

    df_sorted = df_all.sort_values(date_col)
    all_dates, dt, steps_per_week = infer_steps_per_week(df_sorted, date_col=date_col)

    folds = []
    n = len(all_dates)

    for fold_idx, (train_w, test_w) in enumerate(schedule_weeks):
        train_steps = train_w * steps_per_week
        test_steps  = test_w  * steps_per_week

        train_start = 0
        train_end_excl = train_start + train_steps
        test_start = train_end_excl
        test_end_excl = test_start + test_steps

        if train_end_excl > n:
            raise ValueError(
                f"Fold {fold_idx}: train_end_excl={train_end_excl} exceeds available timestamps n={n}."
            )
        if test_start >= n:
            raise ValueError(
                f"Fold {fold_idx}: test_start={test_start} is beyond available timestamps n={n}."
            )
        if test_end_excl > n:
            # For strict reporting, fail fast (recommended).
            # If you prefer to truncate, replace with: test_end_excl = n
            raise ValueError(
                f"Fold {fold_idx}: test_end_excl={test_end_excl} exceeds available timestamps n={n}. "
                f"Adjust schedule_weeks."
            )

        train_dates = all_dates[train_start:train_end_excl]
        val_dates   = all_dates[test_start:test_end_excl]

        df_train_fold = df_sorted[df_sorted[date_col].isin(train_dates)].copy()
        df_val_fold   = df_sorted[df_sorted[date_col].isin(val_dates)].copy()
        df_full_fold  = df_sorted[df_sorted[date_col] <= val_dates.max()].copy()

        folds.append({
            "fold": fold_idx,
            "train_weeks": train_w,
            "test_weeks": test_w,
            "train_start": train_dates.min(),
            "train_end": train_dates.max(),
            "test_start": val_dates.min(),
            "test_end": val_dates.max(),
            "df_train": df_train_fold,
            "df_val": df_val_fold,
            "df_full": df_full_fold,
        })

    return folds

'rolling forecast origin experiment: helper functions'

from models.TST_func import *
from General_Utils import *
from hyperparameter_tuning.tuning_modules import *

def train_and_eval_single_fold_new(
    train_dataset,
    val_dataset,
    full_dataset,
    feature_type,
    freq,
    hyperparams,
    build_and_train_model_fn
):
    # Train
    model, config, device = build_and_train_model_fn(
        train_dataset=train_dataset,
        hyperparams=hyperparams,
        freq=freq,
    )

    training_batch_size_ = len(train_dataset[0]["target"]) - (config.context_length + max(config.lags_sequence))
    validating_batch_size_ = len(val_dataset[0]["target"])

    val_dataloader, _ = create_backtest_dataloader(
        config=config,
        freq=freq,
        data=full_dataset,
        testing_size=validating_batch_size_,
        batch_size=validating_batch_size_,
        batch_size_train=training_batch_size_,
    )

    val_actuals, val_forecasts = model_inference(model, val_dataloader, config, device)

    # (N_series, T_val, num_samples, 1) -> median over samples
    val_forecasts_median = np.median(val_forecasts, axis=2)

    rmse_val, mae_val, _, _ = metric_evaluation(train_dataset, val_dataset, val_forecasts_median, freq)

    # IMPORTANT: assume mae_val is station-wise array-like length N_series
    mae_by_station = np.asarray(mae_val).reshape(-1)

    metrics = {
        "MAE_mean_across_stations": float(np.mean(mae_by_station)),
        "MAE_std_across_stations": float(np.std(mae_by_station, ddof=0)),
        "RMSE_mean_across_stations": float(np.mean(np.asarray(rmse_val).reshape(-1))),
        "RMSE_std_across_stations": float(np.std(np.asarray(rmse_val).reshape(-1), ddof=0)),
        "mae_by_station": mae_by_station,  # keep for downstream reporting
    }
    return metrics

def run_robustness_rolling_origin(
    df_all,
    freq,
    feature_type,
    pred_type,
    final_hyperparams,
    build_and_train_model_fn,
    schedule_weeks=None,
    date_col="date",
    station_id_col=None,   # optional, only needed if you want station labels
):
    """
    Runs the strict expanding-window robustness experiment:
      train 4w test 2w, train 6w test 2w, ..., train 12w test 1w (by default).

    Returns
    -------
    fold_results_df : pd.DataFrame
        One row per fold: mean/stdev across stations, date ranges, etc.
    per_station_mae_df : pd.DataFrame
        (optional) station-wise MAE per fold in long format.
    overall_summary : dict
        Overall aggregates across folds (simple averages + pooled).
    """
    folds = make_expanding_week_folds(
        df_all=df_all,
        freq=freq,
        schedule_weeks=schedule_weeks,
        date_col=date_col,
    )

    fold_rows = []
    per_station_rows = []

    for f in folds:
        df_train_fold = f["df_train"]
        df_val_fold   = f["df_val"]
        df_full_fold  = f["df_full"]

        train_dataset, val_dataset, full_dataset, _ = get_dataset_simple(
            df_train_fold,
            df_val_fold,
            df_full_fold,
            feature_type,
            pred_type,
            freq,
        )

        metrics = train_and_eval_single_fold_new(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            full_dataset=full_dataset,
            feature_type=feature_type,
            freq=freq,
            hyperparams=final_hyperparams,
            build_and_train_model_fn=build_and_train_model_fn,
        )

        new_info = {
            "fold": f["fold"],
            "train_weeks": f["train_weeks"],
            "test_weeks": f["test_weeks"],
            "train_start": f["train_start"],
            "train_end": f["train_end"],
            "test_start": f["test_start"],
            "test_end": f["test_end"],
            "MAE_mean_across_stations": metrics["MAE_mean_across_stations"],
            "MAE_std_across_stations": metrics["MAE_std_across_stations"],
            "RMSE_mean_across_stations": metrics["RMSE_mean_across_stations"],
            "RMSE_std_across_stations": metrics["RMSE_std_across_stations"],
            # "mae_by_station
        }

        fold_rows.append(new_info)

        print(new_info)

        mae_by_station = metrics["mae_by_station"]
        for i, mae_i in enumerate(mae_by_station):
            per_station_rows.append({
                "fold": f["fold"],
                "train_weeks": f["train_weeks"],
                "test_weeks": f["test_weeks"],
                "station_index": i,
                "station_id": (None if station_id_col is None else i),  # replace if you can map IDs
                "MAE": float(mae_i),
            })

    fold_results_df = pd.DataFrame(fold_rows)
    per_station_mae_df = pd.DataFrame(per_station_rows)

    # Overall summaries: (a) average fold mean, (b) average fold std, (c) pooled station MAE across all folds
    overall = {
        "mean_of_fold_mean_MAE": float(fold_results_df["MAE_mean_across_stations"].mean()),
        "mean_of_fold_std_station_MAE": float(fold_results_df["MAE_std_across_stations"].mean()),
        "mean_of_fold_mean_RMSE": float(fold_results_df["RMSE_mean_across_stations"].mean()),
        "mean_of_fold_std_station_RMSE": float(fold_results_df["RMSE_std_across_stations"].mean()),
    }
    pooled_mae = per_station_mae_df["MAE"].to_numpy()
    # pooled_rmse = per_station_mae_df["RMSE"].to_numpy()
    overall.update({
        "pooled_station_MAE_mean": float(np.mean(pooled_mae)),
        "pooled_station_MAE_std": float(np.std(pooled_mae, ddof=0)),
        # "pooled_station_RMSE_mean": float(np.mean(pooled_rmse)),
        # "pooled_station_RMSE_std": float(np.std(pooled_rmse, ddof=0)),
        "n_folds": int(fold_results_df.shape[0]),
        "n_station_fold_points": int(per_station_mae_df.shape[0]),
    })

    return fold_results_df, per_station_mae_df, overall

def make_rolling_week_folds(
    df_all,
    freq,
    train_weeks=8,
    test_weeks=2,
    step_weeks=1,
    date_col="date",
):
    """
    Rolling-window (fixed-length) forecast origin folds.

    Example (train=8, test=2, step=1):
      fold0: train w0-w7, test w8-w9
      fold1: train w1-w8, test w9-w10
      ...
    until the last test window reaches the end.

    Returns a list of fold dicts, each containing df_train/df_val/df_full and metadata.
    """
    df_sorted = df_all.sort_values(date_col)
    all_dates, dt, steps_per_week = infer_steps_per_week(df_sorted, date_col=date_col)

    train_steps = train_weeks * steps_per_week
    test_steps  = test_weeks  * steps_per_week
    step_steps  = step_weeks  * steps_per_week

    n = len(all_dates)
    folds = []

    fold_idx = 0
    train_start = 0

    while True:
        train_end_excl = train_start + train_steps
        test_start = train_end_excl
        test_end_excl = test_start + test_steps

        # stop condition: cannot form a full test window
        if test_end_excl > n:
            break

        train_dates = all_dates[train_start:train_end_excl]
        val_dates   = all_dates[test_start:test_end_excl]

        df_train_fold = df_sorted[df_sorted[date_col].isin(train_dates)].copy()
        df_val_fold   = df_sorted[df_sorted[date_col].isin(val_dates)].copy()
        df_full_fold  = df_sorted[df_sorted[date_col] <= val_dates.max()].copy()

        folds.append({
            "fold": fold_idx,
            "train_weeks": train_weeks,
            "test_weeks": test_weeks,
            "step_weeks": step_weeks,
            "train_start": train_dates.min(),
            "train_end": train_dates.max(),
            "test_start": val_dates.min(),
            "test_end": val_dates.max(),
            "df_train": df_train_fold,
            "df_val": df_val_fold,
            "df_full": df_full_fold,
        })

        fold_idx += 1
        train_start += step_steps

    if len(folds) == 0:
        raise ValueError(
            "No folds were created. Check that you have enough data for "
            f"train_weeks={train_weeks}, test_weeks={test_weeks}."
        )

    return folds

'rolling forecast window experiment: helper functions'

def run_robustness_rolling_window(
    df_all,
    freq,
    feature_type,
    pred_type,
    final_hyperparams,
    build_and_train_model_fn,
    train_weeks=8,
    test_weeks=2,
    step_weeks=1,
    date_col="date",
):
    folds = make_rolling_week_folds(
        df_all=df_all,
        freq=freq,
        train_weeks=train_weeks,
        test_weeks=test_weeks,
        step_weeks=step_weeks,
        date_col=date_col,
    )

    fold_rows = []
    per_station_rows = []

    for f in folds:
        train_dataset, val_dataset, full_dataset, _ = get_dataset_simple(
            f["df_train"],
            f["df_val"],
            f["df_full"],
            feature_type,
            pred_type,
            freq,
        )

        metrics = train_and_eval_single_fold_new(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            full_dataset=full_dataset,
            feature_type=feature_type,
            freq=freq,
            hyperparams=final_hyperparams,
            build_and_train_model_fn=build_and_train_model_fn,
        )

        fold_rows.append({
            "fold": f["fold"],
            "train_weeks": f["train_weeks"],
            "test_weeks": f["test_weeks"],
            "step_weeks": f["step_weeks"],
            "train_start": f["train_start"],
            "train_end": f["train_end"],
            "test_start": f["test_start"],
            "test_end": f["test_end"],
            "MAE_mean_across_stations": metrics["MAE_mean_across_stations"],
            "MAE_std_across_stations": metrics["MAE_std_across_stations"],
            "RMSE_mean_across_stations": metrics["RMSE_mean_across_stations"],
            "RMSE_std_across_stations": metrics["RMSE_std_across_stations"],
        })

        mae_by_station = metrics["mae_by_station"]
        for i, mae_i in enumerate(mae_by_station):
            per_station_rows.append({
                "fold": f["fold"],
                "station_index": i,
                "MAE": float(mae_i),
            })

    fold_results_df = pd.DataFrame(fold_rows)
    per_station_mae_df = pd.DataFrame(per_station_rows)

    pooled = per_station_mae_df["MAE"].to_numpy()
    overall = {
        "mean_of_fold_mean_MAE": float(fold_results_df["MAE_mean_across_stations"].mean()),
        "mean_of_fold_std_station_MAE": float(fold_results_df["MAE_std_across_stations"].mean()),
        "pooled_station_MAE_mean": float(np.mean(pooled)),
        "pooled_station_MAE_std": float(np.std(pooled, ddof=0)),
        "n_folds": int(fold_results_df.shape[0]),
    }

    return fold_results_df, per_station_mae_df, overall

'main - T-STAR'
is_pretrained = True
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
df_demand = pd.read_csv('/content/drive/MyDrive/Paper 1 Revision Runs/intermediate_15min_updated.csv')
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
df_weather = pd.read_csv('/content/drive/MyDrive/PhD/Module I/Data/weather_training_small.csv')
df_holiday = pd.read_csv('/content/drive/MyDrive/PhD/Module I/Data/holiday_training_small.csv')

df_weather['date'] = pd.to_datetime(df_weather['date'])
df_weather['hour'] = df_weather['date'].dt.hour
df_weather['date'] = df_weather['date'].dt.date
df_weather = df_weather[['date', 'hour','station_id','temperature_2m', 'precipitation', 'wind_speed_10m']]

df_holiday['date'] = pd.to_datetime(df_holiday['date']).dt.date

df_demand['date'] = pd.to_datetime(df_demand['date']).dt.date
df_demand = pd.merge(df_demand, df_weather, on=['date', 'hour','station_id'], how='left')
df_demand = pd.merge(df_demand, df_holiday, on=['date'], how='left')

model_name = f"quarterly_{predictor_type}_final"
file_save_path = f'/content/drive/MyDrive/Paper 1 Revision Runs/results/T_STAR/{model_name}'

# create directory if not exist
os.makedirs(file_save_path, exist_ok=True)

train_dataset,test_dataset,full_dataset, uni_dataset_dict = get_dataset_stage2(df_demand,
                                                                        feature_type,
                                                                        predictor_type,
                                                                        training_end,
                                                                        validation_end,
                                                                        freq)

# define fixed hyperparameters
final_hyperparams = {
    "prediction_length": prediction_length,
    "num_dynamic_real_features": num_dynamic_real_features_,
    "num_static_real_features" : num_static_real_features_,
    "lags_sequence": [1,2,3,4,5,6,7,8,24],
    "time_features": time_features_from_frequency_str(freq)[1:3],
    "context_length": 6,
    "num_batches_per_epoch": 256,
    "batch_size":  128,
    "num_epochs": 30,
    "weight_decay": 1e-2,
    "decoder_layers": 2,
    "attention_dropout": 0.1,
    "embedding_dimension": [6],
    "is_pretrained": False,
    # tuned params
    "lr": 0.00072, # 0.00072, 0.00027,#
    "d_model": 16, # 16, 64,#
    "encoder_layers": 3, #3, 2,
    "dropout": 0.2 #0.2, 0.1,
}

schedule = [(4,2), (6,2), (8,2), (10,2), (12,1)]

'rolling origin test'

fold_df, station_df, overall = run_robustness_rolling_origin(
    df_all=df_demand,      # the full 91-day / 13-week dataframe
    freq=freq,
    feature_type=feature_type,
    pred_type=predictor_type,
    final_hyperparams=final_hyperparams,
    build_and_train_model_fn=build_and_train_model_fn,
    schedule_weeks=schedule,
    date_col="date",
)

'rolling window test'

fold_df_rolling, station_df_rolling, overall_rolling = run_robustness_rolling_window(
    df_all=df_demand,     # full 91-day data
    freq=freq,
    feature_type=feature_type,
    pred_type=predictor_type,
    final_hyperparams=final_hyperparams,
    build_and_train_model_fn=build_and_train_model_fn,
    train_weeks=8,
    test_weeks=2,
    step_weeks=1,
    date_col="date",
)

