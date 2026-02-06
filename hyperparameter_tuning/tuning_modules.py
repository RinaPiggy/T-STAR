import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from datetime import timedelta

import sys
import os
import optuna

from models.TST_func import *
from General_Utils import *
# this is for stage 1 model
# def get_dataset_simple(df_train, df_val, df_full, feature_type, freq):
    # stage 1 version
    # train_dataset = to_dataset(df_train,feature_type)
    # val_dataset = to_dataset(df_val,feature_type)
    # full_datset = to_dataset(df_full,feature_type)

# this is for stage 2 model
def get_dataset_simple(df_train, df_val, df_full, feature_type, pred_type, freq):
    # stage 2 version: feature_type, pred_type
    train_dataset = to_dataset_stage2(df_train,feature_type, pred_type)
    val_dataset = to_dataset_stage2(df_val,feature_type, pred_type)
    full_datset = to_dataset_stage2(df_full,feature_type, pred_type)

    # Create DatasetDict
    uni_pickup_dataset_dict = DatasetDict({
        'train': train_dataset,
        'val': val_dataset,
        'full': full_datset
    })
    train_dataset = uni_pickup_dataset_dict["train"]
    val_dataset = uni_pickup_dataset_dict["val"]
    full_dataset = uni_pickup_dataset_dict["full"]

    train_dataset.set_transform(partial(transform_start_field, freq=freq))
    val_dataset.set_transform(partial(transform_start_field, freq=freq))
    full_dataset.set_transform(partial(transform_start_field, freq=freq))

    return train_dataset,val_dataset,full_dataset,uni_pickup_dataset_dict

def make_folds(df_train_full, freq, n_splits=5):
    """
    Create rolling time-series folds where each validation fold is 1 'week'
    of data, inferred from the actual spacing of df_train_full['date'].
    """
    # sort and get unique timestamps
    df_sorted = df_train_full.sort_values("date")
    all_dates = np.sort(df_sorted["date"].unique())

    if len(all_dates) < 2:
        raise ValueError("Not enough unique timestamps in df_train_full['date'].")

    # infer sampling interval (median to be robust)
    dt = pd.Series(all_dates).diff().median()
    if pd.isna(dt) or dt <= pd.Timedelta(0):
        raise ValueError("Cannot infer a valid time step from df_train_full['date'].")

    one_week = pd.Timedelta("7D")
    steps_per_week = int(round(one_week / dt))

    if steps_per_week < 1:
        steps_per_week = 1  # safety

    # sanity check: do we have enough samples for n_splits?
    if steps_per_week * n_splits >= len(all_dates):
        raise ValueError(
            f"Too many splits={n_splits} or too large validation window={steps_per_week} "
            f"for number of samples={len(all_dates)}. "
            f"Try fewer splits or a shorter validation window."
        )

    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=steps_per_week
    )

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(all_dates)):
        train_dates = all_dates[train_idx]
        val_dates   = all_dates[val_idx]

        df_train_fold = df_train_full[df_train_full["date"].isin(train_dates)].copy()
        df_val_fold   = df_train_full[df_train_full["date"].isin(val_dates)].copy()

        # optional: "full" up to end of validation window (if you use it in get_dataset_simple)
        df_full_fold  = df_train_full[df_train_full["date"] <= val_dates.max()].copy()

        # NOTE: shape matches how you use it:
        # for fold_idx, (df_train_fold, df_val_fold, df_full_fold) in enumerate(folds):
        folds.append((df_train_fold, df_val_fold, df_full_fold))

    return folds


def build_and_train_model_fn(train_dataset,
        hyperparams,
        freq):
    """
    Placeholder for your model building and training function.
    You need to implement this function to:
    - take (train_dataset, val_dataset, hyperparams, freq, feature_type, ...)
    - train the model
    - return (model, val_forecasts_median)
    """
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
        num_static_real_features = hyperparams['num_static_real_features'],
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

    model = TimeSeriesTransformerForPrediction(config)
    model, avg_loss_print = train_model(model,
                                        train_dataloader,
                                        config,
                                        hyperparams,
                                        device,accelerator)

    return model, config, device

def train_and_eval_single_fold(
    train_dataset,
    val_dataset,
    full_dataset,
    feature_type,
    freq,
    hyperparams,
    build_and_train_model_fn
):
    """
    One fold: convert to datasets, build model, train, evaluate on validation set.

    Parameters
    ----------
    df_train_fold, df_val_fold : pd.DataFrame
        Subsets of df_train_full for this fold.
    feature_type : str
        'univariate' or 'multivariate' as in your current code.
    freq : str
        '1H' or '15min', etc.
    hyperparams : dict
        Hyperparameters for this run.
    build_and_train_model_fn : callable
        A function you write that:
        - takes (train_dataset, val_dataset, hyperparams, freq, feature_type, ...)
        - trains the model
        - returns (model, config)
    """
    # 2. Train model on this split
    model, config, device = build_and_train_model_fn(
        train_dataset=train_dataset,
        hyperparams=hyperparams,
        freq=freq,
    )
    training_batch_size_ = len(train_dataset[0]['target']) - (config.context_length + max(config.lags_sequence))
    validating_batch_size_ = len(val_dataset[0]['target'])

    val_dataloader, _ = create_backtest_dataloader(
        config=config,
        freq=freq,
        data=full_dataset,
        testing_size=validating_batch_size_,
        batch_size=validating_batch_size_,
        batch_size_train=training_batch_size_,
    )

    val_actuals, val_forecasts = model_inference(model, val_dataloader, config, device)

    # assume val_forecasts has shape (N_series, T_val, num_samples, 1)
    val_forecasts_median = np.median(val_forecasts, axis=2)

    # 3. Compute metrics on validation set (you already have metric_validation)
    rmse_val, mae_val, _, _ = metric_evaluation(train_dataset,val_dataset,val_forecasts_median, freq)

    metrics = {
        "RMSE": np.mean(rmse_val),
        "MAE": np.mean(mae_val),
    }

    return metrics


def hyperparameter_search(
    df_train_full,
    freq,
    feature_type,
    param_grid,
    build_and_train_model_fn,
    train_and_eval_single_fold_fn=train_and_eval_single_fold,
    n_splits=5,
    scoring="RMSE",      # name of metric key in metrics dict
    n_trials=30,         # how many trials Optuna should run
    timeout=None,        # optional: max seconds for tuning
    sampler=None,
    pruner=None,
    fixed_params=None,        # <--- prefixed hyperparameters (not tuned)
    pred_type = None
):
    """
    Run 5-fold time-series CV over df_train_full using Optuna.

    Parameters
    ----------
    df_train_full : pd.DataFrame
        Full training period (no test data).
    freq : str
        E.g., '1H', '15min', ...
    feature_type : str
        As in your current code ('univariate', 'multivariate', ...)
    param_grid : dict
        Search space definition. For Optuna we treat each entry as categorical:
            {
                "learning_rate": [1e-4, 3e-4, 1e-3],
                "batch_size":    [32, 64],
                "d_model":       [64, 128],
                ...
            }
    build_and_train_model_fn : callable
        Your wrapper that builds & trains model given
        (train_dataset, val_dataset, full_dataset, hyperparams, freq, feature_type).
    n_splits : int
        Number of CV folds (default 5).
    scoring : str
        Metric key to optimize (e.g., 'MASE', 'RMSE', 'MAE').
        Must match a key returned by train_and_eval_single_fold_fn.
    n_trials : int
        Number of Optuna trials.
    timeout : int or None
        Wall-clock time limit in seconds for Optuna (optional).
    sampler : optuna.samplers.BaseSampler or None
        If None, TPESampler(seed=42) is used.
    pruner : optuna.pruners.BasePruner or None
        If None, MedianPruner() is used.

    Returns
    -------
    best_params : dict
        Best hyperparameters found by Optuna (minimizing `scoring`).
    cv_results : pd.DataFrame
        Per-trial, per-fold metrics. Columns include:
        ['trial', 'fold', *hyperparams, *metrics_keys]
    """
    if fixed_params is None:
        fixed_params = {}

    folds = make_folds(df_train_full, freq=freq, n_splits=n_splits)
    # store all fold results across all trials
    all_results = []

    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=42)
    if pruner is None:
        pruner = optuna.pruners.MedianPruner()

    # direction="minimize" because MASE/RMSE/MAE are "lower is better"
    study = optuna.create_study(direction="minimize",
                                sampler=sampler,
                                pruner=pruner)
    def objective(trial: optuna.Trial):

        # 1.1) Sample hyperparameters from param_grid as categorical choices
        hyperparams = fixed_params.copy()
        # 1.2) Continuous LR between 5e-5 and 1e-2 on a log scale
        hyperparams["lr"] = trial.suggest_float(
            "lr",
            5e-5,
            1e-2,
            log=True,   # VERY important for learning rates
        )
        for name, values in param_grid.items():
            # We assume values is a list of candidates.
            # If you later want ranges, you can special-case those keys.
            hyperparams[name] = trial.suggest_categorical(name, values)

        fold_scores = []

        # 2) Run K-fold time-series CV
        for fold_idx, (df_train_fold,
                       df_val_fold,
                       df_full_fold) in enumerate(folds):

            # build datasets as you already do
            train_dataset, val_dataset, full_dataset, _ = get_dataset_simple(
                df_train_fold,
                df_val_fold,
                df_full_fold,
                feature_type,
                pred_type,
                freq,
            )

            metrics = train_and_eval_single_fold_fn(
                train_dataset,
                val_dataset,
                full_dataset,
                feature_type=feature_type,
                freq=freq,
                hyperparams=hyperparams,
                build_and_train_model_fn=build_and_train_model_fn,
            )

            score = float(metrics[scoring])
            fold_scores.append(score)

            # store detailed result for this fold + trial
            result_row = {
                "trial": trial.number,
                "fold": fold_idx,
                **hyperparams,
                **metrics,
            }
            all_results.append(result_row)

            # report intermediate value to Optuna for pruning
            trial.report(score, step=fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # 3) Objective value = mean score across folds (lower is better)
        mean_score = float(np.mean(fold_scores))

        print(
            f"[trial {trial.number}] {hyperparams} -> "
            f"mean {scoring} over folds = {mean_score:.4f}"
        )

        return mean_score

    # 4) Run the optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    # 5) Extract best params and cv_results
    best_params = study.best_params
    print(f"\nBest params ({scoring}): {best_params}")
    print(f"Best value ({scoring}): {study.best_value:.4f}")

    cv_results = pd.DataFrame(all_results)

    return best_params, cv_results, study
