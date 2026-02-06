from models.TST_func import *
from tuning_modules import *

# data loading #
is_pretrained = False
# predictor_type = 'pickup' #'dropoff'
predictor_type = 'dropoff' #'pickup' #'dropoff'
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

model_name = f"hourly_{predictor_type}_tuning"
file_save_path = f'results/hourly/{model_name}'

# create directory if not exist
os.makedirs(file_save_path, exist_ok=True)

# NEW: create train and test sets #
df_train_full = df_demand[df_demand['date'] < training_end].copy()
# df_test = df_demand[df_demand['date'] >= training_end].copy()

# define fixed hyperparameters
fixed_params = {
    "prediction_length": prediction_length,
    "num_dynamic_real_features": num_dynamic_real_features_,
    "lags_sequence": [1,2,3,4,24],
    "time_features": time_features_from_frequency_str(freq)[:2],
    "context_length": 6,
    "num_batches_per_epoch": 256,
    "batch_size":  128,
    "num_epochs": 20,
    "weight_decay": 1e-2,
    "decoder_layers": 2,
    "attention_dropout": 0.1,
    "is_pretrained": False,
}
# define hyperparameter grid for tuning
param_grid = {
# "lr": [from 5e-5 to 1e-2], embedded in optuna code
"d_model":       [16, 32, 64],
"encoder_layers":[1, 2, 3],
"dropout":       [0.1, 0.2, 0.3],
}

best_params, cv_results, study = hyperparameter_search(
df_train_full=df_train_full,
freq=freq,
feature_type=feature_type,
param_grid=param_grid,      # only tunable
fixed_params=fixed_params,  # fixed stuff like batch_size = 128
build_and_train_model_fn=build_and_train_model_fn,
n_splits=5,
scoring="RMSE",
n_trials=30,
)

print("Best hyperparameters found: \n")
print("----------Hourly TST Stage 1 Model-----------")
print(f"-----{predictor_type}----")
print(best_params)

# cv_results.to_csv("optuna_cv_results.csv_pickup", index=False)

print("Best hyperparameters found: \n")
print("----------Hourly TST Stage 1 Model-----------")
print(f"-----{predictor_type}----")
print(best_params)

# cv_results.to_csv("optuna_cv_results_dropoff.csv", index=False)