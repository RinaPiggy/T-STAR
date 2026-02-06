from models.TST_func import *
from tuning_modules import *

is_pretrained = False
# predictor_type = 'pickup' #'dropoff'
predictor_type = 'dropoff' #'dropoff'
# feature_type = 'type_6' #'univariate'
# feature_type = 'type_6_1' # for Baseline TST
feature_type = 'type_0' # for Simple TST

target_col, dynamic_features, static_features = set_config_type(feature_type, predictor_type)
num_dynamic_real_features_ = len(dynamic_features) # print to be 3
num_static_real_features_ = len(static_features) # print to be 1

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

model_name = f"quarterly_{predictor_type}_tuning"
file_save_path = f'results/quarterly/{model_name}'

# create directory if not exist
os.makedirs(file_save_path, exist_ok=True)

# NEW: create train and test sets #
df_demand['date'] = pd.to_datetime(df_demand['date'])
df_train_full = df_demand[df_demand['date'] < training_end].copy()
# df_test = df_demand[df_demand['date'] >= training_end].copy()

# define fixed hyperparameters
fixed_params = {
    "prediction_length": prediction_length,
    "num_dynamic_real_features": num_dynamic_real_features_,
    "num_static_real_features": num_static_real_features_,
    "target_col": target_col,
    "lags_sequence": [1,2,3,4,5,6,7,8,24],
    "time_features": time_features_from_frequency_str(freq)[1:3],
    "context_length": 6,
    "num_batches_per_epoch": 256,
    "batch_size":  128,
    "num_epochs": 10,
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
n_splits=3, #5
scoring="RMSE",
n_trials=10,
pred_type = predictor_type
)


print("Best hyperparameters found: \n")
print(f"----------Quarterly T-STAR Stage 2 Model {predictor_type}-----------")
print(best_params)

print("Best hyperparameters found: \n")
print(f"----------Quarterly T-STAR Stage 2 Model {predictor_type}-----------")
print(best_params)