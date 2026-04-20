# Functions to process the data, to open it and to format it in ways that can be utilized in the algorithms

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download

from uni2ts.eval_util.plot import plot_next_multi
from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

scaler = StandardScaler()
min_max_scaler = preprocessing.MinMaxScaler()

MODEL = "moirai2"  # model name: choose from {'moirai', 'moirai-moe', 'moirai2'}
SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
PDT = 100  # prediction length: any positive integer
CTX = 400  # context length: any positive integer (was 200)
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 32  # batch size: any positive integer
TEST = 100  # test set length: any positive integer

"""
for n in range(1,9):
    data = pd.read_csv(f'C:\\Users\\Omistaja\\._.Tampere University\\Thesis\\Code tests\\TimesFMwithElectricdata\\S27\\1\\1\\T{n}EDA.csv')
    time = pd.to_timedelta(np.arange(0, len(data), 1), 'sec')
    plt.plot(time, data, label=f"EDA {n}")
plt.legend()
plt.title("Plot for 8 EDA's")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
"""

def create_timestamp(length_of_time, id):
    start = datetime(2020, 1, 1, 0, 0, 0, 0)
    timestamp = start
    timedata = []
    item_id = []
    for i in range(length_of_time):
        timedata.append(timestamp)
        timestamp += timedelta(seconds=0.25)
        item_id.append(id)



def open_file(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return {}
    return df

def split_train_test(df, testset_length):
    # Split into train/test set
    train, test_template = split(
        df, offset=-testset_length
        )  # assign last TEST time steps as test set
    
    horizon_len = PDT  # Defining how many points are forecasted
    historical_data = df[:-horizon_len]
    true_future_values = df[-horizon_len:]
    return historical_data, true_future_values

def normalize_data(historical, true_future):
    normalized_historical = (historical - np.min(historical)) / (np.max(historical) -np.min(historical))
    normalized_true_future = (true_future - np.min(historical)) / (np.max(historical) -np.min(historical))


    return normalized_historical, normalized_true_future

#def into_gluonts_format(df, id):

def file_to_normalized(file_path, testset_length):
    df = open_file(file_path)
    train, test_template = split_train_test(df, testset_length)
    norm_train, norm_true_future = normalize_data(train, test_template)
    print(train, test_template)
    print(norm_train, norm_true_future)
    return norm_train, norm_true_future

file_to_normalized('C:\\Users\\Omistaja\\._.Tampere University\\Thesis\\Code tests\\TimesFMwithElectricdata\\S27\\1\\1\\T2EDA.csv', 100)

#'C:\\Users\\Omistaja\\._.Tampere University\\Thesis\\Code tests\\TimesFMwithElectricdata\\S27\\1\\1\\T2EDA.csv'

"""
# This is wrong, only the training set should be actually normalized lol
df = pd.DataFrame({"timestamp": timestamp, 'target':data['1711570295.000000'], "item_id": item_id})
#normalizing the array!
float_array = df['target'].values.astype(float).reshape(-1,1)
scaled_array = min_max_scaler.fit_transform(float_array)
df["target"]=scaled_array
"""
"""
# Base from https://github.com/SalesforceAIResearch/uni2ts
# https://www.salesforce.com/blog/moirai-2-0/




MODEL = "moirai2"  # model name: choose from {'moirai', 'moirai-moe', 'moirai2'}
SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
PDT = 100  # prediction length: any positive integer
CTX = 400  # context length: any positive integer (was 200)
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 32  # batch size: any positive integer
TEST = 100  # test set length: any positive integer


# NEW CODE
# Reading the emphatic school data
data=pd.read_csv('T4hr.csv')
#df = pd.DataFrame({'y':data['1711570295.000000']})
#time = pd.to_timedelta(np.arange(0, len(df), 1), 's')

print(data)

#attempting to create a timestamp
start = datetime(2020, 1, 1, 0, 0, 0)
end = datetime(2020, 1, 1, 0, 8, 59)
timestamp = start
timedata = []
item_id = []
while timestamp <= end:
    timedata.append(timestamp)
    timestamp += timedelta(seconds=1)
    item_id.append("A")

    #['1711570295.000000']


# This is wrong, only the training set should be actually normalized lol
df = pd.DataFrame({"timestamp": timestamp, 'target':data['1711570295.000000'], "item_id": item_id})
#normalizing the array!
float_array = df['target'].values.astype(float).reshape(-1,1)
scaled_array = min_max_scaler.fit_transform(float_array)
df["target"]=scaled_array



# --- 2. Prepare Data ---

time_series_data = df['y'].values
horizon_len = 256  # Defining how many seconds are forecasted
historical_data = time_series_data[:-horizon_len]
true_future_values = time_series_data[-horizon_len:]

#End new code
"""

"""
# Read data into pandas DataFrame
url = (
    "https://gist.githubusercontent.com/rsnirwan/c8c8654a98350fadd229b00167174ec4"
    "/raw/a42101c7786d4bc7695228a0f2c8cea41340e18f/ts_wide.csv"
)

df = pd.read_csv(url, index_col=0, parse_dates=True)



ds = PandasDataset.from_long_dataframe(df, target="target", item_id="item_id")

# Split into train/test set
train, test_template = split(
    ds, offset=-TEST
)  # assign last TEST time steps as test set

# Construct rolling window evaluation
test_data = test_template.generate_instances(
    prediction_length=PDT,  # number of time steps for each prediction
    windows=TEST // PDT,  # number of windows in rolling window evaluation
    distance=PDT,  # number of time steps between each window - distance=PDT for non-overlapping windows
)


# Prepare pre-trained model by downloading model weights from huggingface hub
if MODEL == "moirai":
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{SIZE}"),
        prediction_length=PDT,
        context_length=CTX,
        patch_size=PSZ,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )
elif MODEL == "moirai-moe":
    model = MoiraiMoEForecast(
        module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{SIZE}"),
        prediction_length=PDT,
        context_length=CTX,
        patch_size=16,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )
    
elif MODEL == "moirai2":
    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained(
            f"Salesforce/moirai-2.0-R-small",
        ),
        prediction_length=PDT,
        context_length=CTX,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )


predictor = model.create_predictor(batch_size=BSZ)
forecasts = predictor.predict(test_data.input)

input_it = iter(test_data.input)
label_it = iter(test_data.label)
forecast_it = iter(forecasts)



inp = next(input_it)
label = next(label_it)
forecast = next(forecast_it)

plot_single(
    inp, 
    label, 
    forecast, 
    context_length=1000,
    name="pred",
    show_label=True,
)
plt.show()"""