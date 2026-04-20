import torch
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from einops import rearrange

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

from uni2ts.eval_util.plot import plot_next_multi
from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, mean_squared_log_error
from skforecast.metrics import mean_absolute_scaled_error

from scipy.signal import savgol_filter


### Moirai settings ###
MODEL = "moirai2"  # model name: choose from {'moirai', 'moirai-moe', 'moirai2'}
SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
#PDT = 100  # prediction length: any positive integer
#CTX = 400  # context length: any positive integer (was 200)
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 32  # batch size: any positive integer
#TEST = 100  # test set length: any positive integer


filter_window = 11
polyorder = 3
modelname = "Moirai2.0"


### START OF FILE ###
# scripts.data_preprocessing import *
# The following rows have what data preprocessing includes:
# Functions to process the data, to open it and to format it in ways that can be utilized in the algorithms
 

def normalize_data(historical, true_future):
    historical = savgol_filter(historical, window_length=filter_window, polyorder=3)
    true_future = savgol_filter(true_future, window_length=filter_window, polyorder=3)
    normalized_historical = (historical - np.min(historical)) / (np.max(historical) -np.min(historical))
    normalized_historical = pd.DataFrame(normalized_historical, columns=["data"])
    normalized_true_future_array = (true_future - np.min(historical)) / (np.max(historical) -np.min(historical))
    normalized_true_future = pd.DataFrame(normalized_true_future_array, columns=["data"])

    normalized_data = pd.concat([normalized_historical, normalized_true_future], ignore_index=True)
    #print(normalized_data)
    #return normalized_historical, normalized_true_future # Works previously!
    return normalized_data, normalized_true_future_array

# Includes all the functionalities from opening the file, to normalizing it
# Normalizing data using values from the training data only, yet then combining all the normalized data back together
# inputs are file and the length of the test set required
# returns the whole normalized data set, and the unique column name for the data


"""
# With regular datasets
normalized_dataEDA= file_to_normalized('C:\\Users\\Omistaja\\._.Tampere University\\Thesis\\Code tests\\TimesFMwithElectricdata\\S27\\1\\1\\T2EDA.csv', TEST)
normalized_dataEDA.columns = ['EDA']

timestamp, item_id = create_timestamp(len(normalized_dataEDA), "S27")

df = pd.DataFrame({"timestamp": timestamp, 'target':normalized_dataEDA["EDA"], "item_id": item_id})

ds = PandasDataset.from_long_dataframe(df, target="target", item_id="item_id")
"""



def predict_data(task, subject, modality, url):
    try:
        df = pd.read_csv(url, index_col=0, parse_dates=True)
    except FileNotFoundError:
        df = pd.DataFrame()
    if df.empty:
        return np.nan, np.nan, np.nan
    values_df = pd.DataFrame({"y":df[modality]})
    values_array = values_df.to_numpy()
    values_array = np.ravel(values_array)
    #print(values_array)

    # Splitting data to be able to normalize it, and after normalization putting it together

    TEST = math.floor(len(values_array) * 0.3) #testing with the last 30% of the data, TEST is an integer for the index
    print(" the index of the test/train split is ", TEST)

    train_array = values_array[:-TEST]
    test_array = values_array[-TEST:]
    normalized_df, normalized_test_array = normalize_data(train_array, test_array)


    # Replacing the old data with the newly normalized data
    df = df.drop([modality], axis=1)
    df.insert(len(df.columns), 'target', normalized_df.values)
    #df = pd.concat([df, normalized_df], axis=1)

    #print(df["target"])


    if MODEL == "moirai2":
        model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(
                f"Salesforce/moirai-2.0-R-small",
            ),
            prediction_length=len(test_array),
            context_length=len(train_array),
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

    # Split into train/test set
    train, test_template = split(
        df, offset=-TEST
    )  # assign last TEST time steps as test set

    

    inp = {
        "target": df["target"].to_numpy()[:-TEST],
        "start": df.index[0].to_period(freq="s"),
    }
    label = {
        "target": df["target"].to_numpy()[-TEST:],
        "start": df.index[-TEST].to_period(freq="s"),
    }

    past_target = rearrange(
        torch.as_tensor(inp["target"], dtype=torch.float32), "t -> 1 t 1"
    )
    # 1s if the value is observed, 0s otherwise. Shape: (batch, time, variate)
    #past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
    # 1s if the value is padding, 0s otherwise. Shape: (batch, time)
    #past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)

    forecast = model.predict(past_target)
    #print(forecast)

    #print(
    #    "median prediction:\n",
    #    np.round(np.median(forecast[0], axis=0), decimals=4),
    #)
    #print("ground truth:\n", label["target"])
    median_prediction = np.round(np.median(forecast[0], axis=0), decimals=10)

    #print("Forecast shape", forecast.shape)
    #print("Label values", len(label["target"]))

    #print(f"Logarithmic normalizedtestarray {np.log(normalized_test_array)}, Lorarithmic median prediction { np.log(median_prediction)}")

    # Evaluates the MSE of the code
    RMSE = root_mean_squared_error(normalized_test_array, median_prediction)
    MAE = mean_absolute_error(normalized_test_array, median_prediction)
    MASE = mean_absolute_scaled_error(normalized_test_array, median_prediction, y_train=normalized_df['data'][:-TEST].to_numpy())


    print("="*20)
    print(f"Root Mean Squared Error for {task}{subject}{modality}:", RMSE)
    print(f"Mean Absolute Error for {task}{subject}{modality}: ", MAE)
    print(f"Mean Absolute Scaled Error for {task}{subject}{modality}: ", MASE)
    print("="*20)

    return RMSE, MAE, MASE



# Smoothing filter
#values_array = savgol_filter(values_array, window_length=100, polyorder=3)



def write_data(task, subject, modality, RMSE, MAE, MASE):
    index = subject + task
    columnRMSE = modality + "/RMSE"
    columnMAPE = modality + "/MAE"
    columnMASE = modality + "/MASE"
    results_df.at[index,columnRMSE] = RMSE
    results_df.at[index,columnMAPE] = MAE
    results_df.at[index,columnMASE] = MASE


# Opening files, creating results file
datapath = "C:\\Users\\Omistaja\\._.Tampere University\\Thesis\\Code tests\\TimesFMwithElectricdata\\.venv\\scripts\\gluonts_datasets\\"

index_col_array = []
for i in range (1, 9):
        task  = f"T{i}"
        for n in range(1, 29):
            if n == 15:
                continue 
            if n == 20:
                if i > 4:
                    continue
            subject = f"S{n}"
            title = subject + task
            index_col_array.append(title)

results_df = pd.DataFrame(columns= ['TEMP/RMSE', 'TEMP/MAE', 'TEMP/MASE', 'EDA/RMSE', 'EDA/MAE', 'EDA/MASE', 'HR/RMSE', 'HR/MAE', 'HR/MASE'],
                          index= index_col_array)

# Loop to call predictions and write them
for mod in range(0,3):
    if mod == 0:
        modality = "TEMP"
    elif mod == 1:
        modality = "EDA"
    else:
        modality = "HR"
    for i in range (1, 3):
        task  = f"T{i}"
        for n in range(1, 5):
            if n == 15:
                continue 
            subject = f"S{n}"
            url = datapath + subject + "\\" + subject + task + modality + ".csv"
            RMSE, MAE, MASE = predict_data(task, subject, modality, url)
            write_data(task, subject, modality, RMSE, MAE, MASE) # UNCOMMENT THIS!!
            
# Uncomment this!!
#print(results_df)
results_df.to_csv(f"{modelname}.csv")





"""
# Construct rolling window evaluation
test_data = test_template.generate_instances(
    prediction_length=PDT,  # number of time steps for each prediction
    windows=TEST // PDT,  # number of windows in rolling window evaluation
    distance=PDT,  # number of time steps between each window - distance=PDT for non-overlapping windows
)

if MODEL == "moirai2":
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
print(forecast_it)



# Making predictions
inp = next(input_it)
label = next(label_it)
forecast = next(forecast_it) #this doesn't work?

# Evaluates the MSE of the code
RMSE = root_mean_squared_error(normalized_test_array, forecast.median)
MAPE = mean_absolute_percentage_error(normalized_test_array, forecast.median)
print("="*20)
print("Root Mean Squared Error: ", RMSE)
print(f"Mean absolute percentage error: {MAPE:.6f} %")
print("="*20)

plot_single(
    inp, 
    label, 
    forecast, 
    context_length=1000,
    name="pred",
    show_label=True,
    )
plt.show()



"""