
import timesfm
import torch
import numpy as np
import pandas as pd
import math

from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from skforecast.metrics import mean_absolute_scaled_error

from scipy.signal import savgol_filter



filter_window = 11
polyorder = 3
modelname = "TimesFM"
test_data_percent = 0.3 # 30%
percent = test_data_percent * 100


def normalize_data(historical, true_future):
    historical = savgol_filter(historical, window_length=filter_window, polyorder=polyorder)
    true_future = savgol_filter(true_future, window_length=filter_window, polyorder=polyorder)

    normalized_historical = (historical - np.min(historical)) / (np.max(historical) -np.min(historical))
    normalized_historical = pd.DataFrame(normalized_historical, columns=["data"])
    normalized_true_future = (true_future - np.min(historical)) / (np.max(historical) -np.min(historical))
    normalized_true_future = pd.DataFrame(normalized_true_future, columns=["data"])

    normalized_data = pd.concat([normalized_historical, normalized_true_future], ignore_index=True)
    #print(normalized_data)
    #return normalized_historical, normalized_true_future # Works previously!
    return normalized_data, normalized_historical, normalized_true_future


def predict_data(task, subject, modality, url):
    try:
        df = pd.read_csv(url, parse_dates=True)
    except FileNotFoundError:
        df = pd.DataFrame()
    if df.empty:
        return np.nan, np.nan, np.nan
    values_df = pd.DataFrame({"y":df[modality]})
    values_array = values_df.to_numpy()
    values_array = np.ravel(values_array)
    #print(values_array)

    # Smoothing filter
    # values_array = savgol_filter(values_array, window_length=100, polyorder=3)
    # Splitting data to be able to normalize it, and after normalization putting it together



    TEST = math.floor(len(values_array) * test_data_percent) #Changing to test with 20% instead of 30%
    print(" the index of the test/train split is ", TEST)

    train_array = values_array[:-TEST]
    test_array = values_array[-TEST:]


    normalized_df, normalized_historical, normalized_test_array = normalize_data(train_array, test_array)


    # Replacing the old data with the newly normalized data
    df = df.drop([modality], axis=1)
    df.insert(len(df.columns), 'data', normalized_df.values)

    #id_str = f"{subject}{task}{modality}"
    #print(id_str)
    #id = np.full((len(df), 1), id_str)
    #df.insert(len(df.columns), 'item_id', id)

        
    torch.set_float32_matmul_precision("high")

    # --- 2. Prepare Data ---

    time_series_data = normalized_df['data'].values
    horizon_len = TEST  # Forecast the next 90 points
    historical_data = time_series_data[:-horizon_len]
    true_future_values = time_series_data[-horizon_len:]

        
    # --- 3. Initialize TimesFM 2.5 PyTorch Model ---
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(
        timesfm.ForecastConfig(
            max_context=4024,
            max_horizon=660,
            #normalize_inputs=True, 
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )

    # --- 4. Generate Forecast ---
    point_forecast, quantile_forecast = model.forecast(
        horizon=horizon_len,
        inputs=[historical_data],  # Single time series as input
    )
    forecast_values = point_forecast[0]  # Extract the forecast for our single series
    
    

    # Evaluates the MSE of the code
    RMSE = root_mean_squared_error(normalized_test_array, forecast_values)
    MAE = mean_absolute_error(normalized_test_array, forecast_values)
    MASE = mean_absolute_scaled_error(normalized_test_array.to_numpy(), forecast_values, y_train=normalized_df['data'][:-TEST].to_numpy()) #Revert back: delete to.numpy() from normtestar


    print("="*20)
    print(f"Root Mean Squared Error for {task}{subject}{modality}:", RMSE)
    print(f"Mean Absolute Error for {task}{subject}{modality}: ", MAE)
    print(f"Mean Absolute Scaled Error for {task}{subject}{modality}: ", MASE)
    print("="*20)

    return RMSE, MAE, MASE




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
    for i in range (1, 9):
        task  = f"T{i}"
        for n in range(1, 29):
            if n == 15:
                continue 
            subject = f"S{n}"
            url = datapath + subject + "\\" + subject + task + modality + ".csv"
            RMSE, MAE, MASE = predict_data(task, subject, modality, url)
            write_data(task, subject, modality, RMSE, MAE, MASE)
            

print(results_df)
results_df.to_csv(f"{modelname}{test_data_percent}%testdata.csv")

