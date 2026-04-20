#https://auto.gluon.ai/dev/tutorials/timeseries/forecasting-chronos.html

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from chronos import BaseChronosPipeline, Chronos2Pipeline
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError, MeanAbsolutePercentageError, MeanSquaredError
from skforecast.metrics import mean_absolute_scaled_error

from scipy.signal import savgol_filter

# Parameters
#PDT = 100  # prediction length: any positive integer
#CTX = 400  # context length: any positive integer (was 200)
#PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
#BSZ = 32  # batch size: any positive integer
#TEST = 100  # test set length: any positive integer


filter_window = 11
polyorder = 3
modelname = "Chronos2.0"
test_data_percent = 0.3


def normalize_data(historical, true_future):
    historical = savgol_filter(historical, window_length=filter_window, polyorder=polyorder)
    true_future = savgol_filter(true_future, window_length=filter_window, polyorder=polyorder)

    normalized_historical = (historical - np.min(historical)) / (np.max(historical) -np.min(historical))
    normalized_historical = pd.DataFrame(normalized_historical, columns=["data"]) # revert back? normalized_historical
    normalized_true_future_array = (true_future - np.min(historical)) / (np.max(historical) -np.min(historical))
    normalized_true_future = pd.DataFrame(normalized_true_future_array, columns=["data"]) # revert back? normalized_true_fuiture_array

    normalized_data = pd.concat([normalized_historical, normalized_true_future], ignore_index=True)
    #print(normalized_data)
    #return normalized_historical, normalized_true_future # Works previously!
    return normalized_data, normalized_true_future_array


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

    TEST = math.floor(len(values_array) * test_data_percent) #testing with the last 30% of the data, TEST is an integer for the index
    print(" the index of the test/train split is ", TEST)

    train_array = values_array[:-TEST]
    test_array = values_array[-TEST:]
    normalized_df, normalized_test_array = normalize_data(train_array, test_array)

    # Replacing the old data with the newly normalized data
    df = df.drop([modality], axis=1)
    df.insert(len(df.columns), 'target', normalized_df.values)

    id_str = f"{subject}{task}{modality}"
    #print(id_str)
    id = np.full((len(df), 1), id_str)
    df.insert(len(df.columns), 'item_id', id)

    df1 = TimeSeriesDataFrame.from_data_frame(
        df, 
        id_column='item_id', 
        timestamp_column= 'timestamp'
        )

    num_test_windows = 3 # was 3 originally
    prediction_length = TEST
    train_data, test_data = df1.train_test_split(num_test_windows * prediction_length)

    predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
        train_data,
        presets= {"chronos2", "best-quality"},
        num_val_windows="auto"

    )
    predictions = predictor.predict(train_data)
    #print(predictions)

    #predictions_per_window = predictor.backtest_predictions(test_data, num_val_windows=num_test_windows)
    #item_id = test_data.item_ids[:2].tolist()
    #all_predictions = pd.concat(predictions_per_window)
    #predictor.plot(test_data, all_predictions, max_history_length=1000, item_ids=item_id)


#print(normalized_df['data'][:-TEST].tolist())

# Evaluates the MSE of the code
    RMSE = root_mean_squared_error(normalized_test_array, predictions['mean'].values)
    MAE = mean_absolute_error(normalized_test_array, predictions['mean'].values)
    MASE = mean_absolute_scaled_error(normalized_test_array, predictions['mean'].values, y_train=normalized_df['data'][:-TEST].to_numpy())


    print("="*20)
    print(f"Root Mean Squared Error for {task}{subject}{modality}: ", RMSE)
    print(f"Mean Absolute Error for {task}{subject}{modality}: ", MAE)
    print(f"Mean Absolute Scaled Error for {task}{subject}{modality}: ", MASE)
    print("="*20)

    return RMSE, MAE, MASE

"""
for cutoff in range(-num_test_windows * prediction_length, 0, prediction_length):
    for i, ax in enumerate(plt.gcf().axes):
        cutoff_timestamp = test_data.loc[item_id[i]].index[cutoff]
        ax.axvline(cutoff_timestamp, color='gray', linestyle='--')
        plt.show()

"""

def write_data(task, subject, modality, RMSE, MAE, MASE):
    index = subject + task
    columnRMSE = modality + "/RMSE"
    columnMAE = modality + "/MAE"
    columnMASE = modality + "/MASE"
    results_df.at[index,columnRMSE] = RMSE
    results_df.at[index,columnMAE] = MAE
    results_df.at[index,columnMASE] = MASE


# Opening files
datapath = "C:\\Users\\Omistaja\\._.Tampere University\\Thesis\\Code tests\\TimesFMwithElectricdata\\.venv\\scripts\\gluonts_datasets\\"


index_col_array = []
for i in range (1, 9):
        task  = f"T{i}"
        for n in range(1, 29):
            if n == 15:
                continue 
            subject = f"S{n}"
            title = subject + task
            index_col_array.append(title)

results_df = pd.DataFrame(columns= ['TEMP/RMSE', 'TEMP/MAE', 'TEMP/MASE', 'EDA/RMSE', 'EDA/MAE', 'EDA/MASE', 'HR/RMSE', 'HR/MAE', 'HR/MASE'],
                          index= index_col_array)

def count_values(task, subject, modality, url):
    try:
        df = pd.read_csv(url, parse_dates=True)
    except FileNotFoundError:
        df = pd.DataFrame()
    if df.empty:
        return np.nan
    if (len(df) != 2160 and len(df) != 540):
        print("Length of ", task, subject, modality, len(df))

    return len(df)
    


for mod in range(0,3):
    if mod == 0:
        modality = "TEMP"
    elif mod == 1:
        modality = "EDA"
    else:
        modality = "HR"
    for i in range (1, 3):
        task  = f"T{i}"
        for n in range(1, 29):
            if n == 15:
                continue 
            if n == 24:
                if i == 2:
                    continue
            subject = f"S{n}"
            url = datapath + subject + "\\" + subject + task + modality + ".csv"

            # values = count_values(task, subject, modality, url)


            
            try:
                RMSE, MAE, MASE = predict_data(task, subject, modality, url) # T2S24EDA after that it crashes
            except:
                RMSE = np.nan
                MAE = np.nan
                MASE = np.nan
            write_data(task, subject, modality, RMSE, MAE, MASE)
            

print(results_df)
results_df.to_csv(f"{modelname}newtest.csv")