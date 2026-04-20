import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timesfm
from datetime import datetime, timedelta

torch.set_float32_matmul_precision("high")
plt.figure(figsize=(15,7))
df = pd.DataFrame(columns = ["data", "index"])

def create_timestamp(length_of_time, id):
    start = datetime(2020, 1, 1, 0, 0, 0, 0)
    times = pd.DataFrame({"timestamp": pd.date_range(start, periods=length_of_time, freq='250ms')})
    timestamp = start
    timedata = []
    item_id = []
    for i in range(length_of_time):
        timedata.append(timestamp)
        timestamp += timedelta(seconds=0.25)
        item_id.append(id)
    item_id = pd.DataFrame({"unique_id": item_id})
    return times, item_id


def read_data(url):
    # Reading data into cvs files and renaming the columns
    data_EDA = pd.read_csv(f"{url}EDA.csv", usecols=[0])
    data_EDA.columns = ['EDA']
    data_TEMP = pd.read_csv(f"{url}TEMP.csv", usecols=[0])
    data_TEMP.columns = ['TEMP']
    data_HR = pd.read_csv(f"{url}HR.csv", usecols=[0])
    data_HR.columns = ['HR']
    return data_EDA, data_TEMP, data_HR
    #data_HR = pd.DataFrame({'HR': pd.Series(dtype='float')})


# Heart rate is sampled once per second, while others are sampled
# four times per second -extending the dataset to have every value
# four times, to be compatible with other data
#for i in range(len(data_HR_short)):
#    for n in range(4):
#        data_HR.loc[len(data_HR)] = {"HR": data_HR_short['HR_short'].loc[data_HR_short.index[i]]}

#print(data_HR)

def write_data(data_EDA, data_TEMP, data_HR, subject, task):
    timedata_EDA, unique_id = create_timestamp(len(data_EDA), task)
    timedata_TEMP, unique_id = create_timestamp(len(data_TEMP), task)
    timedata_HR, unique_id = create_timestamp(len(data_HR), task)
    #df_T2 = pd.concat([timedata, data_TEMP, data_HR, data_EDA], axis = 1)
    df_T2EDA = pd.concat([timedata_EDA, data_EDA], axis = 1)
    #df_T2.to_csv("T2combined.csv", index=False)
    df_T2TEMP = pd.concat([timedata_TEMP, data_TEMP], axis = 1)
    df_T2HR = pd.concat([timedata_HR, data_HR], axis = 1)
    df_T2EDA.to_csv(f"{subject}{task}EDA.csv", index=False)
    df_T2HR.to_csv(f"{subject}{task}HR.csv", index=False)
    df_T2TEMP.to_csv(f"{subject}{task}TEMP.csv", index=False)


# Defining
datapath = "C:\\Users\\Omistaja\\._.Tampere University\\Thesis\\Code tests\\TimesFMwithElectricdata\\Raw_data\\"
subject = "S18" 
folder = "\\S18\\R\\"

for i in range(1,9):
    task = f"T{i}"
    url = datapath + subject + folder + task + "\\" + task
    data_EDA, data_TEMP, data_HR = read_data(url)
    write_data(data_EDA, data_TEMP, data_HR, subject, task)

#C:\Users\Omistaja\._.Tampere University\Thesis\Code tests\TimesFMwithElectricdata\Raw_data\S18\S18\R\T1\T1EDA.csv
    
#task = "T2"

"""
plt.subplot(1, 2, 1)
# Reading the emphatic school data
for n in range(1,9):
    data = pd.read_csv(f'T{n}hr.csv')
    time = time = pd.to_timedelta(np.arange(0, len(data), 1), 'sec')
    index = n
    plt.plot(time, data, label=f"Heart rate {n}")
    df_n = pd.DataFrame({"data": [data], "index": [index]})
    df[index] = data

df.to_csv("T1-8hr.csv",index=False )
print(df)
plt.legend()
plt.title("Plot for 8 heart rates")
plt.xlabel("Time")
plt.ylabel("Value")

"""
"""
plt.subplot(1, 2, 2)
for n in range(1,9):
    data = pd.read_csv(f'C:\\Users\\Omistaja\\._.Tampere University\\Thesis\\Code tests\\TimesFMwithElectricdata\\S27\\1\\1\\T{n}EDA.csv')
    time = time = pd.to_timedelta(np.arange(0, len(data), 1), 'sec')
    plt.plot(time, data, label=f"EDA {n}")
plt.legend()
plt.title("Plot for 8 EDA's")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
"""


