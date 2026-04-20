import torch
import numpy as np
import pandas as pd
import math



name = "Chronoscov3"

results_df = pd.read_csv("C:\\Users\\Omistaja\\._.Tampere University\\Thesis\\Code tests\\TimesFMwithElectricdata\\Chronoscovariates.csv", index_col=0)
df = pd.read_csv("C:\\Users\\Omistaja\\._.Tampere University\\Thesis\\Code tests\\TimesFMwithElectricdata\\Chronos2.0TEMPwithEDAcovarities.csv")

print("Results_df=", results_df)
mean_values = pd.DataFrame(df.mean(numeric_only=True))


results_df[name] = mean_values
results_df = results_df.round(decimals=4)
results_df.to_csv("Chronoscovariates.csv")
index = mean_values.index

#print(index)
#results_df["Mod/Met"] = index
#results_df[model] = mean_values.iloc[:,[0]]


#results_df.to_csv("Me.csv", index=0)