# calculate.py
#
# Calculates AERs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# AER Ranges
SAMPLE_RANGES = [["2025-11-20 00:00:00", "2025-11-21 00:00:00"],
                 ["2025-11-20 00:00:00", "2025-11-21 00:00:00"],
                 ["2025-11-20 00:00:00", "2025-11-21 00:00:00"]]

# AER Locations
SAMPLE_LOCS = ["BH123", "BH456", "BH789"]

# Calibration Equation
CAL_SLOPE =    1.308079
CAL_INTER = -183.813889

# Load Data
data = pd.read_csv("data/co2_logs.csv")

# Rename Columns
data['timestamp'] = pd.to_datetime(data['Timestamp'])
data['co2'] = data['co2_ppm']

# Partition data by ranges
datas = []
for rng in SAMPLE_RANGES:
    start = pd.to_datetime(rng[0])
    end =  pd.to_datetime(rng[1])
    
    data_range = data[
        (data["timestamp"] >= start) &
        (data["timestamp"] <= end)
    ].copy()

    datas.append(data_range)

    # Plot Sensor Data
    plt.figure(figsize=(10,5))
    plt.plot(data_range["timestamp"], data_range["co2"], label="Measured (MH-Z16)")
    plt.xlabel("Time")
    plt.ylabel("CO2 [ppm]")
    plt.grid(True)
    plt.title("Target and Reference Sensor Raw Inputs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("in_timeseries.png")
    plt.show()

for i in range(len(SAMPLE_RANGES)):
    location = SAMPLE_LOCS[i]
    df = datas[i]

    # Grab External
    c_out = 425
    c_1 = df.iloc[0]['co2']
    c_2 = df.iloc[-1]['co2']
    t_1 = df.iloc[0]['timestamp']
    t_2 = df.iloc[-1]['timestamp']
    dt = t_2 - t_1
    dt = dt.total_seconds() / 3600

    aer = (np.log((c_1-c_out) / (c_2-c_out))) / dt
    print(aer)

    



