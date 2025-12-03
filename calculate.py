# calculate.py
#
# Calculates AERs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# AER Ranges
SAMPLE_RANGES = [["2025-11-18 18:10:00", "2025-11-18 22:30:00"],
                 ["2025-11-20 17:30:00", "2025-11-21 06:30:00"],
                 ["2025-11-21 15:20:00", "2025-11-22 05:00:00"]]

# AER Locations
SAMPLE_LOCS = ["BH123", "BH456", "BH789"]

# Calibration Equation
cal = pd.read_csv("calibration_coefficients.csv")
CAL_SLOPE = float(cal["slope"].iloc[0])
CAL_INTER = float(cal["intercept"].iloc[0])

# Load Data
data = pd.read_csv("data/co2_logs.csv")

# Rename Columns and Apply Calibration
data['timestamp'] = pd.to_datetime(data['Timestamp'])
data['cco2'] = data['co2_ppm'] * CAL_SLOPE + CAL_INTER

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
    plt.plot(data_range["timestamp"], data_range["cco2"], label="Corrected (MH-Z16)")
    plt.plot(data_range["timestamp"], data_range["co2_ppm"], label="Measured (MH-Z16)")
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
    c_out = 0
    c_1 = df.iloc[:5]['cco2'].mean()
    c_2 = df.iloc[-5:]['cco2'].mean()    
    t_1 = df.iloc[:5]['timestamp'].mean()
    t_2 = df.iloc[-5:]['timestamp'].mean()
    dt = t_2 - t_1
    dt = dt.total_seconds() / 3600

    # Compute differences
    num = c_1 - c_out
    den = c_2 - c_out

    # Check to make sure inputs make sense and can be passed to log.
    if num <= 0:
        print(f"[{location}] Invalid: c_1={c_1} ppm is not above C_out={c_out}.")
        continue

    if den <= 0:
        print(f"[{location}] Invalid: c_2={c_2} ppm is not above C_out={c_out}.")
        continue

    ratio = num / den
    if ratio <= 0:
        print(f"[{location}] Invalid: ratio={(num/den):.4f} ≤ 0, cannot take log.")
        continue

    aer = (np.log((c_1-c_out) / (c_2-c_out))) / dt
    
    print(location)
    print(aer)
    print(f'c_1 = {c_1}')
    print(f'c_2 = {c_2}')
    print(f't_1 = {t_1}')
    print(f't_2 = {t_2}')

    
