# ach.py
#
# This script calculates the ACH (Air Changes per Hour) for a given room.
#
# Requirements:
#   - A Calibration Coefficient
#   - B Calibration Coefficient
#   - target.csv


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Input Params
A = 0.5
B = 100

INPUT_CSV = 'target.csv'
OUTPUT_CSV = 'ach_results.csv'

DATE_START = '2023-01-01 00:00:00'
DATE_END = '2023-01-02 00:00:00'


# Import Data
df = pd.read_csv(INPUT_CSV)
df['timestamp'] = pd.to_datetime(df['Timestamp'])

# Dates
start = pd.to_datetime(DATE_START)
end = pd.to_datetime(DATE_END)
mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
df = df.loc[mask]

# TODO: Implement a exponential decay selector.

# TODO: Implement the ACH calculation.

# TODO: Implement plotting of results.
