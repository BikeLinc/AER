# calibrate.py
#
# Performs a linear calibration of a target sensor against a reference sensor.
#
#
# Requires:
#   - target.csv
#   - reference.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load Data
target_data = pd.read_csv('target.csv')
reference_data = pd.read_csv('reference.csv')


# Rename Timestamp Headers
target_data['timestamp'] = pd.to_datetime(target_data['Timestamp'])
reference_data['timestamp'] = pd.to_datetime(reference_data['DATE'] + " " + reference_data['TIME'])


# Rename CO2 Headers
target_data['tco2'] = target_data['CO2_ppm']
reference_data['rco2'] = reference_data['CO2']


# Match Timestamps and Merge Data
merged_data = pd.merge_asof(target_data.sort_values('timestamp'),
                            reference_data.sort_values('timestamp'),
                            on='timestamp',
                            direction='nearest',
                            tolerance=pd.Timedelta('1min'),
                            suffixes=('_target', '_reference'))


# Organize Training Data
x = merged_data['tco2']
y = merged_data['rco2']


# Do Regression
a, b = np.polyfit(x, y, 1)


# Compute Metrics
r2 = np.corrcoef(x, y)[0, 1] ** 2
rmse = np.sqrt(np.mean((y - (a * x + b)) ** 2))


# Print Results
print('Calibration Equation:')
print(f'Y = {a} * X + {b}')
