# calibrate.py
# 
# Linear calibration of target sensor to reference sensor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Calibration Window and Limits
CAL_START = pd.to_datetime("2025-12-01 14:30:00")
CAL_END   = pd.to_datetime("2025-12-02 11:00:00")
LICOR_LIM = 3000

# Load Calibration Data
target_data    = pd.read_csv("cal/co2_logs.csv")
reference_data = pd.read_csv("cal/licor.data",
                             delimiter="\t")

# Licor has weird timestamps, create one from two columns.
target_data["timestamp"] = pd.to_datetime(target_data["Timestamp"], errors="coerce")
reference_data["timestamp"] = pd.to_datetime(
    reference_data["DATE"].astype(str) + " " + reference_data["TIME"].astype(str),
    errors="coerce"
)

# Sensor CO2 -> tco2, Licor CO2 -> rco2
target_data["tco2"] = target_data["co2_ppm"]
reference_data["rco2"] = reference_data["CO2"]

# Clean Data
target_data = target_data.dropna(subset=["timestamp"])
reference_data = reference_data.dropna(subset=["timestamp"])

# Plot Sensor Data
plt.figure(figsize=(10,5))
plt.plot(target_data["timestamp"], target_data["tco2"], label="Target (MH-Z16)")
plt.plot(reference_data["timestamp"], reference_data["rco2"], label="Reference (LI-COR 7810)")
plt.xlabel("Time")
plt.ylabel("CO2 [ppm]")
plt.grid(True)
plt.title("Target and Reference Sensor Raw Inputs")
plt.legend()
plt.tight_layout()
plt.savefig("in_timeseries.png")
plt.show()

# Apply calibration window
tdata = target_data[
    (target_data["timestamp"] >= CAL_START) &
    (target_data["timestamp"] <= CAL_END)
].copy()

rdata = reference_data[
    (reference_data["timestamp"] >= CAL_START) &
    (reference_data["timestamp"] <= CAL_END)
].copy()

# Complain if data is not present
if len(tdata) == 0 or len(rdata) == 0:
    raise RuntimeError("No data found inside calibration window.")


# Print whats going in
print("")
print("Using calibration window:", CAL_START, "to", CAL_END)
print("Target points:", len(tdata))
print("Reference points:", len(rdata))
print("")

# I think they are sorted, but...
tdata = tdata.sort_values("timestamp")
rdata = rdata.sort_values("timestamp")

# Merge datasets
mdata = pd.merge_asof(
    tdata,
    rdata,
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("30s")
)

# Check for NANs and drop columns with LICOR above LICOR_LIM
mdata = mdata.dropna(subset=["rco2"])
mdata = mdata[mdata['rco2'] <= LICOR_LIM]

# Complain if merge is bad
if len(mdata) < 2:
    raise RuntimeError("Not enough matched points after merge.")

### Training Calibration Model
x = mdata["tco2"].values
y = mdata["rco2"].values
a, b = np.polyfit(x, y, 1)
y_fit = a * x + b

# Metrics
r_value = np.corrcoef(x, y)[0,1]
r2 = r_value * r_value
rmse = np.sqrt(np.mean((y - y_fit) ** 2))

print("Calibration Results")
print("Equation: y = %.6f * x + %.6f" % (a, b))
print("R^2:", r2)
print("RMSE:", rmse, "ppm")

# 1 to 1 plpt
plt.figure(figsize=(7,5))
plt.scatter(x, y, s=14, alpha=0.7, label="Data")
plt.plot(np.sort(x), a*np.sort(x) + b, "r-", lw=2, label="Fit")
plt.xlabel("MH-Z16 CO2 [ppm]")
plt.ylabel("LI-COR 7810 CO2 [ppm]")
plt.title("Calibration Plot")
plt.grid(True)
eq_text = "y = %.3f*x + %.2f\nR^2 = %.4f\nRMSE = %.2f ppm" % (a, b, r2, rmse)
plt.text(
    0.05, 0.95, eq_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9)
)
plt.legend()
plt.tight_layout()
plt.savefig("calibration_fit.png")
plt.show()

### Timeseries Plot
plt.figure(figsize=(10,5))
plt.plot(tdata["timestamp"], tdata["tco2"], "b.", alpha=0.3, label="Target (All in Window)")
plt.plot(rdata["timestamp"], rdata["rco2"], "g.", alpha=0.3, label="Reference (All in Window)")
plt.scatter(mdata["timestamp"], mdata["tco2"], s=20, color="blue", label="Target (Used)")
plt.scatter(mdata["timestamp"], mdata["rco2"], s=20, color="green", label="Reference (Used)")

# Calibration window boundaries
plt.axvline(CAL_START, color="gray", linestyle="--", alpha=0.7)
plt.axvline(CAL_END,   color="gray", linestyle="--", alpha=0.7)
plt.title("Training Data Used for Calibration")
plt.xlabel("Time")
plt.ylabel("CO2 [ppm]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("training_data_used.png")
plt.show()



### Residuals (y - y_fit)
residuals = y - y_fit
mu = np.mean(residuals)
sigma = np.std(residuals)

# 95% confidence bounds
ci_upper = mu + 1.96 * sigma
ci_lower = mu - 1.96 * sigma

### Residuals vs CO2
plt.figure(figsize=(7,5))
plt.scatter(x, residuals, s=14, alpha=0.7, label="Residuals")
plt.axhline(mu, color="red", linestyle="--", linewidth=2, label="Mean Residual")
plt.axhline(ci_upper, color="green", linestyle=":", linewidth=2, label="95% CI")
plt.axhline(ci_lower, color="green", linestyle=":", linewidth=2)

plt.xlabel("MH-Z16 CO2 [ppm]")
plt.ylabel("Residual (LI-COR - Fit) [ppm]")
plt.title("Residuals vs MH-Z16 CO2")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("residuals_vs_co2.png")
plt.show()

### Residuals Histogram
plt.figure(figsize=(7,5))
plt.hist(residuals, bins=40, alpha=0.7, color="gray")
plt.axvline(mu, color="red", linestyle="--", linewidth=2, label="Mean")
plt.axvline(ci_upper, color="green", linestyle=":", linewidth=2, label="95% CI")
plt.axvline(ci_lower, color="green", linestyle=":", linewidth=2)

plt.title("Residuals Histogram")
plt.xlabel("Residual [ppm]")
plt.ylabel("Count")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("residual_histogram.png")
plt.show()

### Residuals Violin Plot
plt.figure(figsize=(5,6))
plt.violinplot(residuals, showmeans=True, showextrema=True)
plt.axhline(0, color="black", linestyle="--", linewidth=1)

plt.title("Residuals Violin Plot")
plt.ylabel("Residual [ppm]")
plt.tight_layout()
plt.savefig("residual_violin.png")
plt.show()




### Save Calibration Coefficients
cal_out = pd.DataFrame({
    "slope":     [a],
    "intercept": [b],
    "R2":        [r2],
    "RMSE":      [rmse]
})

# save to csv
cal_out.to_csv("calibration_coefficients.csv", index=False)
print("Calibration coefficients saved to calibration_coefficients.csv")
