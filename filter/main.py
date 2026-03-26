import pandas as pd
import numpy as np

# Load your file
df = pd.read_csv('your_log_file.csv')

# --- CONFIGURATION ---
# window_size: How many neighboring rows to look at (e.g., 5-10)
# threshold: How many standard deviations away a point must be to be an outlier
window_size = 10 
threshold = 3

def is_local_outlier(series, window, k):
    # Calculate rolling median and standard deviation
    rolling_median = series.rolling(window=window, center=True).median()
    rolling_std = series.rolling(window=window, center=True).std()
    
    # Identify values that are too far from their local neighbors
    outliers = (np.abs(series - rolling_median) > (k * rolling_std))
    return outliers

# Select numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
row_outlier_mask = pd.Series(False, index=df.index)

for col in numeric_cols:
    # Build a mask of rows where this specific column has a local spike
    col_outliers = is_local_outlier(df[col], window_size, threshold)
    row_outlier_mask = row_outlier_mask | col_outliers.fillna(False)

# Keep only rows that are NOT local outliers
clean_df = df[~row_outlier_mask]

# Save the result
clean_df.to_csv('local_filtered_logs.csv', index=False)
print(f"Removed {row_outlier_mask.sum()} rows containing local spikes.")
