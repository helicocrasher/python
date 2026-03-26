import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def haversine_m(lat1, lon1, lat2, lon2):
    """Compute great-circle distance in meters between GPS points."""
    radius_m = 6371000.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius_m * c


def parse_timestamps(df):
    if 'Date' in df.columns and 'Time' in df.columns:
        return pd.to_datetime(
            df['Date'].astype(str).str.strip() + ' ' + df['Time'].astype(str).str.strip(),
            errors='coerce',
        )
    if 'Time' in df.columns:
        return pd.to_datetime(df['Time'], errors='coerce')
    return pd.Series(pd.NaT, index=df.index)


def format_percent(count, total):
    if total <= 0:
        return '0.00%'
    return f'{(100.0 * count / total):.2f}%'


def _first_true_run(mask, min_run):
    run = 0
    for i, value in enumerate(mask):
        if value:
            run += 1
            if run >= min_run:
                return i - min_run + 1
        else:
            run = 0
    return None


def find_analysis_window(df, timestamp_col, speed_col):
    speed = pd.to_numeric(df[speed_col], errors='coerce')

    smooth_speed = (
        pd.DataFrame({'t': df[timestamp_col], 's': speed})
        .set_index('t')['s']
        .rolling('3s', min_periods=1)
        .mean()
        .to_numpy()
    )

    # Use hysteresis and a minimum run length to reduce false start/stop triggers.
    start_threshold = 10.0
    stop_threshold = 8.0
    min_run = 3

    moving = smooth_speed > start_threshold

    if not np.any(moving):
        return 0, len(df) - 1

    first_moving_pos = _first_true_run(moving, min_run)
    if first_moving_pos is None:
        return 0, len(df) - 1

    moving_positions = np.flatnonzero(moving)
    last_moving_pos = int(moving_positions[-1])

    first_moving_idx = int(df.index[first_moving_pos])
    last_moving_idx = int(df.index[last_moving_pos])

    start_trigger_time = df.loc[first_moving_idx, timestamp_col]
    start_time = start_trigger_time - pd.Timedelta(seconds=3)

    remaining = smooth_speed[last_moving_pos + 1 :]
    stop_run_pos = _first_true_run(remaining < stop_threshold, min_run)
    if stop_run_pos is not None:
        below_pos = last_moving_pos + 1 + stop_run_pos
        end_trigger_time = df.iloc[below_pos][timestamp_col]
    else:
        end_trigger_time = df.loc[last_moving_idx, timestamp_col]

    end_time = end_trigger_time + pd.Timedelta(seconds=3)

    start_candidates = df.index[df[timestamp_col] >= start_time]
    end_candidates = df.index[df[timestamp_col] <= end_time]

    start_idx = start_candidates[0] if len(start_candidates) > 0 else 0
    end_idx = end_candidates[-1] if len(end_candidates) > 0 else len(df) - 1
    return int(start_idx), int(end_idx)


def clean_and_smooth_gps(df, change_log):
    """Validate, repair, and smooth GPS coordinates in-place."""
    if 'GPS' not in df.columns:
        return {
            'gps_rows': 0,
            'invalid_format': 0,
            'invalid_range': 0,
            'invalid_jump': 0,
            'smoothed_points': 0,
        }

    gps_raw = df['GPS'].astype(str).str.strip()
    gps_split = gps_raw.str.split(r'\s+', expand=True)
    lat = pd.to_numeric(gps_split[0], errors='coerce')
    lon = pd.to_numeric(gps_split[1], errors='coerce') if gps_split.shape[1] > 1 else pd.Series(np.nan, index=df.index)

    timestamps = parse_timestamps(df)

    parsed_mask = lat.notna() & lon.notna()
    range_mask = lat.between(-90.0, 90.0) & lon.between(-180.0, 180.0)

    invalid_format_mask = ~parsed_mask
    invalid_range_mask = parsed_mask & ~range_mask

    jump_mask = pd.Series(False, index=df.index)
    speed_check_mask = parsed_mask & range_mask & timestamps.notna()
    speed_check = pd.DataFrame(
        {
            'idx': df.index[speed_check_mask],
            'lat': lat[speed_check_mask],
            'lon': lon[speed_check_mask],
            'ts': timestamps[speed_check_mask],
        }
    )

    if len(speed_check) >= 2:
        speed_check = speed_check.sort_values('ts').reset_index(drop=True)
        max_speed_mps = 300.0 / 3.6
        for i in range(1, len(speed_check)):
            dt = (speed_check.loc[i, 'ts'] - speed_check.loc[i - 1, 'ts']).total_seconds()
            if dt <= 0:
                jump_mask.loc[speed_check.loc[i, 'idx']] = True
                continue

            dist = haversine_m(
                speed_check.loc[i - 1, 'lat'],
                speed_check.loc[i - 1, 'lon'],
                speed_check.loc[i, 'lat'],
                speed_check.loc[i, 'lon'],
            )
            if (dist / dt) > max_speed_mps:
                jump_mask.loc[speed_check.loc[i, 'idx']] = True

    clean_mask = parsed_mask & range_mask & ~jump_mask

    lat_clean = lat.where(clean_mask)
    lon_clean = lon.where(clean_mask)

    # Fill gaps introduced by invalid points so all rows keep a usable GPS coordinate.
    lat_interp = lat_clean.interpolate(limit_direction='both')
    lon_interp = lon_clean.interpolate(limit_direction='both')

    lat_smooth = lat_interp.copy()
    lon_smooth = lon_interp.copy()

    smoothed_count = 0
    for i in range(1, len(df) - 1):
        if pd.notna(lat_interp.iat[i - 1]) and pd.notna(lat_interp.iat[i]) and pd.notna(lat_interp.iat[i + 1]):
            lat_smooth.iat[i] = 0.25 * lat_interp.iat[i - 1] + 0.5 * lat_interp.iat[i] + 0.25 * lat_interp.iat[i + 1]
            lon_smooth.iat[i] = 0.25 * lon_interp.iat[i - 1] + 0.5 * lon_interp.iat[i] + 0.25 * lon_interp.iat[i + 1]
            smoothed_count += 1

    invalid_rows = df.index[invalid_format_mask | invalid_range_mask | jump_mask]
    for idx in invalid_rows:
        reasons = []
        if invalid_format_mask.loc[idx]:
            reasons.append('invalid-format')
        if invalid_range_mask.loc[idx]:
            reasons.append('out-of-range')
        if jump_mask.loc[idx]:
            reasons.append('impossible-jump')
        repaired_value = f'{lat_smooth.loc[idx]:.6f} {lon_smooth.loc[idx]:.6f}'
        change_log.append(
            {
                'Row_Number': int(df.at[idx, '_source_row']) + 1 if '_source_row' in df.columns else int(idx) + 1,
                'Column': 'GPS',
                'Original_Value': df.at[idx, 'GPS'],
                'Replacement_Value': f'{repaired_value} [reason={"|".join(reasons)}]',
            }
        )

    df['GPS'] = lat_smooth.map(lambda x: f'{x:.6f}') + ' ' + lon_smooth.map(lambda x: f'{x:.6f}')

    gps_rows = int(parsed_mask.sum())
    invalid_format_count = int(invalid_format_mask.sum())
    invalid_range_count = int(invalid_range_mask.sum())
    invalid_jump_count = int(jump_mask.sum())

    summary_rows = [
        ('GPS_WARNING_INVALID_FORMAT', invalid_format_count),
        ('GPS_WARNING_OUT_OF_RANGE', invalid_range_count),
        ('GPS_WARNING_IMPOSSIBLE_JUMP', invalid_jump_count),
        ('GPS_SMOOTHED_POINTS', int(smoothed_count)),
    ]
    for label, count in summary_rows:
        change_log.append(
            {
                'Row_Number': np.nan,
                'Column': label,
                'Original_Value': count,
                'Replacement_Value': format_percent(count, len(df)),
            }
        )

    return {
        'gps_rows': gps_rows,
        'invalid_format': invalid_format_count,
        'invalid_range': invalid_range_count,
        'invalid_jump': invalid_jump_count,
        'smoothed_points': int(smoothed_count),
    }


parser = argparse.ArgumentParser(description='Repair outliers in a CSV file.')
parser.add_argument('file_name', help='Path to the CSV file to repair')
args = parser.parse_args()

input_path = Path(args.file_name)
if not input_path.is_absolute():
    script_relative_path = Path(__file__).resolve().parent / input_path
    if script_relative_path.exists():
        input_path = script_relative_path

original_suffix = '_original'
if input_path.stem.endswith(original_suffix):
    original_path = input_path
    output_stem = input_path.stem[:-len(original_suffix)]
    output_path = input_path.with_name(f'{output_stem}{input_path.suffix}')
else:
    original_path = input_path.with_name(f'{input_path.stem}{original_suffix}{input_path.suffix}')
    output_path = input_path
    if original_path.exists():
        raise FileExistsError(
            f'Cannot rename {input_path.name} to {original_path.name} because the destination already exists.'
        )
    input_path.rename(original_path)

details_path = output_path.with_name(f'{output_path.stem}_repair_details{output_path.suffix}')

# Load your file
df = pd.read_csv(original_path)

original_total_rows = len(df)
trim_start_count = 0
trim_end_count = 0

df['_source_row'] = np.arange(len(df), dtype=int)
timestamps = parse_timestamps(df)

if 'GSpd(kmh)' in df.columns and timestamps.notna().any():
    timed_df = df[timestamps.notna()].copy()
    timed_df['_timestamp'] = timestamps[timestamps.notna()]
    timed_df = timed_df.sort_values('_timestamp').reset_index(drop=True)

    start_idx, end_idx = find_analysis_window(timed_df, '_timestamp', 'GSpd(kmh)')
    trim_start_count = int(start_idx)
    trim_end_count = int(len(timed_df) - 1 - end_idx)
    df = timed_df.loc[start_idx:end_idx].copy()
else:
    df['_timestamp'] = timestamps

change_log = [
    {
        'Row_Number': np.nan,
        'Column': 'TRIM_START_ROWS_REMOVED',
        'Original_Value': trim_start_count,
        'Replacement_Value': format_percent(trim_start_count, original_total_rows),
    },
    {
        'Row_Number': np.nan,
        'Column': 'TRIM_END_ROWS_REMOVED',
        'Original_Value': trim_end_count,
        'Replacement_Value': format_percent(trim_end_count, original_total_rows),
    },
]

# --- CONFIGURATION ---
window_size = 10 
threshold = 3 

# Select numeric columns
exclude_numeric = {'_source_row'}
numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_numeric]

# Rolling medians can produce decimal values, so convert numeric columns to float
# before writing repaired values back into the dataframe.
df[numeric_cols] = df[numeric_cols].astype(float)

for col in numeric_cols:
    # Calculate rolling median and standard deviation
    rolling_median = df[col].rolling(window=window_size, center=True).median()
    rolling_std = df[col].rolling(window=window_size, center=True).std()
    
    # Identify the specific outliers
    is_outlier = (np.abs(df[col] - rolling_median) > (threshold * rolling_std))
    
    # Capture the row indices where outliers were found
    outlier_indices = df.index[is_outlier].tolist()
    
    for idx in outlier_indices:
        change_log.append({
            'Row_Number': int(df.at[idx, '_source_row']) + 1,
            'Column': col,
            'Original_Value': df.at[idx, col],
            'Replacement_Value': rolling_median[idx]
        })
    
    # Replace outliers with the local median
    df.loc[is_outlier, col] = rolling_median[is_outlier]

gps_stats = clean_and_smooth_gps(df, change_log)

# Final cleanup for edge cases
df = df.ffill().bfill()

if '_source_row' in df.columns:
    df = df.drop(columns=['_source_row'])
if '_timestamp' in df.columns:
    df = df.drop(columns=['_timestamp'])

# Save the repaired data
df.to_csv(output_path, index=False)

# Save the change log as a separate CSV
if change_log:
    log_df = pd.DataFrame(change_log)
    log_df.to_csv(details_path, index=False)
    print(f"Repair complete! {len(change_log)} values were corrected.")
    print(f"Original file: {original_path.name}")
    print(f"Repaired file: {output_path.name}")
    print(f"Change log: {details_path.name}")
    print('Trim summary:')
    print(f"  start removed: {trim_start_count} ({format_percent(trim_start_count, original_total_rows)})")
    print(f"  end removed: {trim_end_count} ({format_percent(trim_end_count, original_total_rows)})")
    print('GPS warnings:')
    print(
        f"  invalid format: {gps_stats['invalid_format']} ({format_percent(gps_stats['invalid_format'], len(df))})"
    )
    print(
        f"  out of range: {gps_stats['invalid_range']} ({format_percent(gps_stats['invalid_range'], len(df))})"
    )
    print(
        f"  impossible jump: {gps_stats['invalid_jump']} ({format_percent(gps_stats['invalid_jump'], len(df))})"
    )
    print(
        f"  smoothed points: {gps_stats['smoothed_points']} ({format_percent(gps_stats['smoothed_points'], len(df))})"
    )
else:
    print("No outliers detected with the current settings.")
    print(f"Original file: {original_path.name}")
    print(f"Repaired file: {output_path.name}")
