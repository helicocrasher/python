import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def haversine_m(lat1, lon1, lat2, lon2):
    """Compute great-circle distance in meters between two GPS points."""
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


def filter_valid_gps_points(lat, lon, timestamps):
    """Keep only plausible GPS points and remove impossible jump outliers."""
    valid_range = lat.between(-90.0, 90.0) & lon.between(-180.0, 180.0)
    valid_time = timestamps.notna()
    points = pd.DataFrame({"lat": lat, "lon": lon, "t": timestamps})[valid_range & valid_time].copy()

    if len(points) < 2:
        return points

    keep = np.ones(len(points), dtype=bool)
    lat_np = points["lat"].to_numpy(dtype=float)
    lon_np = points["lon"].to_numpy(dtype=float)
    t_np = points["t"].astype("int64").to_numpy(dtype=float) / 1e9

    # Drop points that imply impossible movement speeds between consecutive samples.
    # 300 km/h is a conservative upper bound for these logs.
    max_speed_mps = 300.0 / 3.6
    for i in range(1, len(points)):
        dt = t_np[i] - t_np[i - 1]
        if dt <= 0:
            keep[i] = False
            continue

        dist = haversine_m(lat_np[i - 1], lon_np[i - 1], lat_np[i], lon_np[i])
        if dist / dt > max_speed_mps:
            keep[i] = False

    return points[keep].reset_index(drop=True)


def parse_datetime(df):
    if "Date" in df.columns and "Time" in df.columns:
        return pd.to_datetime(
            df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
            errors="coerce",
        )
    if "Time" in df.columns:
        return pd.to_datetime(df["Time"], errors="coerce")
    return pd.Series(pd.NaT, index=df.index)


def parse_gps(df):
    if "GPS" not in df.columns:
        return pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)

    gps_split = df["GPS"].astype(str).str.strip().str.split(r"\s+", expand=True)
    lat = pd.to_numeric(gps_split[0], errors="coerce")
    lon = pd.to_numeric(gps_split[1], errors="coerce") if gps_split.shape[1] > 1 else np.nan
    return lat, lon


def find_analysis_window(df, timestamp_col, speed_col):
    speed = pd.to_numeric(df[speed_col], errors="coerce")
    moving = speed > 10.0

    if not moving.any():
        start_idx = 0
        end_idx = len(df) - 1
        return start_idx, end_idx

    first_moving_idx = moving[moving].index[0]
    last_moving_idx = moving[moving].index[-1]

    start_trigger_time = df.loc[first_moving_idx, timestamp_col]
    start_time = start_trigger_time - pd.Timedelta(seconds=3)

    below_after_last = (speed < 10.0) & (df.index > last_moving_idx)
    if below_after_last.any():
        below_idx = below_after_last[below_after_last].index[0]
        end_trigger_time = df.loc[below_idx, timestamp_col]
    else:
        end_trigger_time = df.loc[last_moving_idx, timestamp_col]

    end_time = end_trigger_time + pd.Timedelta(seconds=3)

    start_candidates = df.index[df[timestamp_col] >= start_time]
    end_candidates = df.index[df[timestamp_col] <= end_time]

    start_idx = start_candidates[0] if len(start_candidates) > 0 else df.index[0]
    end_idx = end_candidates[-1] if len(end_candidates) > 0 else df.index[-1]

    return start_idx, end_idx


def format_duration(td):
    total_seconds = int(td.total_seconds())
    if total_seconds < 0:
        total_seconds = 0
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_min_avg_max(series, unit=""):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "n/a"
    suffix = f" {unit}" if unit else ""
    return f"min={s.min():.2f}{suffix}, avg={s.mean():.2f}{suffix}, max={s.max():.2f}{suffix}"


def compute_capacity_mah(currents_a, timestamps):
    valid = currents_a.notna() & timestamps.notna()
    if valid.sum() < 2:
        return 0.0

    cur = currents_a[valid].to_numpy(dtype=float)
    ts = timestamps[valid].astype("int64").to_numpy(dtype=float) / 1e9
    dt = np.diff(ts)
    dt_hours = dt / 3600.0

    avg_current = (cur[:-1] + cur[1:]) / 2.0
    ah = np.sum(avg_current * dt_hours)
    return ah * 1000.0


def find_first_existing_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None


def main():
    parser = argparse.ArgumentParser(description="Analyze a GPS logfile CSV and print ride statistics.")
    parser.add_argument("file_name", help="Path to input CSV file")
    args = parser.parse_args()

    input_path = Path(args.file_name)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        raise SystemExit(1)

    df = pd.read_csv(input_path)

    required_columns = ["Time", "GSpd(kmh)", "Alt(m)", "GPS"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        print("Error: missing required columns: " + ", ".join(missing))
        raise SystemExit(1)

    df["_timestamp"] = parse_datetime(df)
    valid_time = df["_timestamp"].notna()
    df = df[valid_time].copy()

    if df.empty:
        print("Error: no valid timestamps found in Date/Time columns.")
        raise SystemExit(1)

    df = df.sort_values("_timestamp").reset_index(drop=True)
    df["_lat"], df["_lon"] = parse_gps(df)

    start_idx, end_idx = find_analysis_window(df, "_timestamp", "GSpd(kmh)")
    window = df.loc[start_idx:end_idx].copy()

    if window.empty:
        print("Error: computed analysis window is empty.")
        raise SystemExit(1)

    start_time = window["_timestamp"].iloc[0]
    end_time = window["_timestamp"].iloc[-1]
    duration = end_time - start_time

    alt = pd.to_numeric(window["Alt(m)"], errors="coerce")
    alt_rel = alt - alt.iloc[0] if not alt.dropna().empty else alt

    speed = pd.to_numeric(window["GSpd(kmh)"], errors="coerce")
    air_speed = pd.to_numeric(window["ASpd(kmh)"], errors="coerce").clip(lower=0) if "ASpd(kmh)" in window.columns else None
    sats = pd.to_numeric(window["Sats"], errors="coerce") if "Sats" in window.columns else None
    batt_col = find_first_existing_column(window, ["Batt(V)", "EscV(V)", "A2(V)"])
    batt_voltage = pd.to_numeric(window[batt_col], errors="coerce") if batt_col is not None else None
    rx_col = find_first_existing_column(window, ["RxV(V)", "RxBt(V)"])
    rx_voltage = pd.to_numeric(window[rx_col], errors="coerce") if rx_col is not None else None
    tqly_col = find_first_existing_column(window, ["TQly", "TQly(%)"])
    tqly = pd.to_numeric(window[tqly_col], errors="coerce") if tqly_col is not None else None
    current_col = find_first_existing_column(window, ["EscA(A)", "ESCA(A)", "Curr(A)"])
    current = pd.to_numeric(window[current_col], errors="coerce") if current_col is not None else None

    if current is not None:
        rolling_current = (
            pd.DataFrame({"t": window["_timestamp"], "i": current})
            .set_index("t")["i"]
            .rolling("3s", min_periods=1)
            .mean()
        )
        max_avg_current_3s = rolling_current.max() if not rolling_current.empty else np.nan
        capacity_mah = compute_capacity_mah(current, window["_timestamp"])
    else:
        max_avg_current_3s = np.nan
        capacity_mah = np.nan

    lat = pd.to_numeric(window["_lat"], errors="coerce")
    lon = pd.to_numeric(window["_lon"], errors="coerce")
    gps_points = filter_valid_gps_points(lat, lon, window["_timestamp"])

    total_distance_m = 0.0
    max_from_start_avg_m = 0.0

    if len(gps_points) >= 2:
        lat_v = gps_points["lat"]
        lon_v = gps_points["lon"]

        seg_dist = haversine_m(
            lat_v.iloc[:-1].to_numpy(),
            lon_v.iloc[:-1].to_numpy(),
            lat_v.iloc[1:].to_numpy(),
            lon_v.iloc[1:].to_numpy(),
        )
        total_distance_m = float(np.nansum(seg_dist))

        start_avg_cutoff = start_time + pd.Timedelta(seconds=3)
        start_zone = gps_points["t"] <= start_avg_cutoff
        if start_zone.any():
            start_lat = gps_points.loc[start_zone, "lat"].mean()
            start_lon = gps_points.loc[start_zone, "lon"].mean()
        else:
            start_lat = lat_v.iloc[0]
            start_lon = lon_v.iloc[0]

        dist_from_start = haversine_m(
            start_lat,
            start_lon,
            lat_v.to_numpy(),
            lon_v.to_numpy(),
        )
        max_from_start_avg_m = float(np.nanmax(dist_from_start))

    print("Logfile Analysis")
    print("================")
    print(f"Input file: {input_path}")
    print()
    print("Time")
    print(f"  Start:    {start_time}")
    print(f"  End:      {end_time}")
    print(f"  Duration: {format_duration(duration)}")
    print()
    print("Alt(m) above starting point")
    print(f"  {format_min_avg_max(alt_rel, 'm')}")
    print()
    print("GSpd(kmh)")
    print(f"  {format_min_avg_max(speed, 'km/h')}")
    print()
    print("Sats")
    if sats is None:
        print("  n/a (column not present)")
    else:
        print(f"  {format_min_avg_max(sats)}")
    print()
    print("ASpd(kmh) (negative values clipped to 0)")
    if air_speed is None:
        print("  n/a (column not present)")
    else:
        print(f"  {format_min_avg_max(air_speed, 'km/h')}")
    print()
    print("RxV(V)")
    if rx_voltage is None:
        print("  n/a (column not present)")
    else:
        print(f"  Source column: {rx_col}")
        print(f"  {format_min_avg_max(rx_voltage, 'V')}")
    print()
    print("Batt(V) / EscV(V) / A2(V)")
    if batt_voltage is None:
        print("  n/a (column not present)")
    else:
        print(f"  Source column: {batt_col}")
        print(f"  {format_min_avg_max(batt_voltage, 'V')}")
    print()
    print("TQly")
    if tqly is None:
        print("  n/a (column not present)")
    else:
        print(f"  Source column: {tqly_col}")
        print(f"  {format_min_avg_max(tqly)}")
    print()
    print("EscA(A)")
    if current_col is None:
        print("  Current sensor: n/a (EscA(A), ESCA(A), Curr(A) not present)")
    else:
        print(f"  Current sensor column: {current_col}")
    if pd.isna(max_avg_current_3s):
        print("  Max 3s average current: n/a")
    else:
        print(f"  Max 3s average current: {max_avg_current_3s:.2f} A")
    if pd.isna(capacity_mah):
        print("  Capacity used: n/a")
    else:
        print(f"  Capacity used: {capacity_mah:.2f} mAh")
    print()
    print("GPS")
    print(f"  Total distance travelled: {total_distance_m / 1000.0:.3f} km")
    print(f"  Max distance from average start point: {max_from_start_avg_m:.1f} m")


if __name__ == "__main__":
    main()
