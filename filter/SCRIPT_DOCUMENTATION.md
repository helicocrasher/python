# Filter Scripts Documentation

This document describes the three Python scripts in this folder:

- `filt_log.py`
- `flight_stats.py`
- `process_gps.py`

## 1) Setup

These scripts use Python with `pandas` and `numpy`.

Example (from repository root):

```powershell
& ".\.venv\Scripts\python.exe" -m pip install pandas numpy
```

## 2) filt_log.py

### Purpose

Repairs a flight log CSV by:

- Renaming the input file to `<name>_original.csv` (if needed)
- Writing the repaired output back to `<name>.csv`
- Writing a repair log to `<name>_repair_details.csv`
- Trimming start/end idle rows using movement detection
- Replacing numeric outliers with rolling median values
- Validating and repairing GPS points
- Smoothing GPS path

### Usage

```powershell
python .\filt_log.py <input_csv>
```

Example:

```powershell
python .\filt_log.py .\logfiles2analize\Attacko 4m-2025-09-17-132817.csv
```

### Input expectations

- CSV file path is required as positional argument.
- Script supports relative paths.
- For movement-window trimming, these columns should exist:
  - `Time` (or `Date` + `Time`)
  - `GSpd(kmh)`
- For GPS cleanup/smoothing, `GPS` column should exist.

### Output files

Given `mylog.csv`:

- Original: `mylog_original.csv`
- Repaired: `mylog.csv`
- Details: `mylog_repair_details.csv`

### Important behavior

- If an input already ends with `_original`, it is treated as original and output is written to base name.
- If destination `_original` file already exists during rename, the script stops with an error.
- Repair details include:
  - Trim summary (`TRIM_START_ROWS_REMOVED`, `TRIM_END_ROWS_REMOVED`)
  - Per-cell outlier fixes
  - GPS warning summaries

## 3) flight_stats.py

### Purpose

Reads a log CSV and prints statistics to terminal, including:

- Time window and duration
- Altitude relative to start
- Ground speed
- Optional satellites (`Sats`)
- Optional airspeed (`ASpd(kmh)`), with negative values clipped to 0
- Optional receiver voltage (`RxV(V)` or `RxBt(V)`)
- Optional battery voltage (`Batt(V)`, `EscV(V)`, or `A2(V)`)
- Optional link quality (`TQly` or `TQly(%)`)
- Optional current (`EscA(A)`, `ESCA(A)`, or `Curr(A)`)
- Capacity estimate (mAh)
- GPS distance metrics

### Usage

```powershell
python .\flight_stats.py <input_csv>
```

Example:

```powershell
python .\flight_stats.py .\logfiles2analize\Attacko 4m-2025-09-17-132817.csv
```

### Required columns

- `Time`
- `GSpd(kmh)`
- `Alt(m)`
- `GPS`

### Optional columns

- `Sats`
- `ASpd(kmh)`
- `RxV(V)` or `RxBt(V)`
- `Batt(V)` or `EscV(V)` or `A2(V)`
- `TQly` or `TQly(%)`
- `EscA(A)` or `ESCA(A)` or `Curr(A)`

### Notes

- GPS points are range-checked and filtered for impossible jumps.
- Analysis window is based on movement detection from `GSpd(kmh)`.
- Output is printed to stdout only (no file written).

## 4) process_gps.py

### Purpose

Extracts GPS-related fields into a clean CSV for external tools (for example gpsbabel workflows).

### Usage

Default output name:

```powershell
python .\process_gps.py <input_csv>
```

Custom output path:

```powershell
python .\process_gps.py <input_csv> -o <output_csv>
```

Example:

```powershell
python .\process_gps.py .\logfiles2analize\Attacko 4m-2025-09-17-132817.csv
```

### Required columns

- `Time`
- `GPS`
- `GAlt(m)`
- `GSpd(kmh)`

### Output columns

- `Time`
- `Latitude`
- `Longitude`
- `Altitude`
- `Speed`

Rows with invalid GPS latitude/longitude are removed.

## 5) Typical workflow

1. Run `filt_log.py` on raw CSV to clean and trim data.
2. Run `flight_stats.py` on cleaned CSV to inspect flight metrics.
3. Run `process_gps.py` when you need a GPS-only export file.
