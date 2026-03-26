import argparse
from pathlib import Path

import pandas as pd


def parse_gps_column(series):
    gps_parts = series.astype(str).str.strip().str.split(r"\s+", expand=True)
    lat = pd.to_numeric(gps_parts[0], errors="coerce")
    if gps_parts.shape[1] > 1:
        lon = pd.to_numeric(gps_parts[1], errors="coerce")
    else:
        lon = pd.Series(pd.NA, index=series.index, dtype="float64")
    return lat, lon


def main():
    parser = argparse.ArgumentParser(
        description="Extract GPS-related columns from a logfile CSV for gpsbabel processing."
    )
    parser.add_argument("file_name", help="Path to input logfile CSV")
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output CSV path. Default: <input_stem>_gps.csv next to input file.",
    )
    args = parser.parse_args()

    input_path = Path(args.file_name)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        raise SystemExit(1)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
    else:
        output_path = input_path.with_name(f"{input_path.stem}_gps.csv")

    df = pd.read_csv(input_path)

    required_columns = ["Time", "GPS", "GAlt(m)", "GSpd(kmh)"]
    missing = [name for name in required_columns if name not in df.columns]
    if missing:
        print("Error: missing required columns: " + ", ".join(missing))
        raise SystemExit(1)

    lat, lon = parse_gps_column(df["GPS"])

    out_df = pd.DataFrame(
        {
            "Time": df["Time"],
            "Latitude": lat,
            "Longitude": lon,
            "Altitude": pd.to_numeric(df["GAlt(m)"], errors="coerce"),
            "Speed": pd.to_numeric(df["GSpd(kmh)"], errors="coerce"),
        }
    )

    out_df = out_df.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
    out_df.to_csv(output_path, index=False)

    print(f"Created: {output_path}")
    print(f"Rows written: {len(out_df)}")


if __name__ == "__main__":
    main()
