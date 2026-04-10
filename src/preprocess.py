# what it does: Load raw UCI power consumption data, extract time features, save as CSV
# input: data/raw/household_power_consumption.txt
# output: data/processed/features.csv

import pandas as pd
import os
import sys

RAW_PATH = "data/raw/household_power_consumption.txt"
OUTPUT_PATH = "data/processed/features.csv"

if not os.path.exists(RAW_PATH):
    print(f"ERROR: {RAW_PATH} not found. Please download the UCI dataset manually.")
    sys.exit(1)

df = pd.read_csv(RAW_PATH, sep=";", low_memory=False)
print(f"Loaded {len(df)} rows")

df.replace("?", pd.NA, inplace=True)
df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
df = df.dropna(subset=["Global_active_power"])
print(f"After dropping missing: {len(df)} rows")

df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")

df["hour_of_day"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["is_peak_hour"] = df["hour_of_day"].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)

df = df[["hour_of_day", "day_of_week", "is_weekend", "is_peak_hour", "Global_active_power"]].rename(
    columns={"Global_active_power": "target"}
)
df = df.dropna()

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
