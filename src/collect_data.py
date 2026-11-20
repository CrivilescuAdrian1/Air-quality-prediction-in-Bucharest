"""
Colectare dataset: calitatea aerului in Bucuresti (date orare).

Surse (doar API-uri publice, fara dataset-uri gata facute):
  - Open-Meteo Air Quality API        -> PM2.5 (tinta)
  - Open-Meteo Historical Weather API -> temperatura, umiditate, presiune, vant, precipitatii
  - Calendar generat in cod           -> ora, zi a saptamanii, luna, weekend, sarbatori legale RO

Datasetul final contine 16 coloane:
  time + 13 features (6 meteo, 7 calendar) + 2 tinte (pm2_5, pm2_5_category).
Poluantii (PM10, NO2 etc.) nu sunt folositi ca features, ca sa vedem cat se poate
prezice poluarea doar din vreme si calendar.

Rulare:
    pip install -r requirements.txt
    python collect_data.py
Rezultat: data/bucharest_air_quality.csv
"""

from datetime import date, timedelta
from pathlib import Path

import holidays
import pandas as pd
import requests

YEARS = 2  # cati ani de istoric colectezi
LAT, LON = 44.4268, 26.1025  # Bucuresti
TZ = "Europe/Bucharest"

AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

AIR_VARS = ["pm2_5"]
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "precipitation",
]

# Praguri pentru categoria de calitate a aerului (PM2.5, ug/m3):
# good <= 15 (ghid OMS 24h), moderate <= 35, unhealthy > 35
PM25_BINS = [-float("inf"), 15, 35, float("inf")]
PM25_LABELS = ["good", "moderate", "unhealthy"]

FEATURES = (
    ["temperature_2m", "relative_humidity_2m", "surface_pressure",
     "wind_speed_10m", "precipitation", "precip_roll24"]  # meteo
    + ["hour", "day_of_week", "month", "is_weekend", "is_holiday",
       "is_rush_hour", "is_heating_season"]  # calendar
)
TARGETS = ["pm2_5", "pm2_5_category"]


def fetch_hourly(url: str, variables: list[str], start: date, end: date) -> pd.DataFrame:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join(variables),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": TZ,
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    ro_holidays = holidays.Romania(years=sorted(df["time"].dt.year.unique()))
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek  # 0 = luni
    df["month"] = df["time"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"] = df["time"].dt.date.map(lambda d: d in ro_holidays).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["is_heating_season"] = df["month"].isin([10, 11, 12, 1, 2, 3]).astype(int)
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    # precipitatii cumulate in ultimele 24h (ploaia "spala" aerul)
    df = df.sort_values("time").reset_index(drop=True)
    df["precip_roll24"] = df["precipitation"].rolling(24, min_periods=12).sum()
    return df


def main() -> None:
    # arhiva meteo are cateva zile de intarziere
    end = date.today() - timedelta(days=6)
    start = end - timedelta(days=365 * YEARS)
    print(f"Interval: {start} -> {end}")

    air = fetch_hourly(AIR_URL, AIR_VARS, start, end)
    weather = fetch_hourly(WEATHER_URL, WEATHER_VARS, start, end)
    df = air.merge(weather, on="time", how="inner")

    df = add_calendar_features(df)
    df = add_weather_features(df)

    # tinta de clasificare: categoria de calitate a aerului
    df["pm2_5_category"] = pd.cut(df["pm2_5"], bins=PM25_BINS, labels=PM25_LABELS)

    df = df[["time"] + FEATURES + TARGETS]
    df = df.dropna().reset_index(drop=True)

    out = Path("data")
    out.mkdir(exist_ok=True)
    path = out / "bucharest_air_quality.csv"
    df.to_csv(path, index=False)

    print(f"Salvat: {path}  ({len(df)} randuri, {df.shape[1]} coloane)")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    print("\nValori lipsa pe coloana:", "nicio valoare lipsa" if missing.empty else "\n" + missing.to_string())
    print("\nDistributia categoriilor:")
    print(df["pm2_5_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
