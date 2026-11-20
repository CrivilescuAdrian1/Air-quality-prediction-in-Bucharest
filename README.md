# Air Quality Prediction in Bucharest

Data mining course project, **Phase 1 (intermediary)**: dataset, intended analyses and selected algorithms.

**Goal:** find out how well PM2.5 levels in Bucharest can be predicted using only weather and calendar information (no pollutant measurements as inputs), and which factors matter most.

## Task

- **Classification:** predict the air quality category derived from PM2.5: `good` (<= 15 ug/m3), `moderate` (<= 35), `unhealthy` (> 35). The thresholds are loosely based on WHO guidelines.
- **Regression:** predict the PM2.5 value (ug/m3).

## Dataset

The dataset is built from scratch by `collect_data.py` through public API calls. No pre-made dataset is used.

| | |
|---|---|
| Entries | 17,533 hourly records |
| Period | 2024-11-14 to 2026-11-14 (2 years) |
| Location | Bucharest (44.4268 N, 26.1025 E) |
| Columns | 16: `time`, 13 features, 2 targets |
| Missing values | none |

**Sources**
- [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api): PM2.5 (model-based values, not ground station measurements)
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api): weather variables
- Romanian public holidays from the [`holidays`](https://pypi.org/project/holidays/) Python package

**Columns**

| Group | Columns |
|---|---|
| Identifier | `time` (kept for reference, not used as a feature) |
| Weather features (6) | `temperature_2m`, `relative_humidity_2m`, `surface_pressure`, `wind_speed_10m`, `precipitation`, `precip_roll24` (precipitation summed over the previous 24 h) |
| Calendar features (7) | `hour`, `day_of_week`, `month`, `is_weekend`, `is_holiday`, `is_rush_hour`, `is_heating_season` |
| Targets (2) | `pm2_5` (regression), `pm2_5_category` (classification) |

**Target classes:** good 10,744 (61.3%), moderate 3,963 (22.6%), unhealthy 2,826 (16.1%).

Other pollutants (PM10, NO2, ...) are deliberately not collected: they are strongly correlated with PM2.5, so using them would reduce the task to predicting one pollutant from another instead of from weather and calendar factors.

## Intended analyses

1. **Exploratory data analysis:** how PM2.5 varies by month, hour of day and day of week; distribution of PM2.5; correlation of each feature with PM2.5; effect of weekends, public holidays and the heating season.
2. **Clustering:** group days by their weather and calendar profile with k-means++ (number of clusters chosen with the Silhouette score), visualize with t-SNE, and profile each cluster by its average PM2.5.
3. **Prediction:** classification of the air quality category and regression of the PM2.5 value.
4. **Evaluation and interpretation:** compare the algorithms with appropriate metrics and look at which features matter most.

**Evaluation protocol:** chronological train/test split (earlier data for training, the most recent 20% for testing, with a gap between them), so that the models never see the future during training.

## Selected algorithms for benchmarking

| Algorithm | Role |
|---|---|
| Naive Bayes | Simple baseline classifier |
| Random Forest | Main model (classification and regression) |

Classification is benchmarked with accuracy, F1-score and confusion matrices, and compared with a majority-class baseline. Regression uses Random Forest (Naive Bayes is a classifier only) with MAE, RMSE and R2, compared with a mean-prediction baseline.

## Repository content

```
collect_data.py                # builds data/bucharest_air_quality.csv from the APIs
requirements.txt
data/bucharest_air_quality.csv # generated dataset
```

## How to run

Requires Python 3.10+ and an internet connection.

```bash
pip install -r requirements.txt
python collect_data.py
```

The script downloads the data, creates the features and writes `data/bucharest_air_quality.csv`. The number of years collected is set by `YEARS` at the top of the file; the collection period depends on the day the script is run.
