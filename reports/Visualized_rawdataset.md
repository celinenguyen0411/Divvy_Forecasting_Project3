# Visualize the dataset

## Data quality
| Check | Count |
| --- | ---: |
| raw_records | 5,552,994 |
| invalid_timestamp | 0 |
| outside_start_year | 53 |
| end_before_start | 29 |
| missing_ride_id | 0 |
| duplicate_valid_id | 0 |
| retained_trips | 5,552,912 |
| missing_ridership_dates | 0 |
| weather_unmatched_dates | 0 |

Exclusions are sequential: invalid timestamps, starts outside 2025, end before start, missing ID, then duplicate otherwise-valid ID. Zero-duration rides are retained. Missing station details are not grounds for exclusion. Missing calendar dates remain missing rather than becoming zero.

## Findings
Peak daily ridership: 2025-08-02 (32,308 rides).
Lowest daily ridership: 2025-11-29 (1,024 rides).
Highest monthly mean: Aug (25,494 rides/day).
One year cannot establish a repeating seasonal pattern or prove weather causes changes.

## Weather and date alignment
Trip start timestamps are treated as Chicago local wall-clock values; source confirmation remains necessary. Weather comes from the existing hourly export, described as Open-Meteo in the README. Temperature is averaged in Celsius; rain is summed in mm and snowfall in cm. Rain is not total precipitation. Complete daily values require 24 nonmissing hourly observations.
Weather metadata: {'latitude': 41.862915, 'longitude': -87.64877, 'elevation': 179.0, 'utc_offset_seconds': -21600, 'timezone': 'America/Chicago', 'timezone_abbreviation': 'GMT-6', 'hourly_rows_2025': 8760, 'incomplete_days': 0, 'missing_dates': 0}.
Missing merged weather values: {'temperature_mean_c': 0, 'rain_mm': 0, 'snowfall_cm': 0}.
The fixed UTC offset in the export requires checking against Chicago daylight-saving time. This merge is provisional until date alignment and original source request are confirmed.

## Baseline plan
Predict each day using rides on the same weekday of the previous week. Use a chronological holdout, provisionally October through December, and evaluate seven-day blocks using only data available at each issue date. Report MAE, RMSE and mean signed error on identical evaluation dates. No random split. Observed future weather is unavailable at forecast time.
