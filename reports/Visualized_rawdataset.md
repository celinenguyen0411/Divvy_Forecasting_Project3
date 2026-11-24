# Visualize the dataset 

### Data quality
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

### Findings
Peak daily ridership: 2025-08-02 (32,308 rides).
Lowest daily ridership: 2025-11-29 (1,024 rides).
Highest monthly mean: Aug (25,494 rides/day).
One year cannot establish a repeating seasonal pattern or prove weather causes changes.

### Weather and date alignment
We combined daily ridership with the hourly Chicago weather data identified as Open-Meteo in our README. For each day, we calculated the average temperature in °C, total rainfall in mm, and total snowfall in cm. Rainfall and snowfall were kept separate because rainfall alone does not represent total precipitation.

The weather file contains 8,760 hourly observations covering all 365 days of 2025, with no missing values in the daily weather variables. Every ridership date matched a weather date. We treated trip start times as Chicago local time, but we still need to confirm the source’s timestamp convention. We also need to check how the weather export handles daylight-saving time because its metadata lists a fixed UTC−6 offset. Until we confirm those details and the original download settings, we consider the merge provisional.

### Baseline plan
Our baseline will predict each day’s ridership using the number of rides on the same weekday one week earlier. For example, a Monday’s prediction will use the previous Monday’s ride count.

We plan to reserve October through December for evaluation and test forecasts in seven-day blocks. Each forecast will use only information available when it is made. We will compare the baseline and our models on the same dates using MAE and RMSE to measure error size, along with mean signed error to show whether predictions tend to be too high or too low. We will keep the data in chronological order and will not use actual future weather to make predictions.
