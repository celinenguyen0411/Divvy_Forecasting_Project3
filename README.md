Project 3: Divvy Ridership Forecasting
Contributors: Mai Hong Nguyen & Jeremy David Mueller

## Purpose
This project forecasts the number of completed Divvy rides that will start each day in Chicago over the next seven days. The forecast helps a Divvy operations manager decide how many bikes to stage for the next morning and which days may require extra capacity. The target is citywide rides per day. Station-level staging would require additional local demand and inventory data.
Objectives
1. Build a daily ridership dataset from all twelve months of 2025 Divvy trip records and align it with Chicago weather by local date.
2. Explore how ridership varies by season, day of the week, temperature, and precipitation.
3. Use unsupervised learning to identify groups of days or stations with distinct patterns and explain what those groups mean for operations.
4. Train a supervised forecasting model and evaluate it against a simple baseline on a chronological holdout.
5. Produce a dated seven-day outlook using a Chicago weather forecast available at issue time. Explain typical error, the direction of any bias, and how both affect staging decisions.
6. Present the historical analysis, model evaluation, and outlook in an interactive dashboard and summarize the operating implications in a short executive memo.
   
## Data sources
Divvy trip data	January 1–December 31, 2025
Open-Meteo Historical Weather API
Chicago seven-day weather forecast	

## Data preparation
- Assign each completed trip to its start date in America/Chicago. Confirm the source timestamp convention before parsing or converting dates.
- Check that all twelve monthly releases are represented. Document the completed-trip rule, any exclusions and their counts, and the treatment of duplicates or invalid timestamps. Station metadata is only needed for station-level analysis.
- Compare the aggregated ride counts against the 365-date calendar. Investigate missing dates before treating them as zero rides.
- Align the trip and weather tables on local date. Verify unique dates, report unmatched and missing weather rows, and reconcile daily totals against retained trip records.

## Observable objects: (edit later)
- ridable types,
- time interval of renting a bike
- member_casual
- start_station_name : place has a higher frequency of renting a bike


## Proposed tech stack:
- Tableau, Python and pandas, scikit-learn, Plotly and Streamlit, Git and GitHub.


