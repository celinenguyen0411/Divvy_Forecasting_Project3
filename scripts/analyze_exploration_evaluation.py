
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
START = pd.Timestamp("2025-10-01")
END = pd.Timestamp("2025-12-31")
ONE_WEEK = pd.Timedelta(days=7)


def load_data():
    df = pd.read_csv(PROCESSED / "daily_ridership_weather_2025.csv", parse_dates=["date"])
    if not pd.DatetimeIndex(df.date.sort_values()).equals(pd.date_range("2025-01-01", END)) or df.total_rides.isna().any():
        raise ValueError("Expected one nonmissing ridership row per date in 2025.")
    return df.set_index("date").sort_index()


def calendar_features(dates):
    weekdays = pd.Categorical(dates.day_of_week, categories=range(7))
    features = pd.get_dummies(
        pd.Series(weekdays, index=dates), prefix="weekday", dtype=float
    )
    phase = 2 * np.pi * (dates.day_of_year - 1) / 365.25
    features["annual_sin"] = np.sin(phase)
    features["annual_cos"] = np.cos(phase)
    return features


def metrics(actual, predicted):
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "Bias (predicted - actual)": float(np.mean(error)),
        "WAPE (%)": float(np.abs(error).sum() / np.abs(actual).sum() * 100),
    }


def chronological_forecasts(series):
    """Evaluate the fixed training split and weekly inputs in forecasting.ipynb."""
    train_dates = series.index[
        (series.index >= series.index.min() + ONE_WEEK) & (series.index < START)
    ]
    x_train = calendar_features(train_dates)
    x_train["rides_last_week"] = series.reindex(train_dates - ONE_WEEK).to_numpy()
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(x_train, series.loc[train_dates])

    blocks = []
    for origin in pd.date_range(START, END, freq="7D"):
        block_end = min(origin + pd.Timedelta(days=6), END)
        dates = pd.date_range(origin, block_end, freq="D")
        available_rides = series.loc[series.index < origin]
        prior_week = available_rides.reindex(dates - ONE_WEEK)
        if prior_week.isna().any():
            raise ValueError(f"Missing prior-week rides for {origin.date()}.")

        x_test = calendar_features(dates)
        x_test["rides_last_week"] = prior_week.to_numpy()
        blocks.append(pd.DataFrame({
            "date": dates,
            "issue_date": origin,
            "horizon_day": (dates - origin).days + 1,
            "previous_week_baseline": prior_week.to_numpy(),
            "ridge_prediction": model.predict(x_test).clip(min=0),
        }))

    results = pd.concat(blocks, ignore_index=True)
    expected_dates = pd.date_range(START, END, freq="D")
    if not pd.DatetimeIndex(results.date).equals(expected_dates):
        raise ValueError("Expected one prediction for every holdout date.")
    results["actual_rides"] = series.loc[expected_dates].to_numpy()
    return results


def svg_bar(path, title, labels, values, y_label):
    width, height = 920, 430
    left, top, plot_w, plot_h = 75, 65, 800, 285
    vmax = max(values) * 1.12
    bw = plot_w / len(values) * 0.65
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>',
                f'<text x="{left}" y="32" font-family="sans-serif" font-size="20">{title}</text>',
                f'<text x="20" y="{top+plot_h/2}" transform="rotate(-90 20 {top+plot_h/2})" font-family="sans-serif" font-size="13">{y_label}</text>']
    for i in range(5):
        y = top + plot_h * i / 4
        val = vmax * (1 - i / 4)
        elements += [f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" stroke="#ddd"/>',
                     f'<text x="{left-8}" y="{y+4}" text-anchor="end" font-family="sans-serif" font-size="11">{val:,.0f}</text>']
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + (i + 0.5) * plot_w / len(values)
        bar_h = value / vmax * plot_h
        y = top + plot_h - bar_h
        elements += [f'<rect x="{x-bw/2:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bar_h:.1f}" fill="#167d8d"/>',
                     f'<text x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{label}</text>',
                     f'<text x="{x:.1f}" y="{y-6:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:,.0f}</text>']
    elements.append('</svg>')
    path.write_text('\n'.join(elements))


def svg_eval(path, pred):
    data = pred.copy()
    data["date"] = pd.to_datetime(data.date)
    cols = [("Actual", "actual_rides", "#222222"), ("Previous-week baseline", "previous_week_baseline", "#e18b25"), ("Ridge", "ridge_prediction", "#167d8d")]
    width, height, left, top, plot_w, plot_h = 1100, 450, 90, 55, 970, 300
    vmax = float(data[[x[1] for x in cols]].to_numpy().max()) * 1.08
    elems = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>',
             '<text x="90" y="30" font-family="sans-serif" font-size="20">Chronological holdout: actual and predicted daily rides</text>']
    for i in range(5):
        y = top + plot_h * i / 4
        v = vmax * (1 - i / 4)
        elems += [f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" stroke="#ddd"/>', f'<text x="{left-8}" y="{y+4}" text-anchor="end" font-family="sans-serif" font-size="11">{v:,.0f}</text>']
    for j, (label, col, color) in enumerate(cols):
        pts = []
        vals = data[col].to_numpy()
        for i, value in enumerate(vals):
            x = left + i * plot_w / max(len(vals)-1, 1)
            y = top + plot_h * (1 - float(value) / vmax)
            pts.append(f"{x:.1f},{y:.1f}")
        elems.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2"/>')
        elems.append(f'<text x="{left+j*240}" y="420" fill="{color}" font-family="sans-serif" font-size="13">{label}</text>')
    elems += [f'<text x="{left}" y="380" font-family="sans-serif" font-size="11">Oct 1</text>',
              f'<text x="{left+plot_w*0.34:.0f}" y="380" font-family="sans-serif" font-size="11">Nov 1</text>',
              f'<text x="{left+plot_w*0.67:.0f}" y="380" font-family="sans-serif" font-size="11">Dec 1</text>', '</svg>']
    path.write_text('\n'.join(elems))


def svg_scatter(path, x, y, rho):
    width, height, left, top, plot_w, plot_h = 920, 440, 85, 60, 780, 290
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = 0.0, float(y.max()) * 1.08
    elems = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>',
             f'<text x="{left}" y="30" font-family="sans-serif" font-size="20">Daily rides and mean temperature (Spearman ρ = {rho:+.2f})</text>',
             f'<text x="{left+plot_w/2}" y="410" text-anchor="middle" font-family="sans-serif" font-size="13">Daily mean temperature (°C)</text>',
             f'<text x="20" y="{top+plot_h/2}" transform="rotate(-90 20 {top+plot_h/2})" font-family="sans-serif" font-size="13">Daily rides</text>']
    for i in range(5):
        frac = i / 4
        yy = top + plot_h * (1-frac)
        yval = ymax * frac
        xx = left + plot_w * frac
        xval = xmin + (xmax-xmin)*frac
        elems += [f'<line x1="{left}" y1="{yy}" x2="{left+plot_w}" y2="{yy}" stroke="#ddd"/>',
                  f'<text x="{left-8}" y="{yy+4}" text-anchor="end" font-family="sans-serif" font-size="11">{yval:,.0f}</text>',
                  f'<text x="{xx}" y="{top+plot_h+20}" text-anchor="middle" font-family="sans-serif" font-size="11">{xval:.0f}</text>']
    for xv, yv in zip(x, y):
        cx = left + (float(xv)-xmin)/(xmax-xmin)*plot_w
        cy = top + plot_h * (1-float(yv)/ymax)
        elems.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.5" fill="#167d8d" fill-opacity="0.48"/>')
    elems.append('</svg>')
    path.write_text('\n'.join(elems))


def main():
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_data()
    rides = df.total_rides.astype(float)
    forecast_rides = pd.read_csv(
        PROCESSED / "daily_ridership_2025.csv", parse_dates=["date"], index_col="date"
    ).total_rides.sort_index().astype(float)
    if not forecast_rides.equals(rides):
        raise ValueError("Ride counts differ between the forecasting and exploration tables.")
    monthly = rides.groupby(rides.index.month).mean()
    monthly.index = [pd.Timestamp(2025, int(m), 1).strftime("%b") for m in monthly.index]
    weekday = rides.groupby(rides.index.dayofweek).mean().reindex(range(7))
    weekday.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekend = rides[rides.index.dayofweek >= 5].mean()
    weekday_mean = rides[rides.index.dayofweek < 5].mean()
    temp_corr = df[["total_rides", "temperature_mean_c"]].corr(method="spearman").iloc[0, 1]
    rain_corr = df[["total_rides", "rain_mm"]].corr(method="spearman").iloc[0, 1]
    snow_corr = df[["total_rides", "snowfall_cm"]].corr(method="spearman").iloc[0, 1]
    summary = {
        "days": int(len(rides)), "annual_mean_daily_rides": float(rides.mean()),
        "median_daily_rides": float(rides.median()), "daily_std_dev": float(rides.std()),
        "minimum_date": rides.idxmin().date().isoformat(), "minimum_rides": int(rides.min()),
        "maximum_date": rides.idxmax().date().isoformat(), "maximum_rides": int(rides.max()),
        "monthly_mean_rides": {str(k): float(v) for k, v in monthly.items()},
        "weekday_mean_rides": {str(k): float(v) for k, v in weekday.items()},
        "weekday_mean": float(weekday_mean), "weekend_mean": float(weekend),
        "weekend_vs_weekday_pct": float((weekend / weekday_mean - 1) * 100),
        "spearman_ridership_temperature": float(temp_corr),
        "spearman_ridership_rain": float(rain_corr),
        "spearman_ridership_snowfall": float(snow_corr),
    }
    (REPORTS / "exploratory_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame({"day_of_week": weekday.index, "mean_rides": weekday.values}).to_csv(REPORTS / "weekday_summary.csv", index=False)
    svg_bar(FIGURES / "weekday_mean_ridership.svg", "Average Divvy rides by day of week", list(weekday.index), weekday.values, "Mean rides/day")
    svg_bar(FIGURES / "monthly_mean_ridership.svg", "Average Divvy rides by month", list(monthly.index), monthly.values, "Mean rides/day")
    svg_scatter(FIGURES / "ridership_vs_temperature.svg", df.temperature_mean_c, rides, temp_corr)

    pred = chronological_forecasts(forecast_rides)
    pred.to_csv(PROCESSED / "chronological_forecasts_oct_dec_2025.csv", index=False, date_format="%Y-%m-%d")
    metric_rows = []
    for model_name, col in [("Previous-week baseline", "previous_week_baseline"), ("Team Ridge model", "ridge_prediction")]:
        metric_rows.append({"model": model_name, **metrics(pred.actual_rides, pred[col])})
    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(REPORTS / "chronological_metrics.csv", index=False)
    by_horizon = []
    for horizon, group in pred.groupby("horizon_day"):
        by_horizon.append({"horizon_day": int(horizon), "n": len(group),
                           "baseline_MAE": metrics(group.actual_rides, group.previous_week_baseline)["MAE"],
                           "ridge_MAE": metrics(group.actual_rides, group.ridge_prediction)["MAE"]})
    pd.DataFrame(by_horizon).to_csv(REPORTS / "chronological_metrics_by_horizon.csv", index=False)
    svg_eval(FIGURES / "chronological_holdout_predictions.svg", pred)

    b = metric_df.iloc[0]
    r = metric_df.iloc[1]
    delta = (1 - r.MAE / b.MAE) * 100
    better = "lower" if delta > 0 else "higher"
    month_lines = "\n".join(f"| {m} | {v:,.0f} |" for m, v in monthly.items())
    weekday_lines = "\n".join(f"| {d} | {v:,.0f} |" for d, v in weekday.items())
    metric_lines = "\n".join(f"| {row.model} | {row.MAE:,.0f} | {row.RMSE:,.0f} | {row['Bias (predicted - actual)']:+,.0f} | {row['WAPE (%)']:.1f}% |" for _, row in metric_df.iterrows())
    report = f'''# Exploratory findings and chronological evaluation

### Exploratory findings

The 2025 table contains {len(rides)} daily observations. Ridership averaged **{rides.mean():,.0f} rides/day** (median {rides.median():,.0f}; standard deviation {rides.std():,.0f}). The lowest day was **{rides.idxmin():%b %-d}** ({rides.min():,.0f}) and the highest was **{rides.idxmax():%b %-d}** ({rides.max():,.0f}).

Monthly average rides show a pronounced warm-season peak and winter trough:

| Month | Mean rides/day |
|---|---:|
{month_lines}

The mean on weekend days was **{weekend:,.0f} rides**, versus **{weekday_mean:,.0f}** on weekdays ({(weekend/weekday_mean-1)*100:+.1f}%). Day-of-week averages are:

| Day | Mean rides/day |
|---|---:|
{weekday_lines}

Spearman rank correlations with daily rides were {temp_corr:+.2f} for temperature, {rain_corr:+.2f} for rain, and {snow_corr:+.2f} for snowfall. These are descriptive associations, not causal effects. Because temperature and ridership both change over the year, the temperature association may largely reflect season. The weather file's fixed UTC offset still needs DST/source verification; weather findings are provisional. One year contains only one winter and one summer, so it cannot distinguish a repeatable seasonal cycle from 2025-specific conditions.

![Average rides by month](figures/monthly_mean_ridership.svg)

![Average rides by weekday](figures/weekday_mean_ridership.svg)

![Daily rides and temperature](figures/ridership_vs_temperature.svg)

### Chronological evaluation

The October 1–December 31, 2025 holdout covers 92 days. The Ridge model is fit once using eligible January–September dates. At the start of each seven-day block, both forecasts use actual rides from the previous week, which are already available. The final block contains only December 31.

The baseline repeats the count from seven days earlier. The Ridge model in [forecasting.ipynb](../forecasting.ipynb) uses that same count along with weekday indicators and annual sine and cosine terms. The model standardizes the predictors using the training period and uses a Ridge penalty of 10. Both forecasts use calendar dates and past ride counts only.

| Model | MAE (rides) | RMSE (rides) | Bias (predicted − actual) | WAPE |
|---|---:|---:|---:|---:|
{metric_lines}

The team Ridge model’s MAE is {abs(delta):.1f}% {better} than the baseline. Its RMSE is {abs(1 - r.RMSE / b.RMSE):.1%} {"lower" if r.RMSE < b.RMSE else "higher"}. MAE measures the average size of a daily miss. RMSE gives more weight to large misses. Both forecasts overpredicted on average in this holdout, so taking their daily estimates at face value could lead to staging more bikes than needed. WAPE is total absolute error divided by total actual rides.

![Holdout forecasts](figures/chronological_holdout_predictions.svg)

### Limitations and next steps

- The forecast is citywide and daily; it does not determine station-level bike placement.
- The evaluation covers one late-year period in one calendar year. Metrics may not generalize to another year.
- The operational evaluation intentionally excludes weather predictors. Add weather only after verifying local-date/DST alignment and using weather forecasts archived or available at each issue date.
- The previous-week baseline and Ridge model are compared on identical dates. Inspect horizon-specific error in `chronological_metrics_by_horizon.csv` and error direction before making staging recommendations.

### Reproduce

From the repository root, run `python3 scripts/analyze_exploration_evaluation.py`. The script reads the prepared daily merge and writes the evaluation CSVs, summary JSON, this report, and SVG figures.
'''
    (REPORTS / "exploratory_chronological_evaluation.md").write_text(report)
    print(metric_df.to_string(index=False))
    print(f"\nHoldout dates: {pred.date.min():%Y-%m-%d} through {pred.date.max():%Y-%m-%d} ({len(pred)} days)")
    print(f"Exploratory report: {REPORTS / 'exploratory_chronological_evaluation.md'}")


if __name__ == "__main__":
    main()
