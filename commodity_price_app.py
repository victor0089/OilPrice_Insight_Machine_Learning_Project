"""OilPrice Insight: one-observation-ahead WTI crude-oil price forecast.

The bundled CSV contains historical daily spot prices from the U.S. Energy
Information Administration (EIA), mirrored by the datasets/oil-prices project.
Only data available on the forecast origin date are used as model features.

Usage:
    python commodity_price_app.py
    python commodity_price_app.py --data my_wti_data.csv --output results

The forecast is educational, not an investment or trading recommendation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Generate plots in desktop, server, and notebook environments.

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


APP_NAME = "OilPrice Insight"
DEFAULT_DATA = Path(__file__).resolve().parent / "wti_daily_2022_2026.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"
FEATURES = [
    "price_today",
    "price_lag_1",
    "price_lag_2",
    "price_lag_3",
    "price_lag_5",
    "price_lag_10",
    "change_1",
    "change_5",
    "rolling_mean_5",
    "rolling_mean_10",
    "rolling_mean_20",
    "rolling_std_5",
    "rolling_std_20",
]


def read_prices(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read an EIA/FRED-style CSV with Date/Price or observation_date/DCOILWTICO."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    raw = pd.read_csv(path)
    possible_dates = ["Date", "DATE", "observation_date"]
    possible_prices = ["Price", "DCOILWTICO"]
    date_col = next((c for c in possible_dates if c in raw.columns), None)
    price_col = next((c for c in possible_prices if c in raw.columns), None)
    if date_col is None or price_col is None:
        raise ValueError("CSV needs Date and Price columns (or FRED equivalents).")

    prices = pd.DataFrame(
        {
            "Date": pd.to_datetime(raw[date_col], errors="coerce"),
            "Price": pd.to_numeric(raw[price_col], errors="coerce"),
        }
    )
    invalid = int(prices[["Date", "Price"]].isna().any(axis=1).sum())
    prices = prices.dropna().sort_values("Date")
    duplicates = int(prices.duplicated("Date").sum())
    prices = prices.drop_duplicates(subset="Date", keep="last").reset_index(drop=True)
    if len(prices) < 100:
        raise ValueError("At least 100 valid dated price observations are needed.")
    if not np.isfinite(prices["Price"].to_numpy()).all():
        raise ValueError("All included prices must be finite numbers.")
    return prices, {
        "raw_records": len(raw),
        "invalid_dropped": invalid,
        "duplicate_dates_dropped": duplicates,
        "clean_records": len(prices),
    }


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Features at date t use prices at t or earlier; target is price at t+1."""
    frame = prices.copy()
    frame["price_today"] = frame["Price"]
    for days in (1, 2, 3, 5, 10):
        frame[f"price_lag_{days}"] = frame["Price"].shift(days)
    frame["change_1"] = frame["Price"].diff(1)
    frame["change_5"] = frame["Price"].diff(5)
    for window in (5, 10, 20):
        frame[f"rolling_mean_{window}"] = frame["Price"].rolling(window).mean()
    for window in (5, 20):
        frame[f"rolling_std_{window}"] = frame["Price"].rolling(window).std()
    frame["next_price"] = frame["Price"].shift(-1)
    frame["target_change"] = frame["next_price"] - frame["Price"]
    return frame.dropna(subset=FEATURES).reset_index(drop=True)


def new_model() -> RandomForestRegressor:
    """Fixed parameters and seed make repeated runs comparable."""
    return RandomForestRegressor(
        n_estimators=160,
        max_depth=10,
        min_samples_leaf=5,
        max_features=0.9,
        random_state=42,
        n_jobs=-1,
    )


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def cross_validate(train: pd.DataFrame) -> list[float]:
    """Expanding-window validation using only training-period observations."""
    scores: list[float] = []
    for fit_idx, validation_idx in TimeSeriesSplit(n_splits=4).split(train):
        fitting = train.iloc[fit_idx]
        validating = train.iloc[validation_idx]
        model = new_model()
        model.fit(fitting[FEATURES], fitting["target_change"])
        predicted = (
            validating["Price"].to_numpy()
            + model.predict(validating[FEATURES])
        )
        scores.append(float(mean_absolute_error(validating["next_price"], predicted)))
    return scores


def build_result_text(
    prices: pd.DataFrame,
    cleaned: dict[str, int],
    train: pd.DataFrame,
    test: pd.DataFrame,
    forest_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    cv_scores: list[float],
    latest_price: float,
    next_price_prediction: float,
) -> str:
    advantage = baseline_metrics["MAE"] - forest_metrics["MAE"]
    comparison = (
        f"Forest improves MAE over persistence by ${advantage:.3f}/barrel."
        if advantage > 0
        else f"Persistence beats forest by ${-advantage:.3f}/barrel in test MAE."
    )
    return "\n".join(
        [
            APP_NAME + " - run results",
            "=" * 55,
            "Commodity: WTI crude oil spot price, USD per barrel",
            f"Source data: {prices['Date'].min():%Y-%m-%d} to {prices['Date'].max():%Y-%m-%d}",
            f"Original records: {cleaned['raw_records']:,}",
            f"Missing/invalid rows removed: {cleaned['invalid_dropped']:,}",
            f"Duplicate dates removed: {cleaned['duplicate_dates_dropped']:,}",
            f"Clean observations: {cleaned['clean_records']:,}",
            f"Engineered predictors: {len(FEATURES)}",
            f"Training observations: {len(train):,} ({train['Date'].min():%Y-%m-%d} to {train['Date'].max():%Y-%m-%d})",
            f"Test observations: {len(test):,} ({test['Date'].min():%Y-%m-%d} to {test['Date'].max():%Y-%m-%d})",
            "Evaluation: one-step ahead, chronological 80/20 holdout",
            "",
            "Held-out test results (prediction of the next recorded price):",
            f"  Random Forest: MAE ${forest_metrics['MAE']:.3f} | RMSE ${forest_metrics['RMSE']:.3f} | R2 {forest_metrics['R2']:.3f}",
            f"  Persistence:   MAE ${baseline_metrics['MAE']:.3f} | RMSE ${baseline_metrics['RMSE']:.3f} | R2 {baseline_metrics['R2']:.3f}",
            f"  {comparison}",
            f"Training-only time-series CV MAE ($/barrel): {', '.join(f'{v:.3f}' for v in cv_scores)}",
            f"Training-only time-series CV mean MAE: ${np.mean(cv_scores):.3f}",
            "",
            "Forecast using the latest available observation:",
            f"  Latest observed date: {prices['Date'].max():%Y-%m-%d}",
            f"  Latest observed price: ${latest_price:.2f} per barrel",
            f"  Predicted next observation: ${next_price_prediction:.2f} per barrel",
            f"  Predicted one-observation change: {next_price_prediction - latest_price:+.2f} USD/barrel",
            "  Forecast date: the next day with a published price (not guaranteed to be the next calendar day)",
            "",
            "Important: The test predictions use each day's actual price as it becomes",
            "available. They are one-step-ahead backtests, not multi-day forecasts.",
            "The future forecast is unverified until the next observation arrives.",
            "This educational exercise is not financial advice.",
            "",
        ]
    )


def make_plots(
    prices: pd.DataFrame,
    test_predictions: pd.DataFrame,
    model: RandomForestRegressor,
    output_dir: Path,
    forecast: float,
    forest_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    dark = "#102133"
    blue = "#1769aa"
    orange = "#e49429"

    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    ax.plot(test_predictions["TargetDate"], test_predictions["Actual"], color=dark, linewidth=1.8, label="Actual next price")
    ax.plot(test_predictions["TargetDate"], test_predictions["Predicted"], color=blue, linewidth=1.5, label="Random Forest")
    ax.plot(test_predictions["TargetDate"], test_predictions["Persistence"], color=orange, linewidth=1.0, alpha=0.85, label="Persistence baseline")
    ax.set(title="WTI daily oil prices: chronological held-out test", ylabel="USD per barrel", xlabel="Target observation date")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.savefig(output_dir / "test_predictions.png", dpi=160)
    plt.close(fig)

    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    ax.barh(importance.index, importance.values, color=blue)
    ax.set(title="Random Forest feature importance (trained on historical data)", xlabel="Relative importance")
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(output_dir / "feature_importance.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 7), layout="constrained", facecolor="#f7fafc")
    grid = fig.add_gridspec(3, 2, height_ratios=[0.8, 0.85, 3.1])
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(0.02, 0.76, APP_NAME, fontweight="bold", fontsize=21, color=dark, transform=title_ax.transAxes)
    title_ax.text(0.02, 0.15, "WTI crude oil | educational one-step-ahead forecast", fontsize=11, color="#506476", transform=title_ax.transAxes)

    card1 = fig.add_subplot(grid[1, 0])
    card2 = fig.add_subplot(grid[1, 1])
    for card in (card1, card2):
        card.set_facecolor("white")
        card.set_xticks([])
        card.set_yticks([])
        for spine in card.spines.values():
            spine.set_color("#e2e8f0")
    latest = float(prices["Price"].iloc[-1])
    card1.text(0.04, 0.74, "LATEST OBSERVED", fontsize=9, color="#587087", transform=card1.transAxes)
    card1.text(0.04, 0.22, f"${latest:.2f} / bbl", fontsize=22, fontweight="bold", color=dark, transform=card1.transAxes)
    card2.text(0.04, 0.74, "PREDICTED NEXT OBSERVATION", fontsize=9, color="#587087", transform=card2.transAxes)
    card2.text(0.04, 0.22, f"${forecast:.2f} / bbl", fontsize=22, fontweight="bold", color=blue, transform=card2.transAxes)

    ax = fig.add_subplot(grid[2, :])
    recent = prices.tail(100)
    ax.plot(recent["Date"], recent["Price"], color=dark, linewidth=2, label="Observed prices")
    future_date = prices["Date"].iloc[-1] + pd.offsets.BDay(1)
    ax.scatter([future_date], [forecast], s=90, marker="D", color=orange, label="Model forecast", zorder=5)
    ax.plot([prices["Date"].iloc[-1], future_date], [latest, forecast], linestyle="--", color=orange)
    ax.set(ylabel="USD per barrel", xlabel="Date", title="Recent observations and next-observation forecast")
    ax.grid(alpha=0.2)
    ax.legend()
    ax.text(
        0.985, 0.975,
        f"Test MAE: forest ${forest_metrics['MAE']:.2f} | baseline ${baseline_metrics['MAE']:.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#e2e8f0", "alpha": 0.9},
    )
    fig.savefig(output_dir / "app_results.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_console_capture(result_text: str, output_dir: Path) -> None:
    """Create a readable PNG capture containing the program's actual output."""
    lines = result_text.strip().splitlines()
    fig = plt.figure(figsize=(12.5, max(5.5, len(lines) * 0.178 + 1.0)), facecolor="#0d1727")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(
        0.03, 0.975, "$ python commodity_price_app.py", color="#8ce0b0",
        va="top", fontfamily="DejaVu Sans Mono", fontsize=11,
    )
    ax.text(
        0.03, 0.925, result_text, color="#e4eaf2", va="top",
        fontfamily="DejaVu Sans Mono", fontsize=10.0, linespacing=1.20,
    )
    fig.savefig(output_dir / "console_run.png", dpi=145, facecolor=fig.get_facecolor())
    plt.close(fig)


def run(data_path: Path, output_dir: Path) -> None:
    prices, cleaned = read_prices(data_path)
    prepared = build_features(prices)
    labeled = prepared.dropna(subset=["next_price", "target_change"]).copy()
    if len(labeled) < 80:
        raise ValueError("Not enough prepared observations after creating lag features.")

    split = int(0.8 * len(labeled))
    training, testing = labeled.iloc[:split].copy(), labeled.iloc[split:].copy()
    cv_scores = cross_validate(training)

    holdout_model = new_model()
    holdout_model.fit(training[FEATURES], training["target_change"])
    model_test_price = testing["Price"].to_numpy() + holdout_model.predict(testing[FEATURES])
    naive_test_price = testing["Price"].to_numpy()  # tomorrow equals today's price
    actual_test_price = testing["next_price"].to_numpy()
    forest_metrics = evaluate(actual_test_price, model_test_price)
    baseline_metrics = evaluate(actual_test_price, naive_test_price)

    # For deployment, retrain using all labeled historical observations.
    final_model = new_model()
    final_model.fit(labeled[FEATURES], labeled["target_change"])
    final_features = prepared.iloc[[-1]]
    latest_price = float(prices["Price"].iloc[-1])
    if final_features["Date"].iloc[0] != prices["Date"].iloc[-1]:
        raise RuntimeError("Latest record is missing the features needed to forecast.")
    predicted_price = latest_price + float(final_model.predict(final_features[FEATURES])[0])

    # The target date in the historical test is the following observed row.
    target_dates = prices.set_index("Date").index.to_series().shift(-1)
    next_dates = testing["Date"].map(target_dates)
    test_results = pd.DataFrame(
        {
            "OriginDate": testing["Date"].to_numpy(),
            "TargetDate": next_dates.to_numpy(),
            "Actual": actual_test_price,
            "Predicted": model_test_price,
            "Persistence": naive_test_price,
        }
    )
    # Mapping dates via index avoids assuming every calendar weekday has a price.
    if test_results["TargetDate"].isna().any():
        raise RuntimeError("Could not identify some observed target dates.")

    output_dir.mkdir(parents=True, exist_ok=True)
    test_results.to_csv(output_dir / "test_predictions.csv", index=False, float_format="%.4f")
    text = build_result_text(
        prices, cleaned, training, testing, forest_metrics, baseline_metrics,
        cv_scores, latest_price, predicted_price
    )
    (output_dir / "run_results.txt").write_text(text, encoding="utf-8")
    make_plots(prices, test_results, final_model, output_dir, predicted_price, forest_metrics, baseline_metrics)
    make_console_capture(text, output_dir)
    print(text)
    print(f"Saved plots, metrics, and predictions in: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the next reported WTI crude-oil spot price.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="CSV with Date and Price columns")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory to save results and figures")
    args = parser.parse_args()
    run(args.data, args.output)


if __name__ == "__main__":
    main()
