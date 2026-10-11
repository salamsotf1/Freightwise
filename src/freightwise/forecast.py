"""Compare Prophet and XGBoost forecasts for one air-freight trade lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

LAG_DAYS = (1, 7, 14, 28)
EXOGENOUS_COLUMNS = ("base_freight_rate", "fuel_surcharge", "season_flag")


def load_dataset(path: Path) -> pd.DataFrame:
    """Read and validate the CSV in chronological order."""
    data = pd.read_csv(path, parse_dates=["date"])
    required_columns = {
        "date",
        "trade_lane",
        "base_freight_rate",
        "fuel_surcharge",
        "season_flag",
        "total_price",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("Dataset contains no rows")
    if data["date"].duplicated().any():
        raise ValueError("Dataset must contain only one row per date for this lane")
    return data.sort_values("date").reset_index(drop=True)


def add_lag_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add past prices as model inputs; shifting keeps future prices out."""
    featured = data.copy()
    for lag in LAG_DAYS:
        featured[f"lag_{lag}"] = featured["total_price"].shift(lag)
    return featured.dropna().reset_index(drop=True)


def make_xgboost_model() -> XGBRegressor:
    """Return a modest, reproducible model suitable for a laptop demo."""
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=250,
        learning_rate=0.04,
        max_depth=3,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
    )


def fit_xgboost(data: pd.DataFrame) -> XGBRegressor:
    featured = add_lag_features(data)
    feature_columns = [*EXOGENOUS_COLUMNS, *(f"lag_{lag}" for lag in LAG_DAYS)]
    model = make_xgboost_model()
    model.fit(featured[feature_columns], featured["total_price"])
    return model


def forecast_xgboost(
    model: XGBRegressor,
    history: pd.DataFrame,
    future_dates: pd.Series | pd.DatetimeIndex,
) -> np.ndarray:
    """Forecast recursively, holding unknown base and fuel rates flat."""
    prices = history["total_price"].astype(float).tolist()
    last_base_rate = float(history["base_freight_rate"].iloc[-1])
    last_fuel_surcharge = float(history["fuel_surcharge"].iloc[-1])
    predictions: list[float] = []
    feature_columns = [*EXOGENOUS_COLUMNS, *(f"lag_{lag}" for lag in LAG_DAYS)]

    for future_date in pd.to_datetime(future_dates):
        row = {
            "base_freight_rate": last_base_rate,
            "fuel_surcharge": last_fuel_surcharge,
            "season_flag": int(future_date.month in (10, 11)),
        }
        for lag in LAG_DAYS:
            row[f"lag_{lag}"] = prices[-lag]
        prediction = float(model.predict(pd.DataFrame([row])[feature_columns])[0])
        predictions.append(prediction)
        prices.append(prediction)

    return np.asarray(predictions)


def forecast_prophet(history: pd.DataFrame, future_dates: pd.DatetimeIndex) -> np.ndarray:
    """Fit Prophet to date/price only and predict the supplied future dates."""
    training = history.rename(columns={"date": "ds", "total_price": "y"})[["ds", "y"]]
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
    )
    model.fit(training)
    prediction = model.predict(pd.DataFrame({"ds": future_dates}))
    return prediction["yhat"].to_numpy()


def score_predictions(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def make_recommendation(
    model_name: str, current_average: float, forecast_average: float
) -> str:
    """Turn the selected model's next-week movement into plain English."""
    change_percent = (forecast_average - current_average) / current_average * 100
    if change_percent >= 2:
        advice = "consider booking now"
        movement = "rise"
    elif change_percent <= -2:
        advice = "consider waiting if your schedule allows"
        movement = "fall"
    else:
        advice = "prices look broadly stable, so compare quotes and timing"
        movement = "stay broadly stable"
    return (
        f"Using {model_name}, the average price for this lane is likely to "
        f"{movement} by {abs(change_percent):.1f}% over the next week; {advice}."
    )


def run_forecast(
    data: pd.DataFrame,
    holdout_days: int = 14,
    forecast_days: int = 14,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], str, str]:
    """Evaluate on a final time-based holdout, then forecast beyond all data."""
    if holdout_days < 1 or forecast_days < 1:
        raise ValueError("holdout_days and forecast_days must be positive")
    if len(data) <= holdout_days + max(LAG_DAYS):
        raise ValueError(
            f"Need more than {holdout_days + max(LAG_DAYS)} daily rows "
            "for lag features and the requested holdout"
        )

    split_at = len(data) - holdout_days
    training = data.iloc[:split_at].copy()
    holdout = data.iloc[split_at:].copy()
    holdout_dates = pd.DatetimeIndex(holdout["date"])

    # The holdout simulates a real booking decision: future component rates are
    # unknown, so XGBoost carries the latest training rates forward.
    evaluation_prophet = forecast_prophet(training, holdout_dates)
    evaluation_xgboost = forecast_xgboost(
        fit_xgboost(training), training, holdout_dates
    )
    metrics = {
        "prophet": score_predictions(holdout["total_price"], evaluation_prophet),
        "xgboost": score_predictions(holdout["total_price"], evaluation_xgboost),
    }
    best_model_name = min(metrics, key=lambda name: metrics[name]["mae"])

    all_dates = pd.DatetimeIndex(data["date"])
    future_dates = pd.date_range(
        start=all_dates[-1] + pd.Timedelta(days=1), periods=forecast_days, freq="D"
    )
    full_prophet = forecast_prophet(data, future_dates)
    full_xgboost = forecast_xgboost(fit_xgboost(data), data, future_dates)
    forecast = pd.DataFrame(
        {
            "date": future_dates,
            "prophet_total_price": full_prophet,
            "xgboost_total_price": full_xgboost,
        }
    )
    selected_forecast = forecast[f"{best_model_name}_total_price"]
    current_average = float(data["total_price"].tail(7).mean())
    next_week_average = float(selected_forecast.head(min(7, forecast_days)).mean())
    recommendation = make_recommendation(
        best_model_name, current_average, next_week_average
    )
    return forecast, metrics, best_model_name, recommendation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_air_freight.csv"))
    parser.add_argument("--holdout-days", type=int, default=14)
    parser.add_argument("--forecast-days", type=int, default=14)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    data = load_dataset(args.data)
    forecast, metrics, best_model_name, recommendation = run_forecast(
        data, holdout_days=args.holdout_days, forecast_days=args.forecast_days
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = args.output_dir / "forecast.csv"
    metrics_path = args.output_dir / "metrics.json"
    forecast.to_csv(forecast_path, index=False, date_format="%Y-%m-%d")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    lane = data["trade_lane"].iloc[0]
    print(f"Trade lane: {lane}")
    print(f"Validation MAE (lower is better): Prophet {metrics['prophet']['mae']:.3f}, "
          f"XGBoost {metrics['xgboost']['mae']:.3f}")
    print(f"Selected from holdout: {best_model_name}")
    print(recommendation)
    print(f"Forecast saved to {forecast_path}; metrics saved to {metrics_path}.")


if __name__ == "__main__":
    main()