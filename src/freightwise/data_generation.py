"""Create a reproducible, clearly synthetic air-freight dataset."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

TRADE_LANE = "Shanghai, China -> Mumbai, India"


def generate_synthetic_data(
    start_date: date | None = None,
    end_date: date | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate daily USD-per-kg rates with trend, seasonality, and noise."""
    end_date = end_date or date.today()
    start_date = start_date or end_date - timedelta(days=729)
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    random = np.random.default_rng(seed)
    day_number = np.arange(len(dates))
    festival_season = dates.month.isin([10, 11]).astype(int)
    annual_cycle = np.sin(2 * np.pi * day_number / 365.25)
    weekly_cycle = np.sin(2 * np.pi * day_number / 7)

    # Rates are illustrative, not live quotes. The festival premium is strongest
    # in October and November to make the seasonal pattern easy to inspect.
    base_freight_rate = (
        4.35
        + 0.0007 * day_number
        + 0.20 * annual_cycle
        + 0.95 * festival_season
        + 0.025 * weekly_cycle
        + random.normal(0, 0.12, len(dates))
    )
    fuel_surcharge = (
        0.72
        + 0.08 * np.sin(2 * np.pi * day_number / 180)
        + 0.18 * festival_season
        + random.normal(0, 0.035, len(dates))
    )
    total_price = (
        base_freight_rate
        + fuel_surcharge
        + random.normal(0, 0.06, len(dates))
    )

    return pd.DataFrame(
        {
            "date": dates,
            "trade_lane": TRADE_LANE,
            "base_freight_rate": np.round(base_freight_rate, 3),
            "fuel_surcharge": np.round(fuel_surcharge, 3),
            "season_flag": festival_season,
            "total_price": np.round(total_price, 3),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic_air_freight.csv"),
        help="Where to write the generated CSV (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = generate_synthetic_data(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    print(
        f"Wrote {len(dataset)} synthetic daily rows to {args.output} "
        f"({dataset['date'].min().date()} through {dataset['date'].max().date()})."
    )


if __name__ == "__main__":
    main()