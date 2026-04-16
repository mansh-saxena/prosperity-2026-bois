#!/usr/bin/env python3
"""Classify product behavior from basic order book price data.

This script intentionally keeps the analysis small: it loads price CSV files,
computes basic characteristics, classifies each product as TRENDING or
MEAN REVERTING, and generates only mid-price and spread plots.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from pathlib import Path

try:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/prosperity-2026-matplotlib")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError as error:
    missing = getattr(error, "name", None) or str(error)
    print(
        f"Could not import analysis dependency: {missing}. Install dependencies with "
        "`python3 -m pip install -r requirements.txt`.",
        file=sys.stderr,
    )
    raise SystemExit(1) from error


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "ROUND_1" / "ROUND1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "behavior_output"
PRICE_PREFIX = "prices_"
REQUIRED_COLUMNS = [
    "timestamp",
    "product",
    "bid_price_1",
    "bid_volume_1",
    "ask_price_1",
    "ask_volume_1",
    "mid_price",
]
NUMERIC_COLUMNS = [
    "day",
    "timestamp",
    "bid_price_1",
    "bid_volume_1",
    "ask_price_1",
    "ask_volume_1",
    "mid_price",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze price data and classify each product as trending or mean reverting."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"CSV directory or zip archive. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated plots. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--trend-strength-threshold",
        type=float,
        default=1.0,
        help="Minimum absolute net move divided by price std for TRENDING. Default: 1.0",
    )
    parser.add_argument(
        "--trend-r2-threshold",
        type=float,
        default=0.35,
        help="Minimum linear trend R^2 for TRENDING. Default: 0.35",
    )
    return parser.parse_args()


def parse_day_from_name(name: str) -> int | None:
    match = re.search(r"day_(-?\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def clean_price_frame(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["source_file"] = source_name

    file_day = parse_day_from_name(source_name)
    if file_day is not None and "day" not in frame.columns:
        frame["day"] = file_day
    elif file_day is not None:
        frame["day"] = frame["day"].fillna(file_day)

    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].replace("", np.nan)

    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def load_price_data(source: Path) -> tuple[pd.DataFrame, list[str]]:
    if not source.exists():
        raise FileNotFoundError(f"Input path does not exist: {source}")

    frames: list[pd.DataFrame] = []
    loaded_files: list[str] = []

    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise ValueError("File input must be a .zip archive")
        with zipfile.ZipFile(source) as archive:
            csv_names = sorted(name for name in archive.namelist() if name.lower().endswith(".csv"))
            for name in csv_names:
                if not Path(name).name.lower().startswith(PRICE_PREFIX):
                    continue
                with archive.open(name) as raw_file:
                    frame = pd.read_csv(raw_file, sep=";")
                frames.append(clean_price_frame(frame, name))
                loaded_files.append(name)
    else:
        for path in sorted(source.rglob("*.csv")):
            if not path.name.lower().startswith(PRICE_PREFIX):
                continue
            source_name = str(path.relative_to(source))
            frame = pd.read_csv(path, sep=";")
            frames.append(clean_price_frame(frame, source_name))
            loaded_files.append(source_name)

    if not frames:
        return pd.DataFrame(), loaded_files

    prices = pd.concat(frames, ignore_index=True)
    validate_columns(prices)
    prices = add_time_index(prices)
    return clean_top_of_book(prices), loaded_files


def validate_columns(prices: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in prices.columns]
    if missing:
        raise ValueError(f"Price data is missing required columns: {', '.join(missing)}")


def add_time_index(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    if "day" not in prices.columns:
        prices["day"] = 0
    prices["day"] = prices["day"].fillna(0).astype(int)

    max_timestamp = prices["timestamp"].max(skipna=True)
    day_width = int(max_timestamp) + 1 if pd.notna(max_timestamp) else 1
    prices["time_index"] = (prices["day"] - prices["day"].min()) * day_width + prices["timestamp"]
    return prices.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)


def clean_top_of_book(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    mask = prices[REQUIRED_COLUMNS].notna().all(axis=1)
    mask &= prices["mid_price"] > 0
    mask &= (prices["bid_volume_1"] + prices["ask_volume_1"]) > 0
    return prices.loc[mask].sort_values(["product", "day", "timestamp"]).reset_index(drop=True)


def add_basic_metrics(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    grouped = prices.groupby("product", group_keys=False)
    prices["returns"] = grouped["mid_price"].pct_change()
    prices["spread"] = prices["ask_price_1"] - prices["bid_price_1"]
    return prices


def return_autocorrelation(returns: pd.Series) -> float:
    clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean_returns) < 2:
        return np.nan
    return float(clean_returns.autocorr(lag=1))


def linear_trend_r2(product_prices: pd.DataFrame) -> float:
    clean_prices = product_prices[["time_index", "mid_price"]].dropna()
    if len(clean_prices) < 2:
        return np.nan

    x_values = clean_prices["time_index"].to_numpy(dtype=float)
    y_values = clean_prices["mid_price"].to_numpy(dtype=float)
    x_values = x_values - x_values.mean()

    slope, intercept = np.polyfit(x_values, y_values, deg=1)
    fitted = slope * x_values + intercept
    total_sum_squares = np.sum((y_values - y_values.mean()) ** 2)
    if total_sum_squares == 0:
        return 0.0
    residual_sum_squares = np.sum((y_values - fitted) ** 2)
    return float(1.0 - residual_sum_squares / total_sum_squares)


def trend_strength(product_prices: pd.DataFrame) -> float:
    ordered = product_prices.sort_values(["day", "timestamp"])
    mid_prices = ordered["mid_price"].dropna()
    if len(mid_prices) < 2:
        return np.nan

    price_std = mid_prices.std()
    if pd.isna(price_std) or price_std == 0:
        return 0.0
    net_move = mid_prices.iloc[-1] - mid_prices.iloc[0]
    return float(abs(net_move) / price_std)


def classify_market_type(
    product_prices: pd.DataFrame,
    trend_strength_threshold: float,
    trend_r2_threshold: float,
) -> str:
    strength = trend_strength(product_prices)
    r2 = linear_trend_r2(product_prices)

    if pd.notna(strength) and pd.notna(r2):
        if strength >= trend_strength_threshold and r2 >= trend_r2_threshold:
            return "TRENDING"
    return "MEAN REVERTING"


def summarize_products(
    prices: pd.DataFrame,
    trend_strength_threshold: float,
    trend_r2_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for product, product_prices in prices.groupby("product"):
        rows.append(
            {
                "Product": product,
                "Mean Price": product_prices["mid_price"].mean(),
                "Volatility": product_prices["returns"].std(),
                "Avg Spread": product_prices["spread"].mean(),
                "Return Autocorr": return_autocorrelation(product_prices["returns"]),
                "Market Type": classify_market_type(
                    product_prices,
                    trend_strength_threshold,
                    trend_r2_threshold,
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("Product").reset_index(drop=True)


def plot_products(prices: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for product, product_prices in prices.groupby("product"):
        ordered = product_prices.sort_values(["day", "timestamp"])
        product_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", product).strip("_").lower()

        fig, axis = plt.subplots(figsize=(12, 5))
        axis.plot(ordered["time_index"], ordered["mid_price"], linewidth=1)
        axis.set_title(f"{product} - Mid Price vs Time")
        axis.set_xlabel("Time index")
        axis.set_ylabel("Mid price")
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        path = output_dir / f"{product_slug}_mid_price_vs_time.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        generated.append(path)

        fig, axis = plt.subplots(figsize=(12, 5))
        axis.plot(ordered["time_index"], ordered["spread"], linewidth=1)
        axis.set_title(f"{product} - Spread vs Time")
        axis.set_xlabel("Time index")
        axis.set_ylabel("Spread")
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        path = output_dir / f"{product_slug}_spread_vs_time.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        generated.append(path)

    return generated


def format_summary(summary: pd.DataFrame) -> str:
    display = summary.copy()
    for column in ["Mean Price", "Volatility", "Avg Spread", "Return Autocorr"]:
        display[column] = display[column].map(lambda value: "nan" if pd.isna(value) else f"{value:.6g}")
    return display.to_string(index=False)


def main() -> int:
    args = parse_args()
    prices, loaded_files = load_price_data(args.input)
    if prices.empty:
        raise ValueError("No price CSV files found.")

    prices = add_basic_metrics(prices)
    summary = summarize_products(
        prices,
        trend_strength_threshold=args.trend_strength_threshold,
        trend_r2_threshold=args.trend_r2_threshold,
    )
    plot_paths = plot_products(prices, args.output_dir)

    print(f"Input: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print()
    print("Loaded price files")
    print("------------------")
    for file_name in loaded_files:
        print(f"- {file_name}")
    print()
    print("Summary")
    print("-------")
    print(format_summary(summary))
    print()
    print("Generated plots")
    print("---------------")
    for path in plot_paths:
        print(f"- {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
