# Prosperity 2026 Round 1 Product Behavior Analysis

This repo contains a focused Python script for classifying each product's basic
market behavior from the extracted Round 1 price CSV files in `ROUND_1/ROUND1`.

The script ignores trade files and does not include extra signals or trading
logic. It only computes the requested price characteristics and classifies each
product as `TRENDING` or `MEAN REVERTING`.

## Setup

The local `venv` already has the needed packages. You can run the script with:

```bash
venv/bin/python analysis/product_behavior_analysis.py
```

If you want to use your normal Python instead, install dependencies first:

```bash
python3 -m pip install -r requirements.txt
python3 analysis/product_behavior_analysis.py
```

## Input

By default, the script reads semicolon-separated price CSVs from:

```text
ROUND_1/ROUND1
```

You can pass a different extracted CSV directory or zip archive:

```bash
venv/bin/python analysis/product_behavior_analysis.py /path/to/ROUND1
venv/bin/python analysis/product_behavior_analysis.py /path/to/ROUND_1.zip
```

## Output

The script prints this summary table:

```text
Product | Mean Price | Volatility | Avg Spread | Return Autocorr | Market Type
```

It generates only these plots per product:

- Mid price vs time
- Spread vs time

Plots are saved to `analysis/behavior_output` by default. To choose another
folder:

```bash
venv/bin/python analysis/product_behavior_analysis.py --output-dir analysis/my-behavior-output
```

## Classification

For each product, the script computes:

- Returns: percentage change of `mid_price`
- Volatility: standard deviation of returns
- Spread: `ask_price_1 - bid_price_1`
- Average spread
- Lag-1 return autocorrelation

Products with a strong enough directional move and linear trend fit are labeled
`TRENDING`; otherwise they are labeled `MEAN REVERTING`.
