from __future__ import annotations

"""Round 1 trading strategy for the existing Prosperity backtester.

The backtester imports this file, instantiates Trader(), and calls run(state)
once per timestamp. The strategy returns:
    1. orders: dict[product, list[Order]]
    2. conversions: 0, unused for these products
    3. traderData: JSON string persisted into the next timestamp

Product assumptions from the data analysis:
    INTARIAN_PEPPER_ROOT: upward trend, trade with momentum and long bias.
    ASH_COATED_OSMIUM: sideways/mean reverting, quote around fair value.
"""

import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Iterable


def _add_local_backtester_to_path() -> None:
    # This makes the file runnable from the strategy repo as well as importable
    # by the backtester CLI. If you install the backtester elsewhere, set
    # PROSPERITY4BT_ROOT to that folder.
    env_root = os.environ.get("PROSPERITY4BT_ROOT")
    candidates = [Path(env_root).expanduser()] if env_root else []
    candidates.append(Path(__file__).resolve().parent.parent / "imc-prosperity-4-backtester")

    for candidate in candidates:
        if (candidate / "prosperity4bt").is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            return


_add_local_backtester_to_path()

try:
    # Preferred path when this file is run with the local prosperity4bt package.
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState
except ImportError:
    # Fallback for the official competition environment, where datamodel.py is
    # often placed beside the submitted algorithm.
    from datamodel import Order, OrderDepth, TradingState


PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

# Key tuning parameters. Start here when changing behavior.
# LOOKBACK_WINDOW: number of recent mids used for pepper trend and ash fair value.
# ORDER_SIZE: base quantity per quote/order before inventory adjustment.
# POSITION_LIMIT: hard limit used by our own risk checks and the backtester.
# SPREAD_THRESHOLD: below this spread, ash quotes are widened a little.
# TREND_THRESHOLD: pepper signal needed for passive long exposure.
# STRONG_TREND_THRESHOLD: pepper signal needed to lift the best ask.
# FAIR_VALUE_WINDOW: rolling mean window for ash fair value.
# INVENTORY_SKEW_STRENGTH: price skew for ash quotes as inventory grows.
LOOKBACK_WINDOW = 45
ORDER_SIZE = 8
POSITION_LIMIT = 80
SPREAD_THRESHOLD = 4
TREND_THRESHOLD = 0.4
STRONG_TREND_THRESHOLD = 1.5
FAIR_VALUE_WINDOW = 45
INVENTORY_SKEW_STRENGTH = 4.0

# Secondary tuning parameters. These are less likely to need first-pass edits.
# QUOTE_WIDTH_FACTOR: fraction of observed spread used as ash quote distance.
# MIN_QUOTE_WIDTH: minimum ticks away from fair value for ash quotes.
# PEPPER_REVERSAL_THRESHOLD: negative signal needed before pepper sells hard.
# PEPPER_TAKE_PROFIT_INVENTORY: fraction of limit where pepper starts offering out.
# PEPPER_TAKE_PROFIT_TICKS: ticks above mid for pepper profit-taking asks.
# MAX_HISTORY: cap on traderData size; must be larger than the largest window.
QUOTE_WIDTH_FACTOR = 0.35
MIN_QUOTE_WIDTH = 2.0
PEPPER_REVERSAL_THRESHOLD = 3.0
PEPPER_TAKE_PROFIT_INVENTORY = 0.85
PEPPER_TAKE_PROFIT_TICKS = 6
MAX_HISTORY = 120

POSITION_LIMITS = {
    PEPPER: POSITION_LIMIT,
    OSMIUM: POSITION_LIMIT,
}


@dataclass
class StrategyConfig:
    """Container for all strategy knobs so tuning can swap values cleanly."""

    pepper_lookback: int = LOOKBACK_WINDOW
    pepper_order_size: int = ORDER_SIZE
    pepper_trend_threshold: float = TREND_THRESHOLD
    pepper_strong_trend_threshold: float = STRONG_TREND_THRESHOLD
    pepper_reversal_threshold: float = PEPPER_REVERSAL_THRESHOLD
    pepper_take_profit_inventory: float = PEPPER_TAKE_PROFIT_INVENTORY
    pepper_take_profit_ticks: int = PEPPER_TAKE_PROFIT_TICKS

    ash_fair_value_window: int = FAIR_VALUE_WINDOW
    ash_order_size: int = ORDER_SIZE
    ash_spread_threshold: int = SPREAD_THRESHOLD
    ash_quote_width_factor: float = QUOTE_WIDTH_FACTOR
    ash_min_quote_width: float = MIN_QUOTE_WIDTH

    inventory_skew_strength: float = INVENTORY_SKEW_STRENGTH
    position_limits: dict[str, int] = field(default_factory=lambda: POSITION_LIMITS.copy())


@dataclass
class BookSnapshot:
    """Small normalized view of the top of book.

    The raw backtester order depth stores sell quantities as negative numbers.
    This object converts the best ask volume back to a positive quantity and
    gives every strategy method the same compact inputs.
    """

    best_bid: int | None
    best_ask: int | None
    best_bid_volume: int
    best_ask_volume: int
    mid_price: float | None
    spread: int | None


DEFAULT_CONFIG = StrategyConfig()


class Trader:
    def __init__(self, config: StrategyConfig | None = None):
        self.config = config or DEFAULT_CONFIG

        # The backtester may recreate or reload Trader between test runs, but
        # within a run traderData is passed back each timestamp. We keep the
        # in-memory copy and also serialize it so both execution styles work.
        self.mid_history: dict[str, list[float]] = {PEPPER: [], OSMIUM: []}

        # Keep enough history for the largest configured signal, plus slack.
        # This prevents traderData from growing for the full day.
        self.max_history = max(
            MAX_HISTORY,
            self.config.pepper_lookback + 5,
            self.config.ash_fair_value_window + 5,
        )

    def run(self, state: TradingState):
        # Restore rolling price history from the previous timestamp.
        self._load_trader_data(state.traderData)

        result: dict[str, list[Order]] = {}
        for product, depth in state.order_depths.items():
            # Convert the raw OrderDepth into best bid/ask, spread, and mid.
            book = self._book_snapshot(depth)
            if book.mid_price is None:
                # No quoteable price means no signal and no safe order.
                continue

            # Update rolling mid-price history before calculating this tick's
            # signal, so the newest market state is included.
            history = self.mid_history.setdefault(product, [])
            history.append(book.mid_price)
            del history[:-self.max_history]

            # Positions are maintained by the backtester after matching.
            position = state.position.get(product, 0)
            if product == PEPPER:
                orders = self._trade_pepper(product, book, position, history)
            elif product == OSMIUM:
                orders = self._trade_osmium(product, book, position, history)
            else:
                orders = []

            if orders:
                result[product] = orders

        # traderData must be a string. JSON is easy to inspect in logs and lets
        # us carry rolling history without relying only on object state.
        trader_data = json.dumps({"mid_history": self.mid_history}, separators=(",", ":"))
        return result, 0, trader_data

    def _trade_pepper(
        self,
        product: str,
        book: BookSnapshot,
        position: int,
        history: list[float],
    ) -> list[Order]:
        """Trend-following with a long bias for the structurally rising product."""
        config = self.config
        limit = config.position_limits[product]

        # Capacity is how many more lots we can buy/sell without breaching the
        # absolute position limit. All order appends consume this capacity.
        buy_capacity = max(0, limit - position)
        sell_capacity = max(0, limit + position)
        trend = self._pepper_trend_signal(history, config.pepper_lookback)
        orders: list[Order] = []

        # Strong positive momentum gets paid at the current ask. We scale size down
        # as inventory approaches the long limit so one fill cannot invalidate limits.
        if trend >= config.pepper_strong_trend_threshold and book.best_ask is not None:
            size = self._skewed_size(config.pepper_order_size, position, limit, side="buy")
            if book.best_ask_volume > 0:
                # Only cross for visible size at the best ask. The helper below
                # still clips against our remaining position capacity.
                size = min(size, book.best_ask_volume)
            buy_capacity = self._append_buy(orders, product, book.best_ask, size, buy_capacity)

        # Weaker positive momentum still wants long exposure, but passively below
        # mid so the strategy is not constantly lifting a wide spread.
        elif trend >= config.pepper_trend_threshold:
            bid_price = self._passive_bid_price(book, book.mid_price - 1)
            size = self._skewed_size(config.pepper_order_size, position, limit, side="buy")
            buy_capacity = self._append_buy(orders, product, bid_price, size, buy_capacity)

        # Only react to a clear reversal. If already long, this trims inventory;
        # otherwise it allows a small short but avoids fighting the known drift.
        elif trend <= -config.pepper_reversal_threshold and book.best_bid is not None:
            size = self._skewed_size(config.pepper_order_size, position, limit, side="sell")
            if book.best_bid_volume > 0:
                size = min(size, book.best_bid_volume)
            if position > 0:
                # A reversal while long is more about reducing exposure than
                # building a big short, so prioritize trimming existing longs.
                size = min(size * 2, abs(position))
            sell_capacity = self._append_sell(orders, product, book.best_bid, size, sell_capacity)

        # Near the long limit, stop adding risk and leave a profit-taking ask above
        # fair value. This usually rests unless price keeps moving up.
        near_max_long = position >= int(limit * config.pepper_take_profit_inventory)
        if near_max_long and sell_capacity > 0:
            ask_price = self._passive_ask_price(book, book.mid_price + config.pepper_take_profit_ticks)
            sell_size = min(config.pepper_order_size, sell_capacity)
            self._append_sell(orders, product, ask_price, sell_size, sell_capacity)

        return orders

    def _trade_osmium(
        self,
        product: str,
        book: BookSnapshot,
        position: int,
        history: list[float],
    ) -> list[Order]:
        """Mean-reversion market making around a rolling fair value."""
        config = self.config
        limit = config.position_limits[product]

        # Market making posts both sides, but each side is clipped separately so
        # simultaneous buy and sell orders cannot exceed the hard limit.
        buy_capacity = max(0, limit - position)
        sell_capacity = max(0, limit + position)

        # Fair value is deliberately simple: ash is mostly sideways, so a rolling
        # mean is enough for first-pass mean reversion.
        fair_value = self._rolling_mean(history, config.ash_fair_value_window)

        # Wider observed spreads allow wider quotes. Tight spreads get an extra
        # tick of caution so we do not overtrade a low-edge book.
        observed_spread = book.spread if book.spread is not None else config.ash_spread_threshold
        quote_width = max(config.ash_min_quote_width, observed_spread * config.ash_quote_width_factor)
        if observed_spread < config.ash_spread_threshold:
            quote_width += 1.0

        # Positive inventory lowers both quotes: buys become less aggressive and
        # sells become more aggressive. Negative inventory does the opposite.
        inventory_skew = (position / limit) * config.inventory_skew_strength
        bid_price = self._passive_bid_price(book, fair_value - quote_width - inventory_skew)
        ask_price = self._passive_ask_price(book, fair_value + quote_width - inventory_skew)

        # If rounding or a crossed market collapses the quotes, rebuild a clean
        # one-tick minimum spread around fair value.
        if bid_price >= ask_price:
            bid_price = math.floor(fair_value - quote_width)
            ask_price = math.ceil(fair_value + quote_width)
            if bid_price >= ask_price:
                ask_price = bid_price + 1

        buy_size = self._skewed_size(config.ash_order_size, position, limit, side="buy")
        sell_size = self._skewed_size(config.ash_order_size, position, limit, side="sell")

        orders: list[Order] = []
        # The append helpers perform the last risk check by clipping each order
        # to the remaining side-specific capacity.
        buy_capacity = self._append_buy(orders, product, bid_price, buy_size, buy_capacity)
        self._append_sell(orders, product, ask_price, sell_size, sell_capacity)
        return orders

    def _pepper_trend_signal(self, history: list[float], lookback: int) -> float:
        """Return a positive number for upward momentum and negative for reversal."""
        if len(history) < 3:
            return 0.0

        window = history[-min(lookback, len(history)) :]
        fast_count = min(len(window), max(3, lookback // 3))

        # Combine two simple momentum views:
        # - fast_mean - slow_mean captures recent price level vs the window.
        # - slope * fast_count captures directional drift over the window.
        fast_mean = mean(window[-fast_count:])
        slow_mean = mean(window)
        slope = (window[-1] - window[0]) / max(1, len(window) - 1)
        return (fast_mean - slow_mean) + slope * fast_count

    def _rolling_mean(self, history: list[float], window: int) -> float:
        """Simple fair value estimate for the mean-reverting product."""
        values = history[-min(window, len(history)) :]
        return mean(values)

    def _book_snapshot(self, depth: OrderDepth) -> BookSnapshot:
        # Competition order books represent asks with negative quantity. Prices
        # are normal dict keys, so best bid is max buy price and best ask is min
        # sell price.
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        best_bid_volume = depth.buy_orders.get(best_bid, 0) if best_bid is not None else 0
        best_ask_volume = abs(depth.sell_orders.get(best_ask, 0)) if best_ask is not None else 0

        # Some rows may only have one side of the book. Use that visible side as
        # the mid fallback so history remains continuous.
        if best_bid is not None and best_ask is not None:
            mid_price = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid
        elif best_bid is not None:
            mid_price = float(best_bid)
            spread = None
        elif best_ask is not None:
            mid_price = float(best_ask)
            spread = None
        else:
            mid_price = None
            spread = None

        return BookSnapshot(best_bid, best_ask, best_bid_volume, best_ask_volume, mid_price, spread)

    def _passive_bid_price(self, book: BookSnapshot, raw_price: float) -> int:
        # A passive bid must be strictly below the best ask, otherwise it would
        # become an aggressive buy and cross the spread.
        price = math.floor(raw_price)
        if book.best_ask is not None:
            price = min(price, book.best_ask - 1)
        return price

    def _passive_ask_price(self, book: BookSnapshot, raw_price: float) -> int:
        # A passive ask must be strictly above the best bid, otherwise it would
        # become an aggressive sell and cross the spread.
        price = math.ceil(raw_price)
        if book.best_bid is not None:
            price = max(price, book.best_bid + 1)
        return price

    def _skewed_size(self, base_size: int, position: int, limit: int, side: str) -> int:
        # Size skew is gentler than price skew. It slows further accumulation
        # near a limit, but still leaves a small order working for liquidity.
        inventory_ratio = position / limit
        if side == "buy":
            # Long inventory reduces buy size; short inventory increases it.
            scale = 1.0 - max(0.0, inventory_ratio) * 0.75
            scale += max(0.0, -inventory_ratio) * 0.5
        else:
            # Long inventory increases sell size; short inventory reduces it.
            scale = 1.0 + max(0.0, inventory_ratio) * 0.5
            scale -= max(0.0, -inventory_ratio) * 0.75

        scaled = int(round(base_size * scale))
        # Keep the strategy conservative: never more than 2x base size.
        return max(1, min(base_size * 2, scaled))

    def _append_buy(
        self,
        orders: list[Order],
        product: str,
        price: int | float,
        desired_size: int,
        buy_capacity: int,
    ) -> int:
        # Positive quantity means buy. Return remaining capacity so callers can
        # safely place a second buy order later in the same timestamp.
        quantity = max(0, min(int(desired_size), buy_capacity))
        if quantity > 0:
            orders.append(Order(product, int(price), quantity))
        return buy_capacity - quantity

    def _append_sell(
        self,
        orders: list[Order],
        product: str,
        price: int | float,
        desired_size: int,
        sell_capacity: int,
    ) -> int:
        # Negative quantity means sell. The backtester rejects the entire product
        # if aggregate orders breach limits, so we clip before appending.
        quantity = max(0, min(int(desired_size), sell_capacity))
        if quantity > 0:
            orders.append(Order(product, int(price), -quantity))
        return sell_capacity - quantity

    def _load_trader_data(self, trader_data: str) -> None:
        # The official interface gives traderData back exactly as returned from
        # the previous run call. Ignore malformed data instead of failing.
        if not trader_data:
            return

        try:
            payload = json.loads(trader_data)
        except json.JSONDecodeError:
            return

        raw_history = payload.get("mid_history")
        if not isinstance(raw_history, dict):
            return

        loaded: dict[str, list[float]] = {}
        for product_name, values in raw_history.items():
            if isinstance(product_name, str) and isinstance(values, list):
                loaded[product_name] = [float(value) for value in values[-self.max_history :]]

        if loaded:
            self.mid_history.update(loaded)


def _ensure_backtester_importable() -> None:
    # Used only by local tuning helpers. The backtester CLI handles imports on
    # its own when it loads this as an algorithm file.
    if "prosperity4bt" in sys.modules:
        return

    _add_local_backtester_to_path()


def _run_backtests(config: StrategyConfig, days: Iterable[int] = (-2, -1, 0)):
    # Programmatic backtest runner used by tune_parameters(). It reuses the same
    # TestRunner class as the CLI instead of implementing a separate simulator.
    _ensure_backtester_importable()

    from prosperity4bt.models.test_options import TradeMatchingMode
    from prosperity4bt.test_runner import TestRunner
    from prosperity4bt.tools.data_reader import PackageResourcesReader

    reader = PackageResourcesReader()
    results = []
    for day in days:
        # A fresh Trader per day matches the CLI behavior and prevents history
        # from leaking across independent day tests.
        runner = TestRunner(
            Trader(config),
            reader,
            round=1,
            day=day,
            show_progress_bar=False,
            print_output=False,
            trade_matching_mode=TradeMatchingMode.worse,
        )
        results.append(runner.run())
    return results


def _summarize_results(results) -> dict:
    # SummaryPrinter already reports PnL, but this helper also reconstructs max
    # inventory and counts own trades so tuning output is more useful.
    pnl_by_product: dict[str, float] = {}
    trade_counts: dict[str, int] = {}
    max_inventory: dict[str, int] = {}

    for result in results:
        # Final activity rows include mark-to-market PnL by product.
        for activity in result.final_activities():
            pnl_by_product[activity.symbol] = pnl_by_product.get(activity.symbol, 0.0) + activity.profit_loss

        # Rebuild inventory from own trades because the BacktestResult does not
        # store the full position path directly.
        day_positions = {PEPPER: 0, OSMIUM: 0}
        day_max_inventory = {PEPPER: 0, OSMIUM: 0}
        for trade_row in result.trades:
            trade = trade_row.trade
            if trade.buyer == "SUBMISSION":
                day_positions[trade.symbol] = day_positions.get(trade.symbol, 0) + trade.quantity
                trade_counts[trade.symbol] = trade_counts.get(trade.symbol, 0) + 1
            elif trade.seller == "SUBMISSION":
                day_positions[trade.symbol] = day_positions.get(trade.symbol, 0) - trade.quantity
                trade_counts[trade.symbol] = trade_counts.get(trade.symbol, 0) + 1
            else:
                continue

            day_max_inventory[trade.symbol] = max(
                day_max_inventory.get(trade.symbol, 0),
                abs(day_positions[trade.symbol]),
            )

        for product_name, inventory in day_max_inventory.items():
            max_inventory[product_name] = max(max_inventory.get(product_name, 0), inventory)

    return {
        "final_pnl": sum(pnl_by_product.values()),
        "pnl_by_product": pnl_by_product,
        "max_inventory": max_inventory,
        "trade_counts": trade_counts,
    }


def _save_pnl_timeseries(results, output_path: Path) -> None:
    # Lightweight output that can be plotted later without opening the visualizer.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["day", "timestamp", "product", "profit_and_loss"])
        for result in results:
            for activity in result.activity_logs:
                writer.writerow([result.day_num, activity.timestamp, activity.symbol, activity.profit_loss])


def tune_parameters() -> None:
    """Small local grid search for lookback/fair-value windows and quote widths."""
    # Keep the grid intentionally small. The goal is a fast sanity check, not an
    # exhaustive optimizer that overfits three historical days.
    lookbacks = [20, 30, 45]
    quote_width_factors = [0.25, 0.35, 0.45]
    summaries = []

    print("Tuning round 1 strategy on days -2, -1, 0")
    for lookback, quote_width_factor in product(lookbacks, quote_width_factors):
        config = replace(
            DEFAULT_CONFIG,
            pepper_lookback=lookback,
            ash_fair_value_window=lookback,
            ash_quote_width_factor=quote_width_factor,
        )
        results = _run_backtests(config)
        summary = _summarize_results(results)
        # Store the raw results for the best config so we can save its PnL path.
        summaries.append((summary["final_pnl"], lookback, quote_width_factor, config, summary, results))
        print(
            f"lookback={lookback:>2}, quote_width_factor={quote_width_factor:.2f} "
            f"=> final PnL={summary['final_pnl']:,.0f}"
        )

    best_pnl, best_lookback, best_quote_width, best_config, best_summary, best_results = max(
        summaries,
        key=lambda row: row[0],
    )

    output_path = Path("backtests") / "round1_best_pnl_timeseries.csv"
    _save_pnl_timeseries(best_results, output_path)

    print("\nBest parameter set")
    print(f"lookback={best_lookback}, quote_width_factor={best_quote_width:.2f}")
    print(f"Final PnL: {best_pnl:,.0f}")
    print("PnL per product:")
    for product_name, pnl in sorted(best_summary["pnl_by_product"].items()):
        print(f"  {product_name}: {pnl:,.0f}")
    print("Max inventory reached:")
    for product_name, inventory in sorted(best_summary["max_inventory"].items()):
        print(f"  {product_name}: {inventory}")
    print("Number of own trades:")
    for product_name, trade_count in sorted(best_summary["trade_counts"].items()):
        print(f"  {product_name}: {trade_count}")
    print(f"Saved PnL time series to {output_path}")


if __name__ == "__main__":
    tune_parameters()
