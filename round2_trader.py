"""Submission-safe Round 1 strategy.

This file intentionally contains only the Prosperity-facing Trader class and a
small set of helpers. No local backtester imports, no filesystem helpers, and no
debug harness live in the submitted algorithm.

Products:
    INTARIAN_PEPPER_ROOT: long-biased, but now adapts if the live trend rolls.
    ASH_COATED_OSMIUM: mean-reverting around 10000 with inventory targeting.
"""

import json
import math
from statistics import mean

from datamodel import Order, OrderDepth, TradingState


PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

POSITION_LIMITS = {
    PEPPER: 80,
    OSMIUM: 80,
}

# Pepper: the public days reveal an almost clock-driven long-term drift:
# mid ~= day_open + 0.001 * timestamp, with only a few ticks of noise. Use that
# as the broad pattern, and let recent residuals decide aggression.
PEPPER_DRIFT_PER_TIMESTAMP = 0.001
PEPPER_DAY_END_TIMESTAMP = 1_000_000
PEPPER_STRONG_TARGET_LONG = 80
PEPPER_CORE_TARGET_LONG = 55
PEPPER_WEAK_TARGET_LONG = 30
PEPPER_DEFENSIVE_TARGET = 0
PEPPER_SWEEP_LEVELS = 2
PEPPER_MAX_AGGRESSIVE_SIZE = 40
PEPPER_PASSIVE_SIZE = 12
PEPPER_FAST_WINDOW = 10
PEPPER_MEDIUM_WINDOW = 35
PEPPER_SLOW_WINDOW = 100
PEPPER_STRONG_SCORE = 1.00
PEPPER_WEAK_SCORE = 0.15
PEPPER_DEFENSIVE_SCORE = -0.70
PEPPER_CROSS_SCORE = 0.55
PEPPER_MAX_CROSS_SPREAD = 18
PEPPER_TERMINAL_EDGE_STRONG = 0.0
PEPPER_TERMINAL_EDGE_CORE = 0.0
PEPPER_OVER_FAIR_TRIM = 80.0
PEPPER_UNDER_FAIR_CROSS = 4.0
PEPPER_PROFIT_MIN_POSITION = 60
PEPPER_PROFIT_EDGE_MIN = 35
PEPPER_PROFIT_EDGE_SPREAD_MULTIPLIER = 4.0
PEPPER_PROFIT_SIZE = 6

# Osmium: 10000 is the anchor. The rolling adjustment is capped so the strategy
# can follow a small shift without chasing random midpoint noise.
OSMIUM_FAIR_VALUE = 10000.0
OSMIUM_FAIR_WINDOW = 160
OSMIUM_ROLLING_WEIGHT = 0.30
OSMIUM_ROLLING_CAP = 12.0
OSMIUM_TARGET_MULTIPLIER = 5.0
OSMIUM_TARGET_VOL_MULTIPLIER = 18.0
OSMIUM_REVERSAL_MULTIPLIER = 8.0
OSMIUM_TAKE_EDGE_MIN = 3.0
OSMIUM_TAKE_EDGE_VOL_MULTIPLIER = 1.25
OSMIUM_EXIT_EDGE = 0.0
OSMIUM_QUOTE_EDGE_MIN = 4.0
OSMIUM_QUOTE_EDGE_VOL_MULTIPLIER = 1.5
OSMIUM_INVENTORY_SKEW = 8.0
OSMIUM_SWEEP_LEVELS = 1
OSMIUM_MAX_AGGRESSIVE_SIZE = 24
OSMIUM_PASSIVE_SIZE = 14

MAX_HISTORY = 220


class BookSnapshot:
    def __init__(self, depth: OrderDepth):
        self.bids = sorted(depth.buy_orders.items(), reverse=True)
        self.asks = sorted((price, abs(volume)) for price, volume in depth.sell_orders.items())
        self.best_bid = self.bids[0][0] if self.bids else None
        self.best_ask = self.asks[0][0] if self.asks else None
        self.best_bid_volume = self.bids[0][1] if self.bids else 0
        self.best_ask_volume = self.asks[0][1] if self.asks else 0

        if self.best_bid is not None and self.best_ask is not None:
            self.mid_price = (self.best_bid + self.best_ask) / 2.0
            self.spread = self.best_ask - self.best_bid
        elif self.best_bid is not None:
            self.mid_price = float(self.best_bid)
            self.spread = None
        elif self.best_ask is not None:
            self.mid_price = float(self.best_ask)
            self.spread = None
        else:
            self.mid_price = None
            self.spread = None


class Trader:
    def __init__(self):
        self.mid_history = {PEPPER: [], OSMIUM: []}
        self.pepper_origin = None

    def bid(self):
        return 167

    def run(self, state: TradingState):
        self._load_trader_data(state.traderData)

        result = {}
        books = {}
        for product, depth in state.order_depths.items():
            book = BookSnapshot(depth)
            if book.mid_price is None:
                continue

            books[product] = book
            history = self.mid_history.setdefault(product, [])
            history.append(book.mid_price)
            del history[:-MAX_HISTORY]

        pepper_book = books.get(PEPPER)
        if pepper_book is not None:
            self._update_pepper_origin(pepper_book, state.timestamp)
            orders = self._trade_pepper(pepper_book, state.position.get(PEPPER, 0), state.timestamp)
            if orders:
                result[PEPPER] = orders

        osmium_book = books.get(OSMIUM)
        if osmium_book is not None:
            orders = self._trade_osmium(osmium_book, state.position.get(OSMIUM, 0))
            if orders:
                result[OSMIUM] = orders

        trader_data = json.dumps(
            {"mid_history": self.mid_history, "pepper_origin": self.pepper_origin},
            separators=(",", ":"),
        )
        return result, 0, trader_data

    def _trade_pepper(self, book: BookSnapshot, position: int, timestamp: int):
        history = self.mid_history.get(PEPPER, [])
        limit = POSITION_LIMITS[PEPPER]
        score = self._pepper_trend_score(history)
        fair_value = self._pepper_clock_fair(timestamp)
        terminal_fair = self._pepper_clock_fair(PEPPER_DAY_END_TIMESTAMP)
        residual = book.mid_price - fair_value if book.mid_price is not None else 0.0
        terminal_edge = terminal_fair - book.mid_price if book.mid_price is not None else 0.0
        target = self._pepper_target(score, residual, terminal_edge)
        buy_capacity = max(0, limit - position)
        sell_capacity = max(0, limit + position)
        orders = []
        buy_gap = max(0, target - position)
        sell_gap = max(0, position - target)

        if buy_gap > 0 and book.asks and self._pepper_should_cross(book, score, residual, terminal_edge):
            levels = PEPPER_SWEEP_LEVELS if score >= PEPPER_STRONG_SCORE else 1
            price, visible_volume = self._sweep_buy_limit(book, levels)
            max_size = PEPPER_MAX_AGGRESSIVE_SIZE if score >= PEPPER_CROSS_SCORE else PEPPER_PASSIVE_SIZE
            size = min(buy_gap, buy_capacity, max_size, visible_volume)
            before = buy_capacity
            buy_capacity = self._append_buy(orders, PEPPER, price, size, buy_capacity)
            buy_gap -= before - buy_capacity

        if buy_gap > 0 and buy_capacity > 0:
            passive_price = self._pepper_passive_bid_price(book, fair_value, score, residual)
            size = min(buy_gap, PEPPER_PASSIVE_SIZE, buy_capacity)
            buy_capacity = self._append_buy(orders, PEPPER, passive_price, size, buy_capacity)

        if sell_gap > 0 and book.best_bid is not None:
            size = min(sell_gap, PEPPER_MAX_AGGRESSIVE_SIZE, sell_capacity)
            sell_capacity = self._append_sell(orders, PEPPER, book.best_bid, size, sell_capacity)

        # Keep most trend exposure. Only offer a small slice at a wide edge, and
        # only when the regime is not defensive.
        if position >= PEPPER_PROFIT_MIN_POSITION and target > PEPPER_DEFENSIVE_TARGET and sell_capacity > 0:
            spread = book.spread if book.spread is not None else PEPPER_PROFIT_EDGE_MIN
            profit_edge = max(PEPPER_PROFIT_EDGE_MIN, spread * PEPPER_PROFIT_EDGE_SPREAD_MULTIPLIER)
            ask_price = self._passive_ask_price(book, book.mid_price + profit_edge)
            self._append_sell(orders, PEPPER, ask_price, PEPPER_PROFIT_SIZE, sell_capacity)

        return orders

    def _update_pepper_origin(self, book: BookSnapshot, timestamp: int) -> None:
        if self.pepper_origin is None and book.mid_price is not None:
            self.pepper_origin = book.mid_price - timestamp * PEPPER_DRIFT_PER_TIMESTAMP

    def _pepper_clock_fair(self, timestamp: int) -> float:
        origin = self.pepper_origin
        if origin is None:
            history = self.mid_history.get(PEPPER, [])
            origin = history[-1] if history else 0.0
        return origin + timestamp * PEPPER_DRIFT_PER_TIMESTAMP

    def _pepper_target(self, score: float, residual: float, terminal_edge: float) -> int:
        # The strongest public signal is not a fragile indicator; it is the
        # product's structural upward drift. Missing the 80-lot long is far more
        # expensive than occasionally paying a few ticks of spread, so keep the
        # default target at the long limit. Only step down if both the clock
        # model says price is very stretched and recent movement has genuinely
        # rolled over.
        if residual > PEPPER_OVER_FAIR_TRIM and score < PEPPER_DEFENSIVE_SCORE:
            return PEPPER_CORE_TARGET_LONG
        return PEPPER_STRONG_TARGET_LONG

    def _pepper_should_cross(self, book: BookSnapshot, score: float, residual: float, terminal_edge: float) -> bool:
        if residual <= -PEPPER_UNDER_FAIR_CROSS:
            return True
        if terminal_edge >= PEPPER_TERMINAL_EDGE_STRONG and residual <= PEPPER_OVER_FAIR_TRIM:
            if book.spread is None:
                return True
            return book.spread <= PEPPER_MAX_CROSS_SPREAD
        if score < PEPPER_CROSS_SCORE:
            return False
        if book.spread is None:
            return True
        return book.spread <= PEPPER_MAX_CROSS_SPREAD

    def _pepper_trend_score(self, history) -> float:
        if len(history) < 3:
            return PEPPER_STRONG_SCORE

        volatility = self._avg_abs_move(history, PEPPER_SLOW_WINDOW)
        fast = self._net_move(history, PEPPER_FAST_WINDOW) / volatility
        medium = self._net_move(history, PEPPER_MEDIUM_WINDOW) / volatility
        slow = self._net_move(history, PEPPER_SLOW_WINDOW) / volatility

        return fast * 0.45 + medium * 0.35 + slow * 0.20

    def _pepper_passive_bid_price(self, book: BookSnapshot, fair_value: float, score: float, residual: float) -> int:
        if book.best_bid is not None:
            raw_price = book.best_bid + 1
        elif book.best_ask is not None:
            raw_price = book.best_ask - 1
        else:
            raw_price = fair_value - 2

        edge = fair_value - book.mid_price if book.mid_price is not None else 0.0

        if edge >= 6:
            raw_price = max(raw_price, fair_value - 1)
        elif edge >= 3:
            raw_price = max(raw_price, fair_value - 2)

        if score >= PEPPER_STRONG_SCORE and book.best_bid is not None:
            raw_price = max(raw_price, book.best_bid + 1)

        price = int(math.floor(raw_price))

        if book.best_ask is not None:
            price = min(price, book.best_ask - 1)

        if book.best_bid is not None:
            price = max(price, book.best_bid)

        return price

    def _trade_osmium(self, book: BookSnapshot, position: int):
        fair_value = self._osmium_fair_value()
        noise = self._avg_abs_move(self.mid_history.get(OSMIUM, []), OSMIUM_FAIR_WINDOW)
        take_edge = max(OSMIUM_TAKE_EDGE_MIN, noise * OSMIUM_TAKE_EDGE_VOL_MULTIPLIER)
        quote_edge = max(OSMIUM_QUOTE_EDGE_MIN, noise * OSMIUM_QUOTE_EDGE_VOL_MULTIPLIER)
        limit = POSITION_LIMITS[OSMIUM]
        buy_capacity = max(0, limit - position)
        sell_capacity = max(0, limit + position)
        orders = []

        target = self._osmium_target_inventory(book, fair_value, noise)
        buy_gap = max(0, target - position)
        sell_gap = max(0, position - target)

        if buy_gap > 0 and book.best_ask is not None:
            if book.best_ask <= fair_value - take_edge:
                price, visible_volume = self._sweep_buy_limit(book, OSMIUM_SWEEP_LEVELS)
                size = min(
                    buy_gap,
                    buy_capacity,
                    OSMIUM_MAX_AGGRESSIVE_SIZE,
                    visible_volume,
                )
                before = buy_capacity
                buy_capacity = self._append_buy(orders, OSMIUM, price, size, buy_capacity)
                buy_gap -= before - buy_capacity

        if sell_gap > 0 and book.best_bid is not None:
            if book.best_bid >= fair_value + take_edge:
                price, visible_volume = self._sweep_sell_limit(book, OSMIUM_SWEEP_LEVELS)
                size = min(
                    sell_gap,
                    sell_capacity,
                    OSMIUM_MAX_AGGRESSIVE_SIZE,
                    visible_volume,
                )
                before = sell_capacity
                sell_capacity = self._append_sell(orders, OSMIUM, price, size, sell_capacity)
                sell_gap -= before - sell_capacity

        # Flatten toward target as soon as fair value is available at the top.
        if sell_gap > 0 and book.best_bid is not None and book.best_bid >= fair_value + OSMIUM_EXIT_EDGE:
            size = min(sell_gap, OSMIUM_MAX_AGGRESSIVE_SIZE, sell_capacity)
            before = sell_capacity
            sell_capacity = self._append_sell(orders, OSMIUM, book.best_bid, size, sell_capacity)
            sell_gap -= before - sell_capacity
        elif buy_gap > 0 and book.best_ask is not None and book.best_ask <= fair_value - OSMIUM_EXIT_EDGE:
            size = min(buy_gap, OSMIUM_MAX_AGGRESSIVE_SIZE, buy_capacity)
            before = buy_capacity
            buy_capacity = self._append_buy(orders, OSMIUM, book.best_ask, size, buy_capacity)
            buy_gap -= before - buy_capacity

        # Passive market making does most of the work. Skew moves both quotes
        # against our inventory so fills naturally pull us back toward flat.
        inventory_skew = (position / limit) * OSMIUM_INVENTORY_SKEW
        bid_price = self._passive_bid_price(book, fair_value - quote_edge - inventory_skew)
        ask_price = self._passive_ask_price(book, fair_value + quote_edge - inventory_skew)
        if bid_price < ask_price:
            buy_size = min(buy_capacity, max(OSMIUM_PASSIVE_SIZE // 3, buy_gap), self._osmium_passive_size(position, "buy"))
            sell_size = min(sell_capacity, max(OSMIUM_PASSIVE_SIZE // 3, sell_gap), self._osmium_passive_size(position, "sell"))
            self._append_buy(orders, OSMIUM, bid_price, buy_size, buy_capacity)
            self._append_sell(orders, OSMIUM, ask_price, sell_size, sell_capacity)

        return orders

    def _osmium_target_inventory(self, book: BookSnapshot, fair_value: float, noise: float) -> int:
        if book.mid_price is None:
            return 0

        edge = fair_value - book.mid_price
        raw_target = edge * OSMIUM_TARGET_MULTIPLIER
        if noise > 0:
            raw_target += (edge / noise) * OSMIUM_TARGET_VOL_MULTIPLIER
        raw_target -= self._last_move(self.mid_history.get(OSMIUM, [])) * OSMIUM_REVERSAL_MULTIPLIER
        target = int(round(raw_target))
        return self._clamp(target, -POSITION_LIMITS[OSMIUM], POSITION_LIMITS[OSMIUM])

    def _osmium_fair_value(self) -> float:
        history = self.mid_history.get(OSMIUM, [])
        if len(history) < OSMIUM_FAIR_WINDOW:
            return OSMIUM_FAIR_VALUE

        rolling_mean = mean(history[-OSMIUM_FAIR_WINDOW:])
        adjustment = self._clamp(rolling_mean - OSMIUM_FAIR_VALUE, -OSMIUM_ROLLING_CAP, OSMIUM_ROLLING_CAP)
        return OSMIUM_FAIR_VALUE + adjustment * OSMIUM_ROLLING_WEIGHT

    def _osmium_passive_size(self, position: int, side: str) -> int:
        limit = POSITION_LIMITS[OSMIUM]
        inventory_ratio = position / limit
        if side == "buy":
            scale = 1.0 - max(0.0, inventory_ratio)
            scale += max(0.0, -inventory_ratio) * 0.5
        else:
            scale = 1.0 + max(0.0, inventory_ratio) * 0.5
            scale -= max(0.0, -inventory_ratio)
        return max(1, int(round(OSMIUM_PASSIVE_SIZE * scale)))

    def _sweep_buy_limit(self, book: BookSnapshot, levels: int):
        chosen = book.asks[: max(1, levels)]
        return chosen[-1][0], sum(volume for _, volume in chosen)

    def _sweep_sell_limit(self, book: BookSnapshot, levels: int):
        chosen = book.bids[: max(1, levels)]
        return chosen[-1][0], sum(volume for _, volume in chosen)

    def _inside_bid(self, book: BookSnapshot) -> int:
        if book.best_bid is None:
            if book.best_ask is not None:
                return int(book.best_ask - 1)
            return int(math.floor(book.mid_price))
        price = book.best_bid + 1
        if book.best_ask is not None:
            price = min(price, book.best_ask - 1)
        return int(price)

    def _passive_bid_price(self, book: BookSnapshot, raw_price: float) -> int:
        price = int(math.floor(raw_price))
        if book.best_ask is not None:
            price = min(price, book.best_ask - 1)
        return price

    def _passive_ask_price(self, book: BookSnapshot, raw_price: float) -> int:
        price = int(math.ceil(raw_price))
        if book.best_bid is not None:
            price = max(price, book.best_bid + 1)
        return price

    def _append_buy(self, orders, product: str, price: int, desired_size: int, buy_capacity: int) -> int:
        quantity = max(0, min(int(desired_size), buy_capacity))
        if quantity > 0:
            orders.append(Order(product, int(price), quantity))
        return buy_capacity - quantity

    def _append_sell(self, orders, product: str, price: int, desired_size: int, sell_capacity: int) -> int:
        quantity = max(0, min(int(desired_size), sell_capacity))
        if quantity > 0:
            orders.append(Order(product, int(price), -quantity))
        return sell_capacity - quantity

    def _net_move(self, history, window: int) -> float:
        values = history[-min(window, len(history)):]
        if len(values) < 2:
            return 0.0
        return values[-1] - values[0]

    def _last_move(self, history) -> float:
        if len(history) < 2:
            return 0.0
        return history[-1] - history[-2]

    def _avg_abs_move(self, history, window: int) -> float:
        values = history[-min(window, len(history)):]
        if len(values) < 2:
            return 1.0

        total = 0.0
        count = 0
        previous = values[0]
        for value in values[1:]:
            total += abs(value - previous)
            count += 1
            previous = value
        return max(1.0, total / max(1, count))

    def _clamp(self, value, lower, upper):
        return max(lower, min(upper, value))

    def _load_trader_data(self, trader_data: str) -> None:
        if not trader_data:
            return

        try:
            payload = json.loads(trader_data)
        except Exception:
            return

        raw_history = payload.get("mid_history")
        if not isinstance(raw_history, dict):
            return

        raw_origin = payload.get("pepper_origin")
        if raw_origin is not None:
            try:
                self.pepper_origin = float(raw_origin)
            except Exception:
                self.pepper_origin = None

        for product, values in raw_history.items():
            if product not in (PEPPER, OSMIUM) or not isinstance(values, list):
                continue
            cleaned = []
            for value in values[-MAX_HISTORY:]:
                try:
                    cleaned.append(float(value))
                except Exception:
                    continue
            self.mid_history[product] = cleaned
