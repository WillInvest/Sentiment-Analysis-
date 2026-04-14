"""Backtest simple trading strategies driven by our sentiment models.

Three strategies, all on the 18-month out-of-sample period
(2019-07 → 2020-12) so the comparison is apples-to-apples:

  S1. PRETRAINED long-only
        Buy when confidence-weighted FinBERT (pos − neg) > 0.
  S2. FINE-TUNED regressor long-only
        Buy when predicted r_1d > 0.
  S3. FINE-TUNED regressor long-short
        Buy when predicted r_1d > 0, short-sell when predicted r_1d < 0.

Trade rules (same for all three):
  - 5-day maximum holding period.
  - Take profit:   exit when cumulative return reaches +3%.
  - Stop loss:     exit when cumulative return falls to -2%.
  - Otherwise:     exit at the close of day +5.
  - Equal weight across all positions, no leverage, no transaction costs.

Benchmark: buy SPX at the start of the test period and hold.

Outputs:
  results/metrics/trading_strategies.json
  results/figures/trading_pnl.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PRED_DIR = ROOT / "results/predictions"
PRICE = ROOT / "data/raw/price.csv"
OUT_JSON = ROOT / "results/metrics/trading_strategies.json"
OUT_FIG = ROOT / "results/figures/trading_pnl.png"

TAKE_PROFIT = 0.03    # exit when cumulative return ≥ +3%
STOP_LOSS = 0.02      # exit when cumulative return ≤ −2%
MAX_HOLD = 5          # trading days


# ---------- price loading ----------
def load_prices() -> tuple[dict[str, pd.Series], pd.DatetimeIndex]:
    px = pd.read_csv(PRICE, parse_dates=["Date"])
    prices: dict[str, pd.Series] = {}
    for tk, g in px.groupby("ticker"):
        prices[tk] = g.sort_values("Date").set_index("Date")["close"]
    spx = prices["SPX"]
    trading_days = spx.index
    return prices, trading_days


# ---------- single-trade simulator ----------
def simulate_trade(entry_date: pd.Timestamp, ticker: str, direction: int,
                   prices: dict[str, pd.Series],
                   take_profit: float, stop_loss: float, max_hold: int):
    """Returns (exit_date, realized_return) or None if the trade is invalid."""
    if ticker not in prices:
        return None
    s = prices[ticker]
    if entry_date not in s.index:
        return None
    entry_idx = s.index.get_loc(entry_date)
    entry_price = float(s.iloc[entry_idx])
    if entry_price <= 0:
        return None

    cum_ret = 0.0
    for offset in range(1, max_hold + 1):
        if entry_idx + offset >= len(s):
            return None  # not enough future data
        exit_price = float(s.iloc[entry_idx + offset])
        cum_ret = direction * (exit_price - entry_price) / entry_price
        if cum_ret >= take_profit or cum_ret <= -stop_loss:
            return (s.index[entry_idx + offset], cum_ret)

    return (s.index[entry_idx + max_hold], cum_ret)


# ---------- backtest ----------
def backtest(signals: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    """signals: columns (date, ticker, direction). Returns trades DataFrame."""
    trades = []
    for r in signals.itertuples(index=False):
        out = simulate_trade(r.date, r.ticker, r.direction, prices,
                              TAKE_PROFIT, STOP_LOSS, MAX_HOLD)
        if out is None:
            continue
        exit_date, ret = out
        trades.append({"entry_date": r.date, "exit_date": exit_date,
                       "ticker": r.ticker, "direction": r.direction,
                       "return": float(ret)})
    return pd.DataFrame(trades)


def equity_curve(trades: pd.DataFrame,
                 prices: dict[str, pd.Series],
                 trading_days: pd.DatetimeIndex) -> pd.Series:
    """Realistic equal-weight portfolio simulation.

    For each trading day t, take all trades active on day t (entry_date < t ≤ exit_date),
    compute each trade's day-t return from prices, and average them. That mean is the
    portfolio's return for the day. Days with no active trades contribute 0%.
    """
    if len(trades) == 0:
        return pd.Series(1.0, index=trading_days)

    daily_columns = []
    for t in trades.itertuples(index=False):
        s = prices.get(t.ticker)
        if s is None: continue
        if t.entry_date not in s.index or t.exit_date not in s.index: continue
        i_entry = s.index.get_loc(t.entry_date)
        i_exit = s.index.get_loc(t.exit_date)
        if i_exit <= i_entry: continue
        sub = s.iloc[i_entry:i_exit + 1]
        daily_ret = sub.pct_change().dropna() * t.direction  # length = i_exit - i_entry
        daily_columns.append(daily_ret)

    if not daily_columns:
        return pd.Series(1.0, index=trading_days)

    df = pd.concat(daily_columns, axis=1)
    portfolio_daily = df.mean(axis=1)                                  # equal weight
    portfolio_daily = portfolio_daily.reindex(trading_days, fill_value=0.0)
    return (1.0 + portfolio_daily).cumprod()


def stats_from_trades(trades: pd.DataFrame, equity: pd.Series) -> dict:
    if len(trades) == 0:
        return {"n_trades": 0, "win_rate": None, "total_return": None,
                "annualized_return": None, "sharpe": None, "max_drawdown": None}
    total_ret = float(equity.iloc[-1] - 1.0)
    n_days = len(equity)
    annualized = float((1.0 + total_ret) ** (252.0 / n_days) - 1.0) if n_days > 0 else 0.0
    daily = equity.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = float(drawdown.min())
    win_rate = float((trades["return"] > 0).mean())
    return {
        "n_trades": int(len(trades)),
        "win_rate": round(win_rate, 4),
        "total_return": round(total_ret, 4),
        "annualized_return": round(annualized, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
    }


# ---------- signal builders ----------
def load_concat(filename_pattern: str) -> pd.DataFrame:
    parts = []
    for w in ("w1", "w2", "w3"):
        df = pd.read_parquet(PRED_DIR / filename_pattern.format(w=w))
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def signals_from_pretrained() -> pd.DataFrame:
    df = load_concat("baseline_{w}_test.parquet")
    df = df[df["pred"] > 0].copy()
    df["direction"] = 1
    return df[["date", "ticker", "direction"]]


def signals_from_finetuned_long_only() -> pd.DataFrame:
    df = load_concat("regressor_{w}_test.parquet")
    df = df[df["pred"] > 0].copy()
    df["direction"] = 1
    return df[["date", "ticker", "direction"]]


def signals_from_finetuned_long_short() -> pd.DataFrame:
    df = load_concat("regressor_{w}_test.parquet")
    df = df[df["pred"] != 0].copy()
    df["direction"] = np.where(df["pred"] > 0, 1, -1)
    return df[["date", "ticker", "direction"]]


# ---------- benchmark: SPX buy & hold ----------
def spx_buy_and_hold(prices: dict[str, pd.Series],
                     start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, dict]:
    spx = prices["SPX"]
    mask = (spx.index >= start) & (spx.index <= end)
    s = spx[mask]
    eq = s / s.iloc[0]
    daily = eq.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    total = float(eq.iloc[-1] - 1.0)
    n_days = len(eq)
    annualized = float((1.0 + total) ** (252.0 / n_days) - 1.0)
    rolling_max = eq.cummax()
    max_dd = float(((eq - rolling_max) / rolling_max).min())
    info = {
        "n_trades": 1,
        "win_rate": None,
        "total_return": round(total, 4),
        "annualized_return": round(annualized, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
    }
    return eq, info


# ---------- main ----------
def main():
    print("Loading prices…")
    prices, all_trading_days = load_prices()

    print("Building signals…")
    sig_pretrained = signals_from_pretrained()
    sig_ft_long = signals_from_finetuned_long_only()
    sig_ft_ls = signals_from_finetuned_long_short()
    print(f"  pretrained long-only signals:    {len(sig_pretrained):,}")
    print(f"  fine-tuned long-only signals:    {len(sig_ft_long):,}")
    print(f"  fine-tuned long-short signals:   {len(sig_ft_ls):,}")

    test_start = min(sig_pretrained["date"].min(),
                     sig_ft_long["date"].min(),
                     sig_ft_ls["date"].min())
    test_end = max(sig_pretrained["date"].max(),
                   sig_ft_long["date"].max(),
                   sig_ft_ls["date"].max())
    print(f"  test period: {test_start.date()} → {test_end.date()}")

    backtest_days = all_trading_days[(all_trading_days >= test_start) &
                                      (all_trading_days <= test_end + pd.Timedelta(days=14))]

    print("\nBacktesting…")
    print("  S1: pretrained long-only")
    t1 = backtest(sig_pretrained, prices)
    eq1 = equity_curve(t1, prices, backtest_days)
    s1 = stats_from_trades(t1, eq1)

    print("  S2: fine-tuned long-only")
    t2 = backtest(sig_ft_long, prices)
    eq2 = equity_curve(t2, prices, backtest_days)
    s2 = stats_from_trades(t2, eq2)

    print("  S3: fine-tuned long-short")
    t3 = backtest(sig_ft_ls, prices)
    eq3 = equity_curve(t3, prices, backtest_days)
    s3 = stats_from_trades(t3, eq3)

    print("  Benchmark: SPX buy & hold")
    eq_spx, s_spx = spx_buy_and_hold(prices, backtest_days[0], backtest_days[-1])

    summary = {
        "rules": {
            "max_hold_days": MAX_HOLD,
            "take_profit": TAKE_PROFIT,
            "stop_loss": STOP_LOSS,
            "test_period": [str(backtest_days[0].date()), str(backtest_days[-1].date())],
        },
        "S1_pretrained_long_only": s1,
        "S2_finetuned_long_only": s2,
        "S3_finetuned_long_short": s3,
        "benchmark_SPX_buy_hold": s_spx,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {OUT_JSON}")

    # Equity curves
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(eq1.index, eq1.values, label="S1: Pretrained long-only", color="#4C72B0", lw=1.6)
    ax.plot(eq2.index, eq2.values, label="S2: Fine-tuned long-only", color="#55A868", lw=1.6)
    ax.plot(eq3.index, eq3.values, label="S3: Fine-tuned long-short", color="#C44E52", lw=1.6)
    ax.plot(eq_spx.index, eq_spx.values, label="Benchmark: SPX buy & hold",
            color="black", lw=1.4, ls="--")
    ax.axhline(1.0, color="gray", lw=0.5)
    ax.set_xlabel("date")
    ax.set_ylabel("equity (normalized to 1.0 at start)")
    ax.set_title(f"Trading strategy equity curves\n"
                 f"hold≤{MAX_HOLD}d, TP={int(TAKE_PROFIT*100)}%, SL={int(STOP_LOSS*100)}%, "
                 f"equal-weight, no costs")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140)
    print(f"  wrote {OUT_FIG}")

    print("\nResults:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
