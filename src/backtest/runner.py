try:
    import vectorbt as vbt  # type: ignore
except Exception:
    vbt = None

import pandas as pd
import numpy as np
import inspect
from src.strategies.strategy_registry import get_strategy_function
from src.utils.config import config

"""
AgentQuant Backtesting Engine
-----------------------------
Handles strategy execution, portfolio simulation, and metrics generation.
Now includes robust default handling for missing configuration keys.
"""

def run_backtest(ohlcv_data, assets, strategy_name, params, allocation_weights=None):
    """Run a complete backtest for a given strategy and asset list."""

    # Handle single-asset input
    if isinstance(ohlcv_data, pd.DataFrame):
        if not assets or len(assets) != 1:
            print("Warning: A single DataFrame was provided but asset list mismatch.")
            return None
        ohlcv_dict = {assets[0]: ohlcv_data}
    else:
        ohlcv_dict = ohlcv_data or {}

    # Verify that all assets have data
    for asset in assets:
        if asset not in ohlcv_dict or ohlcv_dict[asset].empty:
            print(f"⚠️  Missing or empty OHLCV data for {asset}, skipping.")
            return None

    strategy_func = get_strategy_function(strategy_name)

    def _get_close_series(x: pd.DataFrame | pd.Series) -> pd.Series:
        if isinstance(x, pd.Series):
            return pd.to_numeric(x, errors="coerce").dropna()
        col_map = {str(c).lower(): c for c in x.columns}
        for cand in ("close", "adj close", "adjclose", "price"):
            if cand in col_map:
                return pd.to_numeric(x[col_map[cand]], errors="coerce").dropna()
        for col in x.columns:
            try:
                s = pd.to_numeric(x[col], errors="coerce")
                if s.notna().any():
                    return s.dropna()
            except Exception:
                continue
        raise KeyError(f"No close-like column found. Available: {list(x.columns)}")

    # Setup weights
    if allocation_weights is None:
        weights = {a: 1.0 / len(assets) for a in assets}
    else:
        total = sum(allocation_weights.values())
        weights = {a: allocation_weights.get(a, 0) / total for a in assets}

    # === Default-safe config extraction ===
    bt_cfg = config.get("backtest", {}) if isinstance(config, dict) else {}
    default_cash = bt_cfg.get("initial_cash", 100000)
    default_fee = bt_cfg.get("commission", 0.001)
    default_slip = bt_cfg.get("slippage", 0.0005)

    all_results = {}
    combined_portfolio_value = None

    def _normalize_params_for_strategy(name: str, func, params: dict) -> dict:
        """Normalize and filter parameters to match strategy signature."""
        p = dict(params or {})
        sig = inspect.signature(func)
        accepted = set(sig.parameters.keys())

        # Strategy-specific defaults
        if name == "breakout":
            if "threshold" in p and "threshold_pct" not in p:
                p["threshold_pct"] = p.pop("threshold")
            p.setdefault("window", 20)
            p.setdefault("threshold_pct", 0.02)
        elif name == "trend_following":
            if "window" in p and not ({"short_window", "medium_window", "long_window"} & set(p.keys())):
                w = int(p.pop("window") or 10)
                p["short_window"], p["medium_window"], p["long_window"] = max(2, w), max(w + 5, w * 2), max(w + 10, w * 3)
        elif name == "regime_based":
            p.pop("window", None)
            p.setdefault("regime_data", "neutral")
            p.setdefault("momentum_params", {"fast_window": 21, "slow_window": 63})
            p.setdefault("mean_reversion_params", {"window": 20, "num_std": 2.0})
        elif name == "volatility":
            p.setdefault("window", 21)
            p.setdefault("vol_threshold", 0.2)
        elif name == "mean_reversion":
            p.setdefault("window", 20)
            p.setdefault("num_std", 2.0)

        # Keep only accepted params
        filtered = {k: v for k, v in p.items() if k in accepted}
        return filtered

    # === Run backtest per asset ===
    for asset in assets:
        df = ohlcv_dict[asset]
        if df.empty:
            print(f"⚠️  Empty dataframe for {asset}.")
            continue

        print(f"\nRunning {strategy_name} for {asset}...")

        try:
            norm_params = _normalize_params_for_strategy(strategy_name, strategy_func, params)

            if strategy_name == "momentum":
                entries, exits = strategy_func(_get_close_series(df), **norm_params)
            else:
                signal = strategy_func(df, **norm_params).fillna(0)
                prev_pos, curr_pos = (signal.shift(1) > 0), (signal > 0)
                entries = (curr_pos & ~prev_pos).fillna(False)
                exits = (~curr_pos & prev_pos).fillna(False)
        except Exception as e:
            print(f"❌ Parameter mismatch or signal generation error: {e}")
            continue

        try:
            init_cash = default_cash * weights.get(asset, 1.0)
            commission = default_fee
            slippage = default_slip

            if vbt is not None:
                portfolio = vbt.Portfolio.from_signals(
                    close=_get_close_series(df),
                    entries=entries,
                    exits=exits,
                    freq="D",
                    init_cash=init_cash,
                    fees=commission,
                    slippage=slippage
                )
                pv = portfolio.value()
                all_results[asset] = {
                    "total_return": portfolio.total_return(),
                    "sharpe_ratio": portfolio.sharpe_ratio(),
                    "max_drawdown": portfolio.max_drawdown(),
                    "num_trades": portfolio.trades.count(),
                }
            else:
                close = _get_close_series(df).dropna()
                pos = pd.Series(0.0, index=close.index)
                pos[entries.reindex(close.index, fill_value=False)] = 1.0
                pos[exits.reindex(close.index, fill_value=False)] = 0.0
                pos = pos.ffill().fillna(0.0)
                daily_ret = close.pct_change().fillna(0.0)
                strat_ret = daily_ret * pos.shift(1).fillna(0.0)
                pv = pd.Series(init_cash, index=close.index) * (1.0 + strat_ret).cumprod()

                total_return = (pv.iloc[-1] / pv.iloc[0]) - 1 if len(pv) > 1 else 0.0
                sharpe_ratio = (strat_ret.mean() / (strat_ret.std() + 1e-12)) * np.sqrt(252.0)
                max_dd = abs((pv / pv.cummax() - 1).min())
                num_trades = int(entries.sum())

                all_results[asset] = {
                    "total_return": total_return,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_dd,
                    "num_trades": num_trades,
                }

            if combined_portfolio_value is None:
                combined_portfolio_value = pv
            else:
                common_idx = combined_portfolio_value.index.intersection(pv.index)
                if not common_idx.empty:
                    combined_portfolio_value = combined_portfolio_value.loc[common_idx] + pv.loc[common_idx]

        except Exception as e:
            print(f"❌ Error running backtest for {asset}: {e}")
            continue

    # === Combine metrics ===
    if combined_portfolio_value is not None and not combined_portfolio_value.empty:
        portfolio_returns = combined_portfolio_value.pct_change().dropna()
        sharpe_ratio = (
            (portfolio_returns.mean() / (portfolio_returns.std() + 1e-12)) * np.sqrt(252.0)
            if len(portfolio_returns) > 1
            else 0.0
        )
        max_drawdown = abs((combined_portfolio_value / combined_portfolio_value.cummax() - 1).min())

        metrics = {
            "total_return": (combined_portfolio_value.iloc[-1] / combined_portfolio_value.iloc[0]) - 1,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "num_trades": sum(res["num_trades"] for res in all_results.values()),
        }

        for asset, res in all_results.items():
            for k, v in res.items():
                metrics[f"{asset}_{k}"] = v

        return {
            "equity_curve": combined_portfolio_value,
            "weights": weights,
            "metrics": metrics,
        }

    print("⚠️ No combined portfolio value generated.")
    return None
