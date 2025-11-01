# src/backtest/simple_backtest.py
import pandas as pd
import numpy as np
from typing import Dict, Any

def basic_momentum_backtest(ohlcv_df: pd.DataFrame, params: Dict[str, Any]):
    """
    Simple deterministic 2-moving-average momentum backtest used as fallback.
    Expects ohlcv_df to contain a close-like column.
    Returns a dict with 'equity_curve' (pd.Series) and 'metrics'.
    """
    window_short = int(params.get('fast_window', params.get('short_window', 10)))
    window_long = int(params.get('slow_window', params.get('long_window', 50)))
    init_cash = float(params.get('initial_cash', 100000))
    commission = float(params.get('commission', 0.001))

    # find close series
    if isinstance(ohlcv_df.columns, pd.MultiIndex):
        # pick first 'Close' col in MultiIndex
        for col in ohlcv_df.columns:
            if str(col[0]).lower() == 'close':
                close = pd.to_numeric(ohlcv_df[col], errors='coerce').dropna()
                break
        else:
            close = pd.to_numeric(ohlcv_df.iloc[:, 0], errors='coerce').dropna()
    else:
        # try common names
        lowermap = {str(c).lower(): c for c in ohlcv_df.columns}
        for cand in ('close', 'adj close', 'adjclose'):
            if cand in lowermap:
                close = pd.to_numeric(ohlcv_df[lowermap[cand]], errors='coerce').dropna()
                break
        else:
            close = pd.to_numeric(ohlcv_df.iloc[:, 0], errors='coerce').dropna()

    if close.empty or len(close) < max(window_short, window_long) + 2:
        return None

    ma_short = close.rolling(window_short).mean()
    ma_long = close.rolling(window_long).mean()
    signal = (ma_short > ma_long).astype(int)
    entries = (signal.shift(1) == 0) & (signal == 1)
    exits = (signal.shift(1) == 1) & (signal == 0)

    pos = pd.Series(0.0, index=close.index)
    pos[entries] = 1.0
    pos[exits] = 0.0
    pos = pos.ffill().fillna(0.0)
    daily_ret = close.pct_change().fillna(0.0)
    strat_ret = daily_ret * pos.shift(1).fillna(0.0)
    pv = pd.Series(init_cash, index=close.index) * (1.0 + strat_ret).cumprod()

    total_return = (pv.iloc[-1] / pv.iloc[0]) - 1 if len(pv) > 1 else 0.0
    try:
        sharpe = (strat_ret.mean() / (strat_ret.std() + 1e-12)) * np.sqrt(252.0)
    except Exception:
        sharpe = None
    max_dd = abs((pv / pv.cummax() - 1).min())

    metrics = {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'num_trades': int(entries.sum())
    }

    return {'equity_curve': pv, 'metrics': metrics, 'weights': {close.name: 1.0}}
