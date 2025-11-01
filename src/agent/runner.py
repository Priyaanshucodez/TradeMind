import os
import logging
from dotenv import load_dotenv
from typing import Any, Dict, Optional, List
import pandas as pd
from datetime import datetime

from src.utils.config import config
from src.utils.logging import setup_logging
from src.data.ingest import fetch_ohlcv_data
from src.features.engine import compute_features
from src.features.regime import detect_regime
from src.backtest.runner import run_backtest
from src.backtest.simple_backtest import basic_momentum_backtest
from src.agent.planner import propose_actions
from src.agent.policy import select_best_proposal
from src.utils.backtest_utils import normalize_backtest_results

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def _first_of(d: Dict[str, Any], candidates):
    for k in candidates:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _to_percent_if_decimal(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if abs(v) <= 1.5:
        return v * 100.0
    return v


def _extract_standard_metrics(raw_results: Any) -> Dict[str, Any]:
    """Extract and normalize key metrics from any backtest result object."""
    if raw_results is None:
        return {}

    norm = normalize_backtest_results(raw_results)
    out: Dict[str, Any] = {}

    total_val = _first_of(norm, ['Total Return [%]', 'total_return', 'cumulative_return'])
    if total_val is not None:
        out['Total Return [%]'] = _to_percent_if_decimal(total_val)

    sharpe_val = _first_of(norm, ['Sharpe Ratio', 'sharpe', 'sharpe_ratio'])
    if sharpe_val is not None:
        try:
            out['Sharpe Ratio'] = float(sharpe_val)
        except Exception:
            pass

    md_val = _first_of(norm, ['max_drawdown', 'Max Drawdown [%]'])
    if md_val is not None:
        out['Max Drawdown [%]'] = _to_percent_if_decimal(md_val)

    num_trades = _first_of(norm, ['num_trades', 'Num Trades'])
    if num_trades is not None:
        out['Num Trades'] = int(float(num_trades))

    if 'params' in norm:
        out['params'] = str(norm['params'])

    out['_raw'] = norm
    return out


def _safe_run_backtest(ohlcv_df: pd.DataFrame, asset_ticker: str, strategy_name: str, params: dict):
    """Attempt main backtest, fallback to simple backtest if fails."""
    if ohlcv_df is None or ohlcv_df.empty:
        logger.error(f"No data available for {asset_ticker}. Skipping backtest.")
        return None

    results = None
    try:
        results = run_backtest(ohlcv_df, asset_ticker, strategy_name, params)
        if results is None:
            raise ValueError("run_backtest returned None")
    except Exception as e:
        logger.warning(f"[{strategy_name}] Primary backtest failed for {asset_ticker}: {e}", exc_info=True)
        logger.info(f"Falling back to basic momentum backtest for {asset_ticker}.")

    # Fallback backtest
    if results is None:
        try:
            results = basic_momentum_backtest(ohlcv_df, params)
        except Exception as e:
            logger.error(f"[{strategy_name}] Fallback backtest also failed: {e}", exc_info=True)
            return None

    # Normalize output
    try:
        return normalize_backtest_results(results)
    except Exception as e:
        logger.error(f"Normalization failed for {asset_ticker}: {e}", exc_info=True)
        return None


def _print_table(df: pd.DataFrame, cols: List[str]):
    """Pretty print for CLI and Markdown logging."""
    try:
        print(df[cols].to_markdown(floatfmt=".2f"))
    except Exception:
        print(df[cols].to_string(float_format=lambda x: f"{x:.2f}"))


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------
def main():
    load_dotenv()
    logger.info("🚀 Starting AgentQuant run...")

    # Step 1: Data ingestion
    try:
        ohlcv_data = fetch_ohlcv_data()
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return

    ref_asset = config.get('reference_asset', None)
    if not ref_asset or ref_asset not in ohlcv_data:
        logger.error(f"Reference asset '{ref_asset}' not found in data.")
        return

    # Step 2: Feature Engineering and Regime Detection
    logger.info("🧮 Computing features and detecting regime...")
    features_df = compute_features(ohlcv_data, ref_asset, config['vix_ticker'])
    current_regime = detect_regime(features_df)
    logger.info(f"📈 Detected Market Regime: {current_regime}")

    # Step 3: Baseline Backtest
    baseline_strategy = config['strategies'][0]
    logger.info(f"Running baseline strategy: {baseline_strategy['name']}")
    baseline_norm = _safe_run_backtest(
        ohlcv_data[ref_asset], ref_asset, baseline_strategy['name'], baseline_strategy['default_params']
    )
    if baseline_norm is None:
        logger.error("❌ Baseline backtest failed — stopping pipeline.")
        return

    baseline_result = _extract_standard_metrics(baseline_norm)
    baseline_result['label'] = 'Baseline'

    # Step 4: Generate Proposals (LLM or fallback)
    llm_proposals = []
    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning("⚠️ No GOOGLE_API_KEY found — using deterministic fallback proposals.")
        from src.agent.runner import _fallback_propose_actions
        llm_proposals = _fallback_propose_actions(
            current_regime, features_df, baseline_strategy['default_params']
        )
    else:
        try:
            llm_proposals = propose_actions(current_regime, features_df, pd.DataFrame(baseline_norm))
        except Exception as e:
            logger.error(f"LLM planner failed: {e}")
            llm_proposals = []

    # Step 5: Test Proposals
    all_results = [baseline_result]
    for i, proposal in enumerate(llm_proposals):
        label = f"Proposal_{i+1}_{proposal['strategy_name']}"
        logger.info(f"Testing {label} → Params: {proposal['params']}")
        try:
            result_norm = _safe_run_backtest(
                ohlcv_data[proposal['asset_ticker']],
                proposal['asset_ticker'],
                proposal['strategy_name'],
                proposal['params']
            )
            if result_norm:
                metrics = _extract_standard_metrics(result_norm)
                metrics['label'] = label
                all_results.append(metrics)
            else:
                logger.warning(f"{label} failed to produce results.")
        except Exception as e:
            logger.error(f"{label} error: {e}")

    # Step 6: Policy Selection
    results_df = pd.DataFrame(all_results).set_index('label')
    if results_df.empty:
        logger.error("No valid strategies after backtesting.")
        return

    try:
        best_proposal = select_best_proposal(results_df, config['agent']['risk'])
    except Exception as e:
        logger.error(f"Policy selection failed: {e}")
        best_proposal = None

    # Step 7: Report & Save
    cols = ['Total Return [%]', 'Sharpe Ratio', 'Max Drawdown [%]', 'Num Trades', 'params']
    print("\n📊 Strategy Comparison:\n")
    _print_table(results_df, [c for c in cols if c in results_df.columns])

    if best_proposal is not None:
        print("\n🏆 Selected Best Strategy:")
        try:
            print(best_proposal.to_frame().T.to_markdown(floatfmt=".2f"))
        except Exception:
            print(best_proposal.to_string())

    # Save for dashboard
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(os.path.join(out_dir, f"agent_results_{timestamp}.csv"))
    logger.info(f"✅ Results saved to {out_dir}/agent_results_{timestamp}.csv")

    logger.info("🎯 AgentQuant run completed successfully.")


if __name__ == "__main__":
    main()
