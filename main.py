# main.py
import os
import joblib
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sjm import run_sjm_pipeline

import matplotlib.dates as mdates

# --- plotting utility ---
def plot_factor_with_regimes(factor_name: str,
                             far_df: pd.DataFrame,
                             states_series: pd.Series,
                             window: tuple = None,
                             save_path: str = None,
                             figsize=(12, 4)):
    """
    Plota retornos do fator com regimes (0/1) preenchidos no fundo:
    - far_df: DataFrame com retornos (index datetime)
    - states_series: pd.Series (index datetime, values 0/1)
    - window: (start, end) optional to zoom
    - save_path: if provided, saves the plot
    """
    # usar retorno acumulado para visualização
    returns = (1.0 + far_df[factor_name].fillna(0.0)).cumprod() - 1.0
    returns = returns.dropna()

    # Align states to returns index (causal reindex via ffill)
    try:
        states = states_series.reindex(returns.index, method='ffill').fillna(method='ffill').fillna(0).astype(int)
    except Exception:
        # fallback: create series from values if index mismatch
        states = pd.Series(states_series.values, index=returns.index[:len(states_series)]).reindex(returns.index, method='ffill').fillna(0).astype(int)

    if window is not None:
        start, end = window
        returns = returns.loc[start:end]
        states = states.loc[start:end]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(returns.index, returns.values, lw=0.9, label=f"{factor_name} cumulative returns")

    # Fill background by contiguous regime blocks
    current_state = None
    start_idx = None
    idxs = states.index
    vals = states.values
    for i, dt in enumerate(idxs):
        st = vals[i]
        if current_state is None:
            current_state = st
            start_idx = dt
            continue
        if st != current_state:
            end_idx = idxs[i - 1]
            color = 'green' if current_state == 1 else 'red'
            ax.axvspan(start_idx, end_idx, color=color, alpha=0.12, linewidth=0)
            current_state = st
            start_idx = dt
    # final block
    if start_idx is not None:
        end_idx = idxs[-1]
        color = 'green' if current_state == 1 else 'red'
        ax.axvspan(start_idx, end_idx, color=color, alpha=0.12, linewidth=0)

    ax.set_title(f"{factor_name} cumulative returns with regimes")
    ax.set_ylabel("Cumulative return")
    ax.set_xlabel("Date")
    ax.grid(True)
    ax.legend()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
# --- end plotting utility ---


DATA_FACTORS = "data_factors"
FAR_PATH = os.path.join(DATA_FACTORS, "factor_active_returns.csv")
FACTOR_RETURNS_PATH = os.path.join(DATA_FACTORS, "factor_returns_daily.csv")
SP500_PATH = os.path.join("data", "SP500.csv")

LOOKBACK_EXPANDING = True
LOOKBACK_DAYS = None
LAMBDA = 50.0   # change manually as you like
KAPPA = 9.5     # change manually as you like
TAU = 0.025
DELTA = 2.5
TARGET_TE = 0.02
SAVE_ARTIFACTS_DIR = "artifacts/sjm"

REBALANCE_FREQ = "M"
EXECUTION_LAG_DAYS = 1

# outputs saved under results/
RESULTS_DIR = "results"
PLOT_OUTPUT = os.path.join(RESULTS_DIR, "results_nav.png")
TABLE_OUTPUT = os.path.join(RESULTS_DIR, "results_table.csv")

# train / test split for your dataset (you requested validation to start in 2018)
TRAIN_START = "2005-01-01"
TRAIN_END = "2017-12-31"
TEST_START = "2018-01-01"


def load_returns():
    fr = pd.read_csv(FACTOR_RETURNS_PATH, index_col=0, parse_dates=True)
    fr.index = pd.to_datetime(fr.index)
    fr = fr.sort_index()

    sp = None
    if os.path.exists(SP500_PATH):
        sp_raw = pd.read_csv(SP500_PATH, index_col=0, parse_dates=True).iloc[:, 0]
        sp_raw.index = pd.to_datetime(sp_raw.index)
        sp_raw = sp_raw.sort_index()
        sp_numeric = pd.to_numeric(sp_raw.astype(str).str.replace(',', '').str.strip(), errors='coerce')
        valid = sp_numeric.dropna()
        if len(valid) == 0:
            raise RuntimeError("SP500 file exists but contains no numeric values after coercion.")
        med = valid.abs().median()
        if med > 1.5:
            sp = sp_numeric.pct_change()
        else:
            sp = sp_numeric.copy()
        sp = sp.reindex(fr.index)
        sp = sp.ffill().bfill().fillna(0.0)
        sp.name = "SP500"
    else:
        sp = fr.mean(axis=1)
        sp.name = "SP500"

    fr = fr.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(',', '').str.strip(), errors='coerce'))
    fr.index = pd.to_datetime(fr.index)
    fr = fr.reindex(sp.index)
    fr = fr.fillna(0.0)

    df_all = fr.copy()
    df_all["SP500"] = sp

    return df_all


def get_month_end_rebalances(dates_index):
    months = pd.period_range(start=dates_index.min(), end=dates_index.max(), freq='M')
    month_ends = [m.end_time.normalize() for m in months]
    trading_days = dates_index
    adjusted = []
    for me in month_ends:
        valid = trading_days[trading_days <= me]
        if len(valid) > 0:
            adjusted.append(valid[-1])
    adjusted = sorted(list(dict.fromkeys(adjusted)))
    return adjusted


def compute_metrics(returns_series):
    cum_ret = (1.0 + returns_series).cumprod().iloc[-1] - 1.0
    ann_ret = returns_series.mean() * 252
    ann_vol = returns_series.std(ddof=1) * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-12)
    return {"cumulative_return": float(cum_ret), "ann_return": float(ann_ret), "ann_vol": float(ann_vol), "sharpe": float(sharpe)}


def run_backtest():
    start_time = datetime.now()
    print(f"[Backtest] Starting backtest at {start_time.isoformat()}")

    # ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)

    df_returns = load_returns()
    dates = df_returns.index
    factors = [c for c in df_returns.columns if c != "SP500"]
    benchmark_assets = factors + ["SP500"]

    rebalance_dates = get_month_end_rebalances(dates)
    print(f"[Backtest] Found {len(rebalance_dates)} rebalance dates (first={rebalance_dates[0].date()}, last={rebalance_dates[-1].date()})")
    if len(rebalance_dates) < 2:
        raise RuntimeError("Not enough rebalance dates found.")

    # We removed PRETUNE entirely — tuned_params is always None and parameters are set manually via LAMBDA/KAPPA above
    tuned_params = None

    rebalance_dates = [d for d in rebalance_dates if d >= pd.to_datetime(TEST_START)]
    if len(rebalance_dates) == 0:
        raise RuntimeError(f"No rebalance dates found on or after TEST_START={TEST_START}")

    weights_by_date = {}
    last_weights = pd.Series(np.ones(len(benchmark_assets)) / len(benchmark_assets), index=benchmark_assets)

    out = None
    for idx, rb in enumerate(rebalance_dates, 1):
        print(f"[Backtest] ({idx}/{len(rebalance_dates)}) Rebalance date: {rb.date()}")
        lookback = None if LOOKBACK_EXPANDING else LOOKBACK_DAYS
        t0 = datetime.now()
        out = run_sjm_pipeline(
            far_path=FAR_PATH,
            vix_path=None,
            t10y2y_path=None,
            lookback_days=lookback,
            lambda_penalty=LAMBDA,
            kappa=KAPPA,
            tuned_params=tuned_params,
            tune=False,
            halflife_sigma=126,
            tau=TAU,
            delta=DELTA,
            target_te=TARGET_TE,
            save_artifacts=True,
            artifacts_dir=SAVE_ARTIFACTS_DIR,
            verbose=True,
            as_of_date=rb  # ensure no lookahead
        )
        t1 = datetime.now()
        print(f"[Backtest] SJM pipeline done in {(t1 - t0).total_seconds():.1f}s")

        # --- plotting PER-REBALANCE removed to avoid many files ---
        # --- end removed plotting block ---

        bl_inputs = out["bl_inputs"]
        from black_litterman import bl_pipeline
        bl_out = bl_pipeline(bl_inputs, target_te=TARGET_TE, long_only=True)
        w = bl_out["w"]

        if "SP500" not in w.index:
            w_full = w.reindex(factors).fillna(0.0)
            w_full["SP500"] = 0.0
        else:
            w_full = w.reindex(benchmark_assets).fillna(0.0)

        if w_full.sum() == 0:
            w_full = pd.Series(np.ones(len(w_full)) / len(w_full), index=w_full.index)
        else:
            w_full = w_full / w_full.sum()

        pos = dates.get_indexer_for([rb])[0]
        eff_pos = min(len(dates) - 1, pos + EXECUTION_LAG_DAYS)
        eff_date = dates[eff_pos]
        weights_by_date[eff_date] = w_full
        last_weights = w_full
        print(f"[Backtest] Weights scheduled for {eff_date.date()} (sum={w_full.sum():.4f})")

    print("[Backtest] Building daily weight matrix...")
    df_weights = pd.DataFrame(index=dates, columns=benchmark_assets, dtype=float)
    current = pd.Series(np.ones(len(benchmark_assets)) / len(benchmark_assets), index=benchmark_assets)
    for d in dates:
        if d in weights_by_date:
            current = weights_by_date[d]
        df_weights.loc[d] = current
    df_weights = df_weights.ffill().fillna(0.0)
    print("[Backtest] Daily weights ready")

    strat_daily = (df_weights * df_returns.reindex(df_weights.index)[benchmark_assets]).sum(axis=1)
    bmk_w = pd.Series(np.ones(len(benchmark_assets)) / len(benchmark_assets), index=benchmark_assets)
    benchmark_daily = (df_returns[benchmark_assets] * bmk_w).sum(axis=1)

    # --- slice returns to out-of-sample period first (TEST_START) ---
    strat_daily = strat_daily.loc[strat_daily.index >= pd.to_datetime(TEST_START)]
    benchmark_daily = benchmark_daily.loc[benchmark_daily.index >= pd.to_datetime(TEST_START)]

    # --- NAVs (cumulative) built from the sliced returns ---
    nav_strat = (1.0 + strat_daily).cumprod()
    nav_bmk = (1.0 + benchmark_daily).cumprod()

    # --- ensure NAV starts at 1.0 at TEST_START (defensive) ---
    if not nav_strat.empty and nav_strat.iloc[0] != 0:
        nav_strat = nav_strat / float(nav_strat.iloc[0])
    if not nav_bmk.empty and nav_bmk.iloc[0] != 0:
        nav_bmk = nav_bmk / float(nav_bmk.iloc[0])


    # métricas (somente out-of-sample)
    metrics_strat = compute_metrics(strat_daily)
    metrics_bmk = compute_metrics(benchmark_daily)

    # tabela de resultados
    results_table = pd.DataFrame({
        "strategy": metrics_strat,
        "benchmark": metrics_bmk
    })
    results_table = results_table.T

    # gráfico NAV (somente 2018+)
    plt.figure(figsize=(10, 6))
    plt.plot(nav_strat.index, nav_strat.values, label="Strategy NAV")
    plt.plot(nav_bmk.index, nav_bmk.values, label="Benchmark NAV (EW)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Value")
    plt.title("Strategy vs Benchmark NAV (Out-of-Sample)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT, dpi=150)
    print(f"[Backtest] Saved NAV plot: {PLOT_OUTPUT}")

    # salvar tabela
    results_table.to_csv(TABLE_OUTPUT)
    print(f"[Backtest] Saved results table: {TABLE_OUTPUT}")

    # --- generate one full-period plot per factor using the last SJM fits (one image per factor) ---
    try:
        far_full = pd.read_csv(FAR_PATH, index_col=0, parse_dates=True)
        far_full.index = pd.to_datetime(far_full.index)
        fits = out.get("fits", {}) if out is not None else {}
        os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)
        for factor_name, fit in fits.items():
            states = fit.get("states", None)
            if states is None:
                # skip if no states returned for this factor
                continue
            # if states is not a pd.Series, try to align it to the tail of far_full index
            if not isinstance(states, pd.Series):
                try:
                    states = pd.Series(states, index=far_full.index[-len(states):])
                except Exception:
                    continue
            savep = os.path.join(RESULTS_DIR, "plots", f"{factor_name}_regimes_fullperiod.png")
            plot_factor_with_regimes(factor_name, far_full, states, save_path=savep)
        print(f"[Backtest] Saved full-period factor regime plots to: {os.path.join(RESULTS_DIR, 'plots')}")
    except Exception as _e:
        print(f"[Backtest] full-period plotting skipped/failed: {_e}")
    # --- end full-period plotting ---

    end_time = datetime.now()
    print(f"[Backtest] Finished at {end_time.isoformat()} (elapsed {(end_time - start_time).total_seconds():.1f}s)")

    return {
        "dates": dates,
        "nav_strategy": nav_strat,
        "nav_benchmark": nav_bmk,
        "daily_strategy": strat_daily,
        "daily_benchmark": benchmark_daily,
        "results_table": results_table
    }



if __name__ == "__main__":
    out = run_backtest()
    print("Summary:")
    print(out["results_table"])
