# sjm.py
import os
import math
import time
import joblib
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
from scipy import linalg

TRADING_DAYS_PER_YEAR = 252


def ewma(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rolling_std(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).std(ddof=1).fillna(0.0)


def rsi(s: pd.Series, window: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0.0)
    down = -d.clip(upper=0.0)
    ma_up = up.ewm(alpha=1 / window, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / window, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def downside_std(series: pd.Series, window: int) -> pd.Series:
    def dd(x):
        neg = x[x < 0]
        return np.std(neg, ddof=1) if len(neg) > 1 else 0.0
    return series.rolling(window).apply(lambda x: dd(x), raw=False).fillna(0.0)


def rolling_beta(factor: pd.Series, market: pd.Series, window: int = 63) -> pd.Series:
    cov = factor.rolling(window).cov(market)
    var = market.rolling(window).var()
    return (cov / (var + 1e-12)).fillna(0.0)


def compute_features_from_active_returns(far: pd.DataFrame,
                                         vix: Optional[pd.Series] = None,
                                         t10y2y: Optional[pd.Series] = None) -> pd.DataFrame:
    factors = far.columns.tolist()
    market_proxy = far.mean(axis=1)
    blocks = []
    for f in factors:
        r = far[f].fillna(0.0)
        cols = {
            f + "__ewma8": ewma(r, 8),
            f + "__ewma21": ewma(r, 21),
            f + "__ewma63": ewma(r, 63),
            f + "__std21": rolling_std(r, 21),
            f + "__std63": rolling_std(r, 63),
            f + "__rsi14": rsi(r, 14),
            f + "__macd": macd(r),
            f + "__downstd63": downside_std(r, 63),
            f + "__beta63": rolling_beta(r, market_proxy, 63),
            f + "__active_ret": r
        }
        blocks.append(pd.DataFrame(cols))
    features = pd.concat(blocks, axis=1).sort_index()
    features = features.ffill().bfill().fillna(0.0)
    env = pd.DataFrame(index=features.index)
    if vix is not None:
        v = vix.reindex(features.index).ffill().bfill()
        env['VIX_log'] = np.log(v + 1e-12)
        env['VIX_ewma21'] = ewma(env['VIX_log'], 21)
    if t10y2y is not None:
        slope = t10y2y.reindex(features.index).ffill().bfill()
        env['slope_ewma63'] = ewma(slope, 63)
    if not env.empty:
        features = pd.concat([features, env], axis=1).fillna(0.0)
    return features


def assign_states_dp(X: np.ndarray, centroids: np.ndarray, lambda_penalty: float) -> np.ndarray:
    T, D = X.shape
    K = centroids.shape[0]
    costs = np.zeros((T, K))
    for k in range(K):
        dif = X - centroids[k]
        costs[:, k] = np.sum(dif * dif, axis=1)
    dp = np.zeros((T, K))
    ptr = np.zeros((T, K), dtype=int)
    dp[0] = costs[0]
    ptr[0] = -1
    for t in range(1, T):
        for k in range(K):
            k_other = 1 - k
            same = dp[t-1, k] + costs[t, k]
            switch = dp[t-1, k_other] + costs[t, k] + lambda_penalty
            if same <= switch:
                dp[t, k] = same
                ptr[t, k] = k
            else:
                dp[t, k] = switch
                ptr[t, k] = k_other
    states = np.zeros(T, dtype=int)
    states[-1] = int(np.argmin(dp[-1]))
    for t in range(T - 2, -1, -1):
        states[t] = ptr[t + 1, states[t + 1]]
    return states


def fit_sjm_factor(X: pd.DataFrame,
                   lambda_penalty: float = 50.0,
                   kappa: float = 9.5,
                   max_iter: int = 50,
                   tol: float = 1e-4,
                   verbose: bool = False) -> Dict[str, Any]:
    Xnp = X.values.astype(float)
    T, D = Xnp.shape
    mu = np.nanmean(Xnp, axis=0)
    sigma = np.nanstd(Xnp, axis=0, ddof=1) + 1e-12
    Xs = (Xnp - mu) / sigma
    idx0 = 0
    dists = np.linalg.norm(Xs - Xs[idx0], axis=1)
    idx1 = int(np.argmax(dists))
    centroids = np.vstack([Xs[idx0], Xs[idx1]])
    weights = np.ones(D) / D
    for it in range(max_iter):
        W_sqrt = np.sqrt(np.maximum(weights, 1e-12))[None, :]
        Xw = Xs * W_sqrt
        cent_w = centroids * W_sqrt
        states = assign_states_dp(Xw, cent_w, lambda_penalty=lambda_penalty)
        new_cent = np.zeros_like(centroids)
        for k in (0, 1):
            mask = states == k
            if mask.sum() > 0:
                new_cent[k] = Xs[mask].mean(axis=0)
            else:
                new_cent[k] = centroids[k]
        imp = np.abs(new_cent[0] - new_cent[1])
        if imp.sum() == 0:
            new_weights = np.ones(D) / D
        else:
            imp_norm = imp / (imp.sum() + 1e-12)
            k_active = int(max(1, min(D, math.ceil(kappa))))
            idx_top = np.argsort(imp_norm)[-k_active:]
            w_sparse = np.zeros_like(imp_norm)
            w_sparse[idx_top] = imp_norm[idx_top]
            if w_sparse.sum() > 0:
                new_weights = w_sparse / w_sparse.sum()
            else:
                new_weights = np.ones(D) / D
        cent_change = np.linalg.norm(new_cent - centroids)
        w_change = np.linalg.norm(new_weights - weights)
        centroids = new_cent
        weights = new_weights
        if cent_change < tol and w_change < tol:
            break
    centroids_denorm = centroids * sigma[None, :] + mu[None, :]
    return {
        'centroids': centroids_denorm,
        'weights': weights,
        'states': states,
        'mu': mu,
        'sigma': sigma,
        'feature_names': list(X.columns)
    }


def online_infer(X_window: pd.DataFrame,
                 centroids: np.ndarray,
                 weights: np.ndarray,
                 lambda_penalty: float,
                 filter_len: int = 60) -> int:
    Xf = X_window.values.astype(float)
    W_sqrt = np.sqrt(np.maximum(weights, 1e-12))[None, :]
    Xw = Xf * W_sqrt
    cent_w = centroids * W_sqrt
    st = assign_states_dp(Xw, cent_w, lambda_penalty=lambda_penalty)
    return int(st[-1])


def estimate_ewma_sigma(returns: pd.DataFrame, halflife: int = 126) -> pd.DataFrame:
    lam = 0.5 ** (1 / halflife)
    R = returns.fillna(0.0).values
    T, N = R.shape
    S = np.zeros((N, N))
    for t in range(T):
        rt = R[t][:, None]
        S = lam * S + (1 - lam) * (rt @ rt.T)
    return pd.DataFrame(S, index=returns.columns, columns=returns.columns)


def equilibrium_returns_from_benchmark(Sigma: pd.DataFrame, w_b: np.ndarray, delta: float) -> np.ndarray:
    return delta * (Sigma.values @ w_b)


def bl_posterior(pi: np.ndarray, Sigma: np.ndarray, P: np.ndarray, q: np.ndarray, Omega: np.ndarray, tau: float) -> np.ndarray:
    tauSigma = tau * Sigma
    M = P @ tauSigma @ P.T + Omega
    invM = linalg.inv(M)
    adj = tauSigma @ P.T @ invM @ (q - P @ pi)
    mu = pi + adj
    return mu


def mv_optimal_weights(mu: np.ndarray, Sigma: np.ndarray, delta: float) -> np.ndarray:
    invS = linalg.inv(Sigma)
    return (1.0 / delta) * (invS @ mu)


def tracking_error(w: np.ndarray, w_b: np.ndarray, Sigma: np.ndarray) -> float:
    diff = w - w_b
    te2 = float(diff.T @ Sigma @ diff)
    return float(np.sqrt(max(te2, 0.0)))


def calibrate_omega_to_te(P: np.ndarray, q: np.ndarray, Sigma: np.ndarray, tau: float,
                          pi: np.ndarray, w_b: np.ndarray, delta: float,
                          base_omega_diag: np.ndarray, target_te: float) -> np.ndarray:
    low, high = 1e-6, 1e6
    for _ in range(40):
        mid = math.sqrt(low * high)
        Omega = np.diag(base_omega_diag * mid)
        mu_post = bl_posterior(pi, Sigma, P, q, Omega, tau)
        w_post = mv_optimal_weights(mu_post, Sigma, delta)
        te = tracking_error(w_post, w_b, Sigma)
        if np.isnan(te):
            high = mid
            continue
        if te > target_te:
            low = mid
        else:
            high = mid
    alpha = high
    return np.diag(base_omega_diag * alpha)


def simulate_single_factor_strategy(far: pd.Series, regimes: pd.Series, cost_bps: float = 0.0005) -> Dict[str, float]:
    idx = far.index.intersection(regimes.index)
    r = far.loc[idx]
    s = regimes.loc[idx].fillna(method='ffill').fillna(0).astype(int)
    pos = s.replace({0: -1, 1: 1}).astype(float)
    pos_shift = pos.shift(1).fillna(pos.iloc[0])
    turnover = (pos - pos_shift).abs()
    costs = turnover * cost_bps
    strat_ret = pos * r - costs
    mean = strat_ret.mean() * TRADING_DAYS_PER_YEAR
    std = strat_ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = mean / (std + 1e-12)
    return {'sharpe': float(sharpe), 'mean_ann': float(mean), 'std_ann': float(std)}


def tune_hyperparams(far_df: pd.DataFrame,
                     features_df: pd.DataFrame,
                     factor_name: str,
                     lambda_grid: List[float],
                     kappa_grid: List[float],
                     lookback_train: int = 252 * 8,
                     validation_periods: int = 252 * 3,
                     refit_freq: int = 21,
                     cost_bps: float = 0.0005,
                     verbose: bool = False) -> Tuple[float, float]:
    dates = features_df.index
    if len(dates) < lookback_train + validation_periods:
        raise ValueError("insufficient data for tuning")

    total_combos = len(lambda_grid) * len(kappa_grid)
    combo_count = 0
    best = {'sharpe': -np.inf, 'lambda': None, 'kappa': None}
    cols_all = features_df.columns.tolist()

    for lam in lambda_grid:
        for kap in kappa_grid:
            combo_count += 1
            start_msg = f"[Tune {factor_name}] ({combo_count}/{total_combos}) λ={lam}, κ={kap}"
            print(start_msg, end="", flush=True)

            start_val = lookback_train
            end_val = lookback_train + validation_periods
            regimes = pd.Series(index=dates[start_val:end_val], dtype=float)
            i = start_val
            while i < end_val:
                window_start = max(0, i - lookback_train)
                X_train = features_df.iloc[window_start:i]
                cols = [c for c in cols_all if c.startswith(factor_name + "__") or ("VIX" in c or "slope" in c)]
                X_train_fac = X_train[cols]
                fit = fit_sjm_factor(X_train_fac, lambda_penalty=lam, kappa=kap, max_iter=20)
                for j in range(i, min(i + refit_freq, end_val)):
                    idx_start = max(0, j - 60 + 1)
                    Xf = features_df.iloc[idx_start:j+1][cols]
                    st = online_infer(Xf, fit['centroids'], fit['weights'], lambda_penalty=lam)
                    regimes.iloc[j - start_val] = st
                i += refit_freq

            far = far_df[factor_name]
            regimes_full = pd.Series(index=dates, dtype=float)
            regimes_full.loc[dates[start_val:end_val]] = regimes.values
            regimes_full = regimes_full.ffill().fillna(0).astype(int)
            sim = simulate_single_factor_strategy(far, regimes_full, cost_bps=cost_bps)
            sharpe = sim['sharpe']

            end_msg = f" → Sharpe={sharpe:.3f}"
            print(end_msg)

            if sharpe > best['sharpe']:
                best = {'sharpe': sharpe, 'lambda': lam, 'kappa': kap}

    print(f"[Tune {factor_name}] best λ={best['lambda']} κ={best['kappa']} Sharpe={best['sharpe']:.3f}")
    return best['lambda'], best['kappa']


def prepare_bl_inputs(far_df: pd.DataFrame,
                      last_regimes: Dict[str, int],
                      tau: float = 0.025,
                      delta: float = 2.5,
                      halflife_sigma: int = 126,
                      target_te: float = 0.02) -> Dict[str, Any]:
    Sigma = estimate_ewma_sigma(far_df, halflife=halflife_sigma)
    factors = far_df.columns.tolist()
    K = len(factors)
    w_b = np.ones(K) / K
    pi = equilibrium_returns_from_benchmark(Sigma, w_b, delta)
    P = np.eye(K)
    q_list = []
    for f in factors:
        r = far_df[f]
        last_r = last_regimes.get(f, 1)
        mask = r > 0 if last_r == 1 else r <= 0
        qm = r.loc[mask].mean() if mask.sum() > 0 else r.mean()
        q_list.append(qm)
    q_vec = np.array(q_list)
    var = far_df.var(axis=0).values
    base_omega_diag = var * (1.0 / (tau + 1e-12))
    Omega = calibrate_omega_to_te(P, q_vec, Sigma.values, tau, pi, w_b, delta, base_omega_diag, target_te)
    return {
        'P': pd.DataFrame(P, index=factors, columns=factors),
        'q': pd.Series(q_vec, index=factors),
        'Omega': pd.DataFrame(Omega, index=factors, columns=factors),
        'Sigma': Sigma,
        'tau': tau,
        'delta': delta,
        'benchmark_weights': pd.Series(w_b, index=factors)
    }


def run_sjm_pipeline(far_path: str,
                     vix_path: Optional[str] = None,
                     t10y2y_path: Optional[str] = None,
                     lookback_days: Optional[int] = 252 * 8,
                     lambda_penalty: float = 50.0,
                     kappa: float = 9.5,
                     tuned_params: Optional[Dict[str, Tuple[float, float]]] = None,
                     tune: bool = False,
                     lambda_grid: Optional[List[float]] = None,
                     kappa_grid: Optional[List[float]] = None,
                     halflife_sigma: int = 126,
                     tau: float = 0.025,
                     delta: float = 2.5,
                     target_te: float = 0.02,
                     save_artifacts: bool = True,
                     artifacts_dir: str = "artifacts/sjm",
                     verbose: bool = False) -> Dict[str, Any]:
    print(f"[SJM] starting pipeline | lookback_days={lookback_days} | lambda={lambda_penalty} | kappa={kappa}")
    far = pd.read_csv(far_path, index_col=0, parse_dates=True)
    vix = pd.read_csv(vix_path, index_col=0, parse_dates=True).iloc[:, 0] if (vix_path and os.path.exists(vix_path)) else None
    t10y2y = pd.read_csv(t10y2y_path, index_col=0, parse_dates=True).iloc[:, 0] if (t10y2y_path and os.path.exists(t10y2y_path)) else None
    features = compute_features_from_active_returns(far, vix=vix, t10y2y=t10y2y)
    fits: Dict[str, Any] = {}
    last_regimes: Dict[str, int] = {}
    factors = far.columns.tolist()
    os.makedirs(artifacts_dir, exist_ok=True)

    for i, f in enumerate(factors, 1):
        t0 = time.time()
        print(f"[SJM] ({i}/{len(factors)}) processing factor: {f}")
        cols = [c for c in features.columns if c.startswith(f + "__")] + [c for c in features.columns if ("VIX" in c or "slope" in c)]
        Xf = features[cols]

        if tuned_params is not None and f in tuned_params:
            lam, kap = tuned_params[f]
        else:
            lam, kap = lambda_penalty, kappa

        if tune and lambda_grid is not None and kappa_grid is not None:
            try:
                lookback_train_val = lookback_days if lookback_days is not None else 252 * 8
                lam, kap = tune_hyperparams(far, features, f, lambda_grid, kappa_grid, lookback_train=lookback_train_val, validation_periods=252*3, verbose=verbose)
                if verbose:
                    print(f"[SJM] tuning result for {f} -> lambda={lam} kappa={kap}")
            except Exception as e:
                if verbose:
                    print(f"[SJM] tuning failed for {f}: {e}")

        last_idx = len(Xf) - 1
        min_window = 252 * 8
        max_window = 252 * 12

        if lookback_days is None:
            available = last_idx + 1
            use_days = min(max(available, min_window), max_window)
            use_days = min(use_days, available)
            window_start = max(0, last_idx + 1 - use_days)
        else:
            window_start = max(0, last_idx - lookback_days + 1)

        X_train = Xf.iloc[window_start:last_idx + 1]
        print(f"[SJM] {f}: fitting with {len(X_train)} samples")
        fit = fit_sjm_factor(X_train, lambda_penalty=lam, kappa=kap, max_iter=50, verbose=verbose)
        fits[f] = fit
        if save_artifacts:
            fname = f"{f}_{pd.Timestamp.now().strftime('%Y%m%d')}.joblib"
            joblib.dump(fit, os.path.join(artifacts_dir, fname))
        filter_len = min(60, len(X_train))
        Xf_filter = Xf.iloc[max(0, last_idx - filter_len + 1):last_idx + 1]
        state_last = online_infer(Xf_filter, fit['centroids'], fit['weights'], lambda_penalty=lam)
        last_regimes[f] = int(state_last)
        t1 = time.time()
        print(f"[SJM] {f}: current regime = {state_last} | fit time = {(t1 - t0):.2f}s")

    print("[SJM] preparing Black-Litterman inputs")
    bl_inputs = prepare_bl_inputs(far, last_regimes, tau=tau, delta=delta, halflife_sigma=halflife_sigma, target_te=target_te)
    print("[SJM] pipeline done")
    return {'features': features, 'fits': fits, 'last_regimes': last_regimes, 'bl_inputs': bl_inputs}


if __name__ == "__main__":
    base = os.path.join(os.getcwd(), "data_factors")
    farp = os.path.join(base, "factor_active_returns.csv")
    vixp = os.path.join(base, "VIX.csv")
    t10p = os.path.join(base, "T10Y2Y.csv")
    if not os.path.exists(farp):
        print("factor_active_returns.csv not found.")
    else:
        out = run_sjm_pipeline(farp, vix_path=vixp if os.path.exists(vixp) else None,
                               t10y2y_path=t10p if os.path.exists(t10p) else None,
                               lookback_days=252 * 8,
                               lambda_penalty=50.0, kappa=9.5,
                               tune=False, verbose=True)
        print("last_regimes:", out['last_regimes'])
        print("BL q head:")
        print(out['bl_inputs']['q'].head())
