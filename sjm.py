# sjm.py
import os
import math
import time
import joblib
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
from scipy import linalg
import traceback

TRADING_DAYS_PER_YEAR = 252

def safe_inv(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Stable matrix inversion: use regularized inverse or pseudo-inverse if ill-conditioned."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    try:
        cond = np.linalg.cond(A)
    except Exception:
        cond = float('inf')
    if cond < 1e12:
        try:
            return linalg.inv(A + eps * np.eye(n))
        except Exception:
            return np.linalg.pinv(A)
    else:
        return np.linalg.pinv(A)


def ewma(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rolling_std(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=1).std(ddof=1)


def rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = -1.0 * delta.clip(upper=0.0)
    ma_up = up.rolling(period, min_periods=1).mean()
    ma_down = down.rolling(period, min_periods=1).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - 100 / (1 + rs)


def macd(s: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    return ewma(s, fast) - ewma(s, slow)


def downside_std(s: pd.Series, window: int) -> pd.Series:
    roll = s.rolling(window, min_periods=1)
    down = roll.apply(lambda x: np.std(x[x < 0]) if np.any(x < 0) else 0.0, raw=False)
    return down


def rolling_beta(s: pd.Series, benchmark: pd.Series, window: int) -> pd.Series:
    # alinhar índices e converter para numeric
    s_al = s.reindex(benchmark.index).astype(float)
    b_al = benchmark.astype(float).reindex(s_al.index)

    # covariância rolling (cov returns a Series when passing another Series)
    cov = s_al.rolling(window=window, min_periods=1).cov(b_al)
    var_b = b_al.rolling(window=window, min_periods=1).var(ddof=1)

    # evitar divisão por zero / NaNs
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = cov / (var_b + 1e-12)

    # onde var_b é 0 ou NaN, colocar 0.0
    beta = beta.fillna(0.0)
    beta[~np.isfinite(beta)] = 0.0
    return beta

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
    # only forward-fill (causal); do NOT back-fill because that would use future information
    features = features.ffill().fillna(0.0)
    env = pd.DataFrame(index=features.index)
    if vix is not None:
        # align VIX to features index and forward-fill only; replace remaining missing with historic mean
        v = vix.reindex(features.index).ffill()
        if v.isna().any():
            v = v.fillna(v.mean())
        env['VIX_log'] = np.log(v + 1e-12)
        env['VIX_ewma21'] = ewma(env['VIX_log'], 21)
    if t10y2y is not None:
        s = t10y2y.reindex(features.index).ffill()
        if s.isna().any():
            s = s.fillna(s.mean())
        env['slope'] = s
        env['slope_ewma21'] = ewma(env['slope'], 21)
    # merge features and env
    features = pd.concat([features, env], axis=1)
    return features


def assign_states_dp(X: np.ndarray, centroids: np.ndarray, lambda_penalty: float = 50.0):
    T, D = X.shape
    K = centroids.shape[0]
    dp = np.zeros((T, K)) + 1e12
    ptr = np.zeros((T, K), dtype=int)
    # cost to start at centroid 0 at t=0.. etc
    for k in range(K):
        dp[0, k] = np.linalg.norm(X[0] - centroids[k])
    for t in range(1, T):
        for k in range(K):
            costs = dp[t - 1] + np.linalg.norm(X[t] - centroids[k])
            # add switching penalty
            costs += lambda_penalty * (1.0)  # simple penalization; kept generic
            dp[t, k] = np.min(costs)
            ptr[t, k] = int(np.argmin(costs))
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
        # re-estimate centroids
        for k in range(centroids.shape[0]):
            mask = states == k
            if mask.sum() == 0:
                continue
            centroids[k] = np.mean(Xs[mask], axis=0)
        # update weights (sparsity encouragement)
        var_k = np.var(centroids, axis=0) + 1e-12
        weights = np.exp(-kappa * var_k)
        ssum = weights.sum()
        if ssum == 0 or not np.isfinite(ssum):
            # fallback to uniform if numerical issues arise
            weights = np.ones_like(weights) / float(len(weights))
        else:
            weights = weights / ssum
    return {'centroids': centroids, 'weights': weights, 'states': states}


def online_infer(Xf: pd.DataFrame, centroids: np.ndarray, weights: np.ndarray, lambda_penalty: float = 50.0):
    Xnp = Xf.values.astype(float)
    T, D = Xnp.shape
    mu = np.nanmean(Xnp, axis=0)
    sigma = np.nanstd(Xnp, axis=0, ddof=1) + 1e-12
    Xs = (Xnp - mu) / sigma
    W_sqrt = np.sqrt(np.maximum(weights, 1e-12))[None, :]
    Xw = Xs * W_sqrt
    cent_w = centroids * W_sqrt
    states = assign_states_dp(Xw, cent_w, lambda_penalty=lambda_penalty)
    return states[-1]


def estimate_ewma_sigma(returns: pd.DataFrame, halflife: int = 126) -> pd.DataFrame:
    lam = 0.5 ** (1 / halflife)
    # drop rows that are entirely NaN (e.g., days with no returns) to avoid introducing artificial zeros
    returns_clean = returns.dropna(how='all')
    # for any remaining NaNs, fill with 0.0 (minimal imputation); prefer causal imputations upstream
    R = returns_clean.fillna(0.0).values
    T, N = R.shape
    S = np.zeros((N, N))
    for t in range(T):
        rt = R[t][:, None]
        S = lam * S + (1 - lam) * (rt @ rt.T)
    return pd.DataFrame(S, index=returns_clean.columns, columns=returns_clean.columns)


def equilibrium_returns_from_be(Sigma: np.ndarray, market_caps: np.ndarray, risk_aversion: float = 2.5) -> np.ndarray:
    # simple CAPM equilibrium returns proxy: pi = delta * Sigma * w_mkt
    return risk_aversion * (Sigma @ market_caps)


def bl_posterior(pi: np.ndarray, Sigma: np.ndarray, P: np.ndarray, q: np.ndarray, Omega: np.ndarray, tau: float) -> np.ndarray:
    tauSigma = tau * Sigma
    M = P @ tauSigma @ P.T + Omega
    invM = safe_inv(M)
    adj = tauSigma @ P.T @ invM @ (q - P @ pi)
    mu = pi + adj
    return mu


def mv_optimal_weights(mu: np.ndarray, Sigma: np.ndarray, delta: float) -> np.ndarray:
    invS = safe_inv(Sigma)
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
    regimes = regimes.loc[idx]
    pos = regimes.replace({0: -1, 1: 1}).shift(1).fillna(0.0)  # naive sign position
    gross = pos * r
    net = gross - cost_bps * pos.diff().abs().fillna(0.0)
    ret = net
    # safe stats: handle empty or extremely short series
    ret = np.asarray(ret)  # ensure numpy array
    ret = ret[~np.isnan(ret)]  # drop nans

    if ret.size == 0:
        ann_ret = 0.0
        ann_vol = 0.0
        sharpe = 0.0
    else:
        ann_ret = float(np.nanmean(ret)) * TRADING_DAYS_PER_YEAR
        # if only 1 sample, std with ddof=1 is invalid -> use ddof=0 fallback
        if ret.size <= 1:
            ann_vol = float(np.nanstd(ret, ddof=0)) * np.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            ann_vol = float(np.nanstd(ret, ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe = ann_ret / (ann_vol + 1e-12)

    return {'ann_return': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe}


def tune_hyperparams(far: pd.DataFrame, features: pd.DataFrame, factor: str,
                     la_grid: List[float], ka_grid: List[float],
                     lookback_train: int = 252 * 6, validation_days: int = 252, verbose: bool = False) -> Tuple[float, float]:
    # very small tuning: choose (lambda, kappa) by simple rolling validation (kept minimal)
    best = {'sharpe': -np.inf, 'lam': la_grid[0], 'kap': ka_grid[0]}
    for lam in la_grid:
        for kap in ka_grid:
            try:
                # split by available history
                n = len(features)
                if n < 2:
                    continue
                train_end = max(0, n - validation_days - 1)
                train_start = max(0, train_end - lookback_train)
                X_train = features.iloc[train_start:train_end + 1]
                # be conservative: simulate on the following validation window only
                val_start = train_end + 1
                val_end = min(n - 1, train_end + validation_days)
                X_val = features.iloc[val_start:val_end + 1]
                if X_train.shape[0] < 10 or X_val.shape[0] < 5:
                    continue
                fit = fit_sjm_factor(X_train, lambda_penalty=lam, kappa=kap, max_iter=30, tol=1e-3)
                # infer on validation window using last few points for filtering
                state = online_infer(X_val, fit['centroids'], fit['weights'], lambda_penalty=lam)
                # simulate a naive single-factor strategy as proxy:
                # get returns series for the factor from far aligned by features index
                f_ret = far.reindex(X_val.index)
                regimes = pd.Series([state] * len(f_ret), index=f_ret.index)
                sim = simulate_single_factor_strategy(f_ret, regimes)
                if sim['sharpe'] > best['sharpe']:
                    best = {'sharpe': sim['sharpe'], 'lam': lam, 'kap': kap}
            except Exception as e:
                # always print traceback for easier debugging (do not swallow silently)
                traceback.print_exc()
                if verbose:
                    print(f"[SJM] tuning failed for {factor}: {e}")
    # ensure returned values are numeric
    lam_out = float(best.get('lam', la_grid[0]))
    kap_out = float(best.get('kap', ka_grid[0]))
    return lam_out, kap_out


def tracking_positions_to_weights(positions: Dict[str, float]) -> np.ndarray:
    keys = sorted(positions.keys())
    w = np.array([positions[k] for k in keys], dtype=float)
    s = np.nansum(np.abs(w))
    if s == 0:
        return w
    return w / s


def mv_optimal_weights_constrained(mu: np.ndarray, Sigma: np.ndarray, delta: float,
                                   bounds: List[Tuple[float, float]] = None):
    # simple unconstrained then clip
    w = mv_optimal_weights(mu, Sigma, delta)
    if bounds is None:
        return w
    w = np.maximum(w, np.array([b[0] for b in bounds]))
    w = np.minimum(w, np.array([b[1] for b in bounds]))
    # re-normalize
    s = np.nansum(np.abs(w))
    if s == 0:
        return w
    return w / s


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
                     verbose: bool = False,
                     as_of_date: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
    print(f"[SJM] starting pipeline | lookback_days={lookback_days} | lambda={lambda_penalty} | kappa={kappa} | as_of_date={as_of_date}")
    far = pd.read_csv(far_path, index_col=0, parse_dates=True)
    vix = pd.read_csv(vix_path, index_col=0, parse_dates=True).iloc[:, 0] if (vix_path and os.path.exists(vix_path)) else None
    t10y2y = pd.read_csv(t10y2y_path, index_col=0, parse_dates=True).iloc[:, 0] if (t10y2y_path and os.path.exists(t10y2y_path)) else None

    # If as_of_date provided, restrict all input series to data <= as_of_date (prevent lookahead)
    if as_of_date is not None:
        asof = pd.to_datetime(as_of_date)
        far = far.loc[:asof].copy()
        if vix is not None:
            vix = vix.loc[:asof].copy()
        if t10y2y is not None:
            t10y2y = t10y2y.loc[:asof].copy()

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
                lam, kap = tune_hyperparams(far, features, f, lambda_grid, kappa_grid, lookback_train=lookback_train_val, validation_days=252, verbose=verbose)
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

        # ensure states is a pandas Series indexed by the training index when possible
        if 'states' in fit and not isinstance(fit['states'], pd.Series):
            try:
                fit_states = pd.Series(fit['states'], index=X_train.index)
            except Exception:
                # fallback: keep as plain Series with integer index if shape mismatch
                fit_states = pd.Series(fit['states'])
            fit['states'] = fit_states

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

    # prepare BL inputs (q, P, base omega, etc.)
    # NOTE: far here is already sliced to as_of_date if provided above
    Sigma = estimate_ewma_sigma(far, halflife=halflife_sigma)
    factors_sorted = far.columns.tolist()
    P = np.eye(len(factors_sorted))
    # compute q as mean active returns by regime using only available data (far is already sliced if as_of_date provided)
    q = pd.Series(index=factors_sorted, dtype=float)
    base_omega = np.ones(len(factors_sorted)) * 1e-4
    # Example: for each factor compute mean active return over the entire available (up to as_of_date) period
    for idx, fac in enumerate(factors_sorted):
        if fac in last_regimes:
            # use all available far data up to as_of_date for that factor
            vals = far[fac].dropna()
            q.iloc[idx] = vals.mean() if len(vals) > 0 else 0.0
        else:
            q.iloc[idx] = 0.0

    # ensure benchmark_weights (equally-weighted across assets) — minimal change
    asset_index = list(Sigma.index) if hasattr(Sigma, "index") else ["Market", "Value", "Size", "Momentum", "Quality", "LowVolatility", "Growth"]
    w_b = pd.Series([1.0 / len(asset_index)] * len(asset_index), index=asset_index)

    bl_inputs = {
        'q': q,
        'P': P,
        'Sigma': Sigma,
        'base_omega': base_omega,
        'tau': tau,
        'benchmark_weights': w_b
    }
    return {'fits': fits, 'last_regimes': last_regimes, 'bl_inputs': bl_inputs}
