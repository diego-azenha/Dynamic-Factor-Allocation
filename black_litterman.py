# black_litterman.py (compact)
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scipy import linalg
from scipy.optimize import minimize

def safe_inv(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    try:
        cond = np.linalg.cond(A)
    except Exception:
        return np.linalg.pinv(A)
    if not np.isfinite(cond) or cond > 1e12:
        return np.linalg.pinv(A)
    try:
        return linalg.inv(A + eps * np.eye(A.shape[0]))
    except Exception:
        return np.linalg.pinv(A)

def compute_bl_posterior(bl_inputs: Dict[str, Any]) -> Dict[str, Any]:
    # required fields
    P_in = bl_inputs.get("P")
    q_in = bl_inputs.get("q")
    Sigma_in = bl_inputs.get("Sigma")
    if P_in is None or q_in is None or Sigma_in is None:
        raise KeyError("bl_inputs must contain at least 'P', 'q' and 'Sigma'.")

    # Omega handling: prefer explicit 'Omega', fallback to diagonal from 'base_omega'
    Omega_in = bl_inputs.get("Omega", None)
    base_omega = bl_inputs.get("base_omega", None)
    if Omega_in is None and base_omega is None:
        # default small diagonal to avoid singularities
        if isinstance(Sigma_in, pd.DataFrame):
            k = P_in.shape[0] if isinstance(P_in, pd.DataFrame) else np.asarray(P_in).shape[0]
        else:
            k = np.asarray(P_in).shape[0]
        Omega = np.eye(k) * 1e-6
        print("[BL] Warning: neither 'Omega' nor 'base_omega' provided — using tiny diagonal Omega (1e-6).")
    elif Omega_in is None and base_omega is not None:
        # base_omega may be array-like or Series
        bo = np.asarray(base_omega, dtype=float)
        if bo.ndim == 1:
            Omega = np.diag(bo)
        else:
            Omega = np.asarray(base_omega, dtype=float)
    else:
        Omega = Omega_in.values if isinstance(Omega_in, pd.DataFrame) else np.asarray(Omega_in, dtype=float)

    # Sigma to ndarray and capture names if DataFrame
    if isinstance(Sigma_in, pd.DataFrame):
        Sigma = Sigma_in.values.astype(float)
        names = list(Sigma_in.columns)
    else:
        Sigma = np.asarray(Sigma_in, dtype=float)
        names = None

    # P and q to arrays
    P = P_in.values if isinstance(P_in, pd.DataFrame) else np.asarray(P_in, dtype=float)
    q = q_in.values if isinstance(q_in, pd.Series) else np.asarray(q_in, dtype=float)

    # benchmark_weights: fallback to uniform if missing
    w_b_in = bl_inputs.get("benchmark_weights", None)
    if w_b_in is None:
        n = Sigma.shape[0]
        w_b = np.ones(n, dtype=float) / float(n)
        print("[BL] Warning: 'benchmark_weights' not provided — using uniform benchmark weights.")
    else:
        w_b = w_b_in.values if isinstance(w_b_in, pd.Series) else np.asarray(w_b_in, dtype=float)

    tau = float(bl_inputs.get("tau", 0.025))
    delta = float(bl_inputs.get("delta", 2.5))

    # basic shape checks
    if Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be square (n x n).")
    n = Sigma.shape[0]
    k = P.shape[0]
    if P.shape[1] != n:
        raise ValueError("P must have shape (k, n) with n == Sigma.shape[0].")
    if q.shape[0] != k:
        raise ValueError("q length must match number of rows of P (k).")
    if Omega.shape[0] != k or Omega.shape[1] != k:
        raise ValueError("Omega must be (k x k).")
    if w_b.shape[0] != n:
        raise ValueError("benchmark_weights must have length n (number of assets).")

    # equilibrium returns (pi)
    pi = delta * (Sigma @ w_b)

    # posterior
    tauSigma = tau * Sigma
    M = P @ tauSigma @ P.T + Omega
    invM = safe_inv(M)
    adj = tauSigma @ P.T @ invM @ (q - P @ pi)
    mu = pi + adj

    if names is not None:
        mu_post = pd.Series(mu, index=names)
        pi_s = pd.Series(pi, index=names)
        Sigma_df = pd.DataFrame(Sigma, index=names, columns=names)
    else:
        mu_post = pd.Series(mu)
        pi_s = pd.Series(pi)
        Sigma_df = pd.DataFrame(Sigma)

    return {"mu_post": mu_post, "Sigma": Sigma_df, "pi": pi_s}


def _project_simplex(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
    if rho.size == 0:
        return np.ones(n) / n
    rho = rho[-1]
    theta = (cssv[rho] - 1) / float(rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    return (w / s) if s > 0 else np.ones(n) / n

def optimize_portfolio(mu: pd.Series,
                       Sigma: pd.DataFrame,
                       benchmark: pd.Series,
                       delta: float = 2.5,
                       target_te: Optional[float] = None,
                       long_only: bool = True) -> pd.Series:
    mu_v = mu.values.astype(float)
    S = Sigma.values.astype(float)
    n = len(mu_v)
    w_b = benchmark.values.astype(float)

    bounds = [(0, 1)] * n if long_only else [(-1, 1)] * n
    def obj(w): return 0.5 * float(w.T @ S @ w) - (1.0 / delta) * float(mu_v.T @ w)
    def grad(w): return S @ w - (1.0 / delta) * mu_v

    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0, 'jac': lambda w: np.ones(n)}]
    if target_te is not None:
        te2 = float(target_te ** 2)
        cons.append({'type': 'ineq',
                     'fun': lambda w: te2 - float((w - w_b).T @ S @ (w - w_b)),
                     'jac': lambda w: -2.0 * (S @ (w - w_b))})

    x0 = w_b.copy()
    s = x0.sum()
    x0 = (x0 / s) if s != 0 else np.ones(n) / n

    res = minimize(obj, x0, jac=grad, bounds=bounds, constraints=cons, options={'ftol': 1e-9, 'maxiter': 1000})
    if res.success:
        w_opt = res.x
        if long_only:
            w_opt = np.clip(w_opt, 0.0, 1.0)
            ss = w_opt.sum()
            w_opt = (w_opt / ss) if ss > 0 else np.ones(n) / n
        return pd.Series(w_opt, index=mu.index)

    try:
        w = (1.0 / delta) * (safe_inv(S) @ mu_v)
    except Exception:
        w = (1.0 / delta) * (np.linalg.pinv(S) @ mu_v)

    if long_only:
        w = _project_simplex(w)
    else:
        s = np.sum(w)
        w = (w / s) if np.isfinite(s) and s != 0 else np.ones(n) / n
    return pd.Series(w, index=mu.index)

def bl_pipeline(bl_inputs: Dict[str, Any],
                target_te: Optional[float] = None,
                long_only: bool = True,
                delta: Optional[float] = None) -> Dict[str, Any]:
    bl = compute_bl_posterior(bl_inputs)
    mu_post, Sigma = bl["mu_post"], bl["Sigma"]
    if delta is None:
        delta = float(bl_inputs.get("delta", 2.5))
    w_b = bl_inputs["benchmark_weights"]
    w = optimize_portfolio(mu_post, Sigma, w_b, delta=delta, target_te=target_te, long_only=long_only)
    return {"mu_post": mu_post, "Sigma": Sigma, "w": w}

if __name__ == "__main__":
    np.random.seed(0)
    names = ["Value", "Size", "Momentum", "Quality", "LowVol", "Growth"]
    N = len(names)
    Sigma = pd.DataFrame(np.diag([0.02] * N), index=names, columns=names)
    P = pd.DataFrame(np.eye(N), index=names, columns=names)
    q = pd.Series(np.random.uniform(-0.001, 0.003, N), index=names)
    Omega = pd.DataFrame(np.diag([1e-6] * N), index=names, columns=names)
    w_b = pd.Series(np.ones(N) / N, index=names)
    bl_inputs = {"P": P, "q": q, "Omega": Omega, "Sigma": Sigma,
                 "tau": 0.025, "delta": 2.5, "benchmark_weights": w_b}
    out = bl_pipeline(bl_inputs, target_te=0.02)
    print("Posterior mu:")
    print(out["mu_post"])
    print("Weights:")
    print(out["w"])
