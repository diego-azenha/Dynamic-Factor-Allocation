# black_litterman.py
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scipy import linalg
from scipy.optimize import minimize


# ---- Posterior ----
def compute_bl_posterior(bl_inputs: Dict[str, Any]) -> Dict[str, Any]:
    P = np.asarray(bl_inputs["P"])
    q = np.asarray(bl_inputs["q"])
    Omega = np.asarray(bl_inputs["Omega"])
    Sigma = np.asarray(bl_inputs["Sigma"])
    tau = float(bl_inputs.get("tau", 0.025))
    delta = float(bl_inputs.get("delta", 2.5))
    w_b = np.asarray(bl_inputs["benchmark_weights"])

    pi = delta * (Sigma @ w_b)
    tauSigma = tau * Sigma
    M = P @ tauSigma @ P.T + Omega
    invM = linalg.inv(M + 1e-10 * np.eye(M.shape[0]))
    mu = pi + tauSigma @ P.T @ invM @ (q - P @ pi)

    names = None
    if isinstance(bl_inputs.get("Sigma"), pd.DataFrame):
        names = bl_inputs["Sigma"].columns
    mu_post = pd.Series(mu, index=names)
    pi = pd.Series(pi, index=names)
    Sigma_df = pd.DataFrame(Sigma, index=names, columns=names)
    return {"mu_post": mu_post, "Sigma": Sigma_df, "pi": pi}


# ---- Helpers ----
def _project_simplex(v: np.ndarray) -> np.ndarray:
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / float(rho + 1)
    w = np.maximum(v - theta, 0)
    return w / (w.sum() + 1e-12)


# ---- Optimization ----
def optimize_portfolio(mu: pd.Series,
                       Sigma: pd.DataFrame,
                       benchmark: pd.Series,
                       delta: float = 2.5,
                       target_te: Optional[float] = None,
                       long_only: bool = True) -> pd.Series:
    mu_v = mu.values
    S = Sigma.values
    n = len(mu_v)
    w_b = benchmark.values
    bounds = [(0, 1)] * n if long_only else [(-1, 1)] * n

    def obj(w): return 0.5 * w.T @ S @ w - (1 / delta) * mu_v.T @ w
    def grad(w): return S @ w - (1 / delta) * mu_v

    cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1, 'jac': lambda w: np.ones(n)}]
    if target_te:
        te2 = target_te ** 2
        cons.append({'type': 'ineq',
                     'fun': lambda w: te2 - (w - w_b).T @ S @ (w - w_b),
                     'jac': lambda w: -2 * S @ (w - w_b)})

    x0 = w_b / w_b.sum()
    res = minimize(obj, x0, jac=grad, bounds=bounds, constraints=cons,
                   options={'ftol': 1e-9, 'maxiter': 1000})

    if not res.success:
        try:
            w = (1 / delta) * linalg.inv(S) @ mu_v
        except np.linalg.LinAlgError:
            w = (1 / delta) * np.linalg.pinv(S) @ mu_v
        w = _project_simplex(w) if long_only else w / w.sum()
        return pd.Series(w, index=mu.index)

    return pd.Series(res.x, index=mu.index)


# ---- Pipeline ----
def bl_pipeline(bl_inputs: Dict[str, Any],
                target_te: Optional[float] = None,
                long_only: bool = True,
                delta: Optional[float] = None) -> Dict[str, Any]:
    bl = compute_bl_posterior(bl_inputs)
    mu_post, Sigma = bl["mu_post"], bl["Sigma"]
    if delta is None:
        delta = bl_inputs.get("delta", 2.5)
    w_b = bl_inputs["benchmark_weights"]
    w = optimize_portfolio(mu_post, Sigma, w_b, delta=delta,
                           target_te=target_te, long_only=long_only)
    return {"mu_post": mu_post, "Sigma": Sigma, "w": w}


# ---- Quick test ----
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
