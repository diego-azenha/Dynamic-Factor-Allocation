import os
import math
import numpy as np
import pandas as pd

# ----------------------
# Config
# ----------------------
DATA_DIR = "data"
PATH_PRICES = os.path.join(DATA_DIR, "stock_prices.csv")
PATH_MCAP = os.path.join(DATA_DIR, "market_cap.csv")
PATH_PE = os.path.join(DATA_DIR, "price_to_earnings.csv")
PATH_EPS = os.path.join(DATA_DIR, "EPS.csv")
PATH_ROA = os.path.join(DATA_DIR, "ROA.csv")
OUT_CSV = "factor_indices.csv"

LOOKBACK_DAYS = 126  # ~6 meses (dias úteis)
TOP_PCT = 0.3        # top 30%

# ----------------------
# Utilitários de leitura
# ----------------------
def read_wide_csv_with_date(path, date_col_name="Data", date_format=None):
    """
    Lê CSV wide (primeira coluna com datas chamada 'Data' ou 'Date').
    Tenta converter a coluna de data usando date_format (ex: '%m/%d/%Y') e fallback para infer.
    Retorna DataFrame com DatetimeIndex.
    """
    df = pd.read_csv(path, dtype=str)
    # identificar coluna de data (case-insensitive)
    cols_lower = [c.lower() for c in df.columns]
    if date_col_name.lower() in cols_lower:
        date_col = [c for c in df.columns if c.lower() == date_col_name.lower()][0]
    elif 'date' in cols_lower:
        date_col = [c for c in df.columns if c.lower() == 'date'][0]
    else:
        # assume índice é já coluna de data ou erro
        # tentar converter índice (não muito provável aqui)
        raise ValueError(f"Arquivo {path} não contém coluna 'Data' nem 'Date'.")
    # converter data
    if date_format:
        df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
    else:
        # tentar inferir; a amostra fornecida usa MM/DD/YYYY, então dayfirst=False
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=False, errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col)
    # converter colunas restantes para float
    other_cols = [c for c in df.columns]
    df = df.replace('', np.nan)
    for c in other_cols:
        # remover possíveis espaços e converter
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.sort_index()
    return df

# ----------------------
# Funções fatoriais & portfolio
# ----------------------
def compute_daily_returns(price_wide):
    return price_wide.pct_change()

def get_quarter_end_rebalance_dates(price_index):
    """
    Retorna lista de datas de rebalance ajustadas para último dia de trading <= quarter-end.
    """
    start = price_index.min()
    end = price_index.max()
    quarters = pd.period_range(start=start, end=end, freq='Q')
    quarter_ends = [q.end_time.normalize() for q in quarters]
    trading_days = price_index
    adjusted = []
    for qe in quarter_ends:
        valid = trading_days[trading_days <= qe]
        if len(valid) == 0:
            continue
        adjusted.append(valid[-1])
    # remover duplicatas e ordenar
    adjusted = sorted(list(dict.fromkeys(adjusted)))
    return adjusted

def last_obs_before(df_wide, date):
    """Retorna Series com última observação <= date para cada coluna (ticker)."""
    slice_df = df_wide[df_wide.index <= date]
    if slice_df.empty:
        return pd.Series(index=df_wide.columns, dtype=float)
    return slice_df.iloc[-1].astype(float)

def select_top_n(scores: pd.Series, top_pct=0.3, higher_is_better=True):
    s = scores.dropna()
    if s.empty:
        return []
    n_top = int(math.ceil(top_pct * len(s)))
    n_top = max(1, n_top)
    sorted_s = s.sort_values(ascending=not higher_is_better)
    return sorted_s.iloc[:n_top].index.tolist()

def build_equal_weighted_returns(returns_df, tickers, start_date, end_date):
    """
    Retorna Series de retornos diários do portfólio equal-weighted por tickers entre start_date e end_date (ambos inclusive).
    Renormaliza pesos nos dias com NaN para alguns tickers.
    """
    if not tickers:
        return pd.Series(dtype=float)
    period = returns_df.loc[(returns_df.index >= start_date) & (returns_df.index <= end_date)]
    if period.empty:
        return pd.Series(dtype=float)
    base_w = pd.Series(1.0 / len(tickers), index=tickers)
    out = []
    for dt, row in period.iterrows():
        r = row.reindex(tickers)
        valid = r.dropna()
        if valid.empty:
            out.append(np.nan)
            continue
        w = base_w.reindex(valid.index)
        w = w / w.sum()
        out.append((valid * w).sum())
    return pd.Series(out, index=period.index)

# ----------------------
# Main
# ----------------------
def main():
    print("Lendo arquivos em /data ...")
    # usar formato mm/dd/yyyy como default (conforme imagem), mas permitir inferência
    price_wide = read_wide_csv_with_date(PATH_PRICES, date_format=None)
    mcap = read_wide_csv_with_date(PATH_MCAP, date_format=None)
    pe = read_wide_csv_with_date(PATH_PE, date_format=None)
    eps = read_wide_csv_with_date(PATH_EPS, date_format=None)
    roa = read_wide_csv_with_date(PATH_ROA, date_format=None)

    # normalizar nomes de colunas para str
    for df in (price_wide, mcap, pe, eps, roa):
        df.columns = df.columns.astype(str)

    # universe de tickers: união de todas as colunas (permitir que fatores tenham disponibilidade diferente)
    all_tickers = sorted(set(price_wide.columns) | set(mcap.columns) | set(pe.columns) | set(eps.columns) | set(roa.columns))
    print(f"Tickers detectados (união): {len(all_tickers)}")

    # reindex tables to same columns (preencher com NaN onde não existe)
    price_wide = price_wide.reindex(columns=all_tickers)
    mcap = mcap.reindex(columns=all_tickers)
    pe = pe.reindex(columns=all_tickers)
    eps = eps.reindex(columns=all_tickers)
    roa = roa.reindex(columns=all_tickers)

    # retornos diários
    returns_daily = compute_daily_returns(price_wide)

    # datas de rebalance trimestral
    rebalance_dates = get_quarter_end_rebalance_dates(price_wide.index)
    if not rebalance_dates:
        raise RuntimeError("Nenhuma data de rebalance encontrada no período de preços.")
    print(f"Encontradas {len(rebalance_dates)} datas de rebalance (quarter-ends ajustados).")

    # containers de retornos diários por fator (append por período)
    factor_returns = { 'Size': pd.Series(dtype=float),
                       'Value': pd.Series(dtype=float),
                       'Growth': pd.Series(dtype=float),
                       'Momentum': pd.Series(dtype=float),
                       'Quality': pd.Series(dtype=float),
                       'LowVolatility': pd.Series(dtype=float) }

    # para cada rebalance, calcular scores e formar portfólio até o próximo rebalance
    for i, rb in enumerate(rebalance_dates):
        start_period = rb
        if i < len(rebalance_dates) - 1:
            end_period = rebalance_dates[i+1]
        else:
            end_period = price_wide.index[-1]
        print(f"Rebalance em {rb.date()} -> período até {end_period.date()}")

        # pegar últimas observações <= rb
        mcap_at = last_obs_before(mcap, rb)
        pe_at = last_obs_before(pe, rb)
        roa_at = last_obs_before(roa, rb)

        # EPS growth: EPS_t / EPS_{t-4} - 1 (precisa de 5 observações históricas por ticker)
        eps_growth = pd.Series(index=eps.columns, dtype=float)
        for tk in eps.columns:
            s = eps[tk].dropna()
            s = s[s.index <= rb]
            if len(s) >= 5:
                val_t = s.iloc[-1]
                val_t4 = s.iloc[-5]
                if pd.isna(val_t) or pd.isna(val_t4) or val_t4 == 0:
                    eps_growth[tk] = np.nan
                else:
                    eps_growth[tk] = (val_t / val_t4) - 1.0
            else:
                eps_growth[tk] = np.nan

        # momentum: retorno cumulativo na janela LOOKBACK_DAYS imediatamente anterior ao rb (first->last)
        trading_days = price_wide.index
        # idx_rb: índice do rb em trading_days (rb foi ajustado para trading day)
        idx_rb = trading_days.get_indexer_for([rb])[0]
        start_idx = max(0, idx_rb - LOOKBACK_DAYS)
        price_window = price_wide.iloc[start_idx: idx_rb+1]
        # se a janela tem apenas NaNs, momentum será NaN
        momentum = (price_window.iloc[-1] / price_window.iloc[0]) - 1.0
        momentum = momentum.replace([np.inf, -np.inf], np.nan)

        # volatility: std dos retornos diários na mesma janela
        returns_window = returns_daily.iloc[max(1, start_idx): idx_rb+1]  # evitar idx 0 para pct_change
        vol = returns_window.std()

        # agrupa scores
        scores = {
            'Size': mcap_at.astype(float),
            'Value': pe_at.astype(float),
            'Growth': eps_growth.astype(float),
            'Momentum': momentum.astype(float),
            'Quality': roa_at.astype(float),
            'LowVolatility': vol.astype(float)
        }

        # definição de direção (higher_is_better)
        direction = {
            'Size': True,
            'Value': False,        # menor P/E melhor
            'Growth': True,
            'Momentum': True,
            'Quality': True,
            'LowVolatility': False # menor vol melhor
        }

        # para cada fator, selecionar top 30% e construir retornos equal-weighted para o período
        for fname, score in scores.items():
            selected = select_top_n(score, top_pct=TOP_PCT, higher_is_better=direction[fname])
            if len(selected) == 0:
                # preencher período com NaNs se não houver seleção
                idxs = returns_daily.loc[(returns_daily.index >= start_period) & (returns_daily.index <= end_period)].index
                s = pd.Series([np.nan] * len(idxs), index=idxs)
                factor_returns[fname] = pd.concat([factor_returns[fname], s])  # Use pd.concat instead of append
            else:
                s = build_equal_weighted_returns(returns_daily, selected, start_period, end_period)
                factor_returns[fname] = pd.concat([factor_returns[fname], s])  # Use pd.concat instead of append

    # limpar duplicados e ordenar
    for k in factor_returns:
        series = factor_returns[k]
        series.index = pd.to_datetime(series.index)
        series = series[~series.index.duplicated(keep='first')]
        series = series.sort_index()
        factor_returns[k] = series

    # montar DataFrame final com base nas datas de price_wide (diárias)
    base_dates = price_wide.index
    result = pd.DataFrame(index=base_dates)

    # converter retornos diários em índices cumulativos começando em 1.0
    for fname, ret_series in factor_returns.items():
        rs = ret_series.reindex(base_dates)
        level = pd.Series(index=base_dates, dtype=float)
        prev = 1.0
        for d in base_dates:
            r = rs.loc[d]
            if pd.isna(r):
                # mantém o nível anterior
                level.loc[d] = prev
            else:
                prev = prev * (1.0 + float(r))
                level.loc[d] = prev
        result[fname if fname != 'LowVolatility' else 'LowVolatility'] = level

    result.index.name = 'Date'
    # salvar CSV
    result.to_csv(OUT_CSV, float_format="%.8f")
    print(f"Arquivo gerado: {OUT_CSV}")

if __name__ == "__main__":
    main()
