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
PATH_SP500 = os.path.join(DATA_DIR, "SP500.csv")

OUT_DIR = "data_factors"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FACTOR_RETURNS = os.path.join(OUT_DIR, "factor_returns_daily.csv")
OUT_FACTOR_INDICES = os.path.join(OUT_DIR, "factor_indices.csv")
OUT_ACTIVE_RETURNS = os.path.join(OUT_DIR, "factor_active_returns.csv")
OUT_ACTIVE_INDICES = os.path.join(OUT_DIR, "factor_active_indices.csv")

LOOKBACK_DAYS = 126  # ~6 meses (dias úteis)
TOP_PCT = 0.3        # top 30%

# ----------------------
# Utilitários de leitura
# ----------------------
def read_wide_csv_with_date(path, date_col_name="Data", date_format=None):
    """
    Lê CSV wide (primeira coluna com datas chamada 'Data' ou 'Date').
    Tenta converter a coluna de data usando date_format e fallback para infer.
    Retorna DataFrame com DatetimeIndex.
    """
    df = pd.read_csv(path, dtype=str)
    cols_lower = [c.lower() for c in df.columns]
    if date_col_name.lower() in cols_lower:
        date_col = [c for c in df.columns if c.lower() == date_col_name.lower()][0]
    elif 'date' in cols_lower:
        date_col = [c for c in df.columns if c.lower() == 'date'][0]
    else:
        raise ValueError(f"Arquivo {path} não contém coluna 'Data' nem 'Date'.")
    if date_format:
        df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
    else:
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=False, errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col)
    df = df.replace('', np.nan)
    for c in df.columns:
        # remove vírgulas e converte
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].str.replace(',', ''), errors='coerce')
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.sort_index()
    return df

# ----------------------
# Funções fatoriais & portfolio
# ----------------------
def compute_daily_returns(price_wide):
    return price_wide.pct_change()

def get_quarter_end_rebalance_dates(price_index):
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
    adjusted = sorted(list(dict.fromkeys(adjusted)))
    return adjusted

def last_obs_before(df_wide, date):
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
    if not tickers:
        return pd.Series(dtype=float)
    period = returns_df.loc[(returns_df.index >= start_date) & (returns_df.index <= end_date)]
    if period.empty:
        return pd.Series(dtype=float)
    base_w = pd.Series(1.0 / len(tickers), index=tickers)
    out_idx = []
    out_vals = []
    for dt, row in period.iterrows():
        out_idx.append(dt)
        r = row.reindex(tickers)
        valid = r.dropna()
        if valid.empty:
            out_vals.append(np.nan)
            continue
        w = base_w.reindex(valid.index)
        w = w / w.sum()
        out_vals.append((valid * w).sum())
    return pd.Series(out_vals, index=out_idx)

# ----------------------
# Main
# ----------------------
def main():
    print("Lendo arquivos em /data ...")
    price_wide = read_wide_csv_with_date(PATH_PRICES, date_format=None)
    mcap = read_wide_csv_with_date(PATH_MCAP, date_format=None)
    pe = read_wide_csv_with_date(PATH_PE, date_format=None)
    eps = read_wide_csv_with_date(PATH_EPS, date_format=None)
    roa = read_wide_csv_with_date(PATH_ROA, date_format=None)
    sp500_df = read_wide_csv_with_date(PATH_SP500, date_format=None)

    if sp500_df.shape[1] == 0:
        raise RuntimeError("Arquivo SP500 não contém colunas com preços.")
    sp500_price = sp500_df.iloc[:, 0].astype(float)
    sp500_price.name = "SP500"
    sp500_returns = sp500_price.pct_change()

    for df in (price_wide, mcap, pe, eps, roa):
        df.columns = df.columns.astype(str)

    all_tickers = sorted(set(price_wide.columns) | set(mcap.columns) | set(pe.columns) | set(eps.columns) | set(roa.columns))
    print(f"Tickers detectados (união): {len(all_tickers)}")

    price_wide = price_wide.reindex(columns=all_tickers)
    mcap = mcap.reindex(columns=all_tickers)
    pe = pe.reindex(columns=all_tickers)
    eps = eps.reindex(columns=all_tickers)
    roa = roa.reindex(columns=all_tickers)

    returns_daily = compute_daily_returns(price_wide)

    rebalance_dates = get_quarter_end_rebalance_dates(price_wide.index)
    if not rebalance_dates:
        raise RuntimeError("Nenhuma data de rebalance encontrada no período de preços.")
    print(f"Encontradas {len(rebalance_dates)} datas de rebalance (quarter-ends ajustados).")

    # ------------- localizar start_rb unificado -------------
    # requisito 1: data de preço com LOOKBACK_DAYS disponível (para momentum & lowvol)
    if len(price_wide.index) <= LOOKBACK_DAYS:
        raise RuntimeError("Histórico de preços menor que LOOKBACK_DAYS; não é possível calcular Momentum/LowVol.")
    min_price_ready_date = price_wide.index[LOOKBACK_DAYS]  # primeiro dia com lookback completo

    # requisito 2: EPS - pelo menos um ticker com 5 observações até rb
    # calcular para cada ticker a data da 5a observação não-nula (se existir)
    eps_5th_dates = {}
    for tk in eps.columns:
        non_na = eps[tk].dropna()
        if len(non_na) >= 5:
            eps_5th_dates[tk] = non_na.index[4]  # índice 4 = 5a observação cronológica
    if len(eps_5th_dates) == 0:
        # se nenhum ticker tem 5 EPS obs, growth não poderá ser calculado em lugar nenhum.
        # ainda assim permitimos continuar mas emitimos aviso; growth ficará NaN até EPS acumularem.
        print("Aviso: nenhum ticker tem 5 observações de EPS. Growth ficará NaN até existirem 5 observações.")
        earliest_eps_ready = price_wide.index.max() + pd.Timedelta(days=1)  # força espera inválida
    else:
        earliest_eps_ready = min(eps_5th_dates.values())

    # requisito 3: mcap/pe/roa - ao menos um valor disponível antes da data
    def earliest_non_na_before(df_wide):
        dates = []
        for col in df_wide.columns:
            s = df_wide[col].dropna()
            if not s.empty:
                dates.append(s.index[0])
        return min(dates) if dates else (price_wide.index.max() + pd.Timedelta(days=1))
    earliest_mcap = earliest_non_na_before(mcap)
    earliest_pe = earliest_non_na_before(pe)
    earliest_roa = earliest_non_na_before(roa)

    # escolher start_rb: primeiro rebalance_date que seja >= todos os requisitos
    start_rb = None
    for rb in rebalance_dates:
        if rb >= min_price_ready_date and rb >= earliest_eps_ready and rb >= earliest_mcap and rb >= earliest_pe and rb >= earliest_roa:
            start_rb = rb
            break

    if start_rb is None:
        # fallback conservador: use min_price_ready_date (momento em que preço tem lookback) e avisar
        start_rb = min_price_ready_date
        print("Aviso: não foi possível encontrar rebalance date que satisfaça todos os requisitos simultaneamente.")
        print("Usando data mínima compatível com janela de preços (LOOKBACK_DAYS):", start_rb.date())
    else:
        print("Data inicial unificada para começar a calcular fatores:", start_rb.date())

    # ------------- construir retornos dos fatores só a partir de start_rb -------------
    factor_returns = { 'Size': pd.Series(dtype=float),
                       'Value': pd.Series(dtype=float),
                       'Growth': pd.Series(dtype=float),
                       'Momentum': pd.Series(dtype=float),
                       'Quality': pd.Series(dtype=float),
                       'LowVolatility': pd.Series(dtype=float) }

    # filtrar rebalance_dates para começar em start_rb
    rebalance_dates_filtered = [rb for rb in rebalance_dates if rb >= start_rb]
    if not rebalance_dates_filtered:
        raise RuntimeError("Nenhuma data de rebalance disponível após start_rb.")

    for i, rb in enumerate(rebalance_dates_filtered):
        start_period = rb
        end_period = rebalance_dates_filtered[i+1] if i < len(rebalance_dates_filtered) - 1 else price_wide.index[-1]
        print(f"Rebalance em {rb.date()} -> período até {end_period.date()}")

        mcap_at = last_obs_before(mcap, rb)
        pe_at = last_obs_before(pe, rb)
        roa_at = last_obs_before(roa, rb)

        # EPS growth (t / t-4) - buscar últimos 5 observações **até rb**
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

        trading_days = price_wide.index
        # garantir que idx_rb existe nos trading_days
        idx_rb = trading_days.get_indexer_for([rb])[0]
        start_idx = max(0, idx_rb - LOOKBACK_DAYS)
        price_window = price_wide.iloc[start_idx: idx_rb+1]
        if price_window.shape[0] >= 2:
            momentum = (price_window.iloc[-1] / price_window.iloc[0]) - 1.0
        else:
            momentum = pd.Series(index=price_wide.columns, dtype=float)
        momentum = momentum.replace([np.inf, -np.inf], np.nan)

        returns_window = returns_daily.iloc[max(1, start_idx): idx_rb+1]
        vol = returns_window.std()

        scores = {
            'Size': mcap_at.astype(float),
            'Value': pe_at.astype(float),
            'Growth': eps_growth.astype(float),
            'Momentum': momentum.astype(float),
            'Quality': roa_at.astype(float),
            'LowVolatility': vol.astype(float)
        }

        direction = {
            'Size': True,
            'Value': False,
            'Growth': True,
            'Momentum': True,
            'Quality': True,
            'LowVolatility': False
        }

        for fname, score in scores.items():
            selected = select_top_n(score, top_pct=TOP_PCT, higher_is_better=direction[fname])
            if len(selected) == 0:
                idxs = returns_daily.loc[(returns_daily.index >= start_period) & (returns_daily.index <= end_period)].index
                s = pd.Series([np.nan] * len(idxs), index=idxs)
                factor_returns[fname] = pd.concat([factor_returns[fname], s])
            else:
                s = build_equal_weighted_returns(returns_daily, selected, start_period, end_period)
                factor_returns[fname] = pd.concat([factor_returns[fname], s])

    # limpar duplicados e ordenar
    for k in factor_returns:
        series = factor_returns[k]
        series.index = pd.to_datetime(series.index)
        series = series[~series.index.duplicated(keep='first')]
        series = series.sort_index()
        factor_returns[k] = series

    # base_dates = preços a partir de start_rb (alinhado)
    base_dates = price_wide.index[price_wide.index >= start_rb]

    # DataFrame de retornos diários dos fatores (alinhados às datas base)
    factor_returns_df = pd.DataFrame(index=base_dates)
    for fname, ser in factor_returns.items():
        factor_returns_df[fname] = ser.reindex(base_dates).astype(float)

    # SALVAR retornos diários dos fatores (arquivo separado)
    factor_returns_df.index.name = "Date"
    factor_returns_df.to_csv(OUT_FACTOR_RETURNS, float_format="%.10f")
    print(f"Arquivo gerado: {OUT_FACTOR_RETURNS} (retornos diários dos fatores)")

    # Construir índices cumulativos dos fatores (níveis começando em 1.0)
    factor_indices = pd.DataFrame(index=base_dates)
    for col in factor_returns_df.columns:
        r = factor_returns_df[col]
        level = pd.Series(index=base_dates, dtype=float)
        prev = 1.0
        for d in base_dates:
            val = r.loc[d]
            if pd.isna(val):
                level.loc[d] = prev
            else:
                prev = prev * (1.0 + float(val))
                level.loc[d] = prev
        factor_indices[col] = level
    factor_indices.index.name = "Date"
    factor_indices.to_csv(OUT_FACTOR_INDICES, float_format="%.8f")
    print(f"Arquivo gerado: {OUT_FACTOR_INDICES} (níveis cumulativos dos fatores)")

    # --- CÁLCULO DOS RETORNOS ATIVOS ---
    sp500_ret_aligned = sp500_returns.reindex(base_dates).astype(float)

    # DataFrame de retornos ativos diários (fator - SP500)
    active_returns_df = factor_returns_df.subtract(sp500_ret_aligned, axis=0)
    active_returns_df.index.name = "Date"
    active_returns_df.to_csv(OUT_ACTIVE_RETURNS, float_format="%.10f")
    print(f"Arquivo gerado: {OUT_ACTIVE_RETURNS} (retornos ativos diários)")

    # Índices cumulativos dos retornos ativos (começando em 1.0)
    active_indices = pd.DataFrame(index=base_dates)
    for col in active_returns_df.columns:
        r = active_returns_df[col]
        level = pd.Series(index=base_dates, dtype=float)
        prev = 1.0
        for d in base_dates:
            val = r.loc[d]
            if pd.isna(val):
                level.loc[d] = prev
            else:
                prev = prev * (1.0 + float(val))
                level.loc[d] = prev
        active_indices[col] = level
    active_indices.index.name = "Date"
    active_indices.to_csv(OUT_ACTIVE_INDICES, float_format="%.8f")
    print(f"Arquivo gerado: {OUT_ACTIVE_INDICES} (níveis cumulativos dos retornos ativos)")

if __name__ == "__main__":
    main()
