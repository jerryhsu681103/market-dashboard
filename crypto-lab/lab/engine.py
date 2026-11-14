"""回測引擎：所有回測數字的唯一來源。

對外函式：
    run(signal_fn, params, df, fee, slippage, funding) -> dict

成交規則（杜絕前視）：
    - signal_fn(df, **params) 回傳與 df 同索引的 Series，值只能是 1（做多）/ -1（做空）/ 0（空倉）。
    - 第 t 根 K 線的信號，在第 t+1 根的開盤價成交。
    - 持倉在第 t 根期間的報酬 = open[t+1] / open[t] - 1。
    - 預設會做「截斷檢查」：把資料截短後重算信號，若前段信號不同，代表用到了未來資料，直接報錯。

成本（永續合約）：
    - fee、slippage：單邊比例（例如 0.0004 = 4 bps）。沒有預設值，傳 None 直接報錯。
    - 每次部位變動按變動量收取 (fee + slippage)；多翻空算 2 個單位。
    - 資金費率：在結算時點 T 持有部位者支付 position * rate（多單在正費率時付錢）。
      持有「結束於 T 的那根 K 線」的部位，即視為在 T 持倉。
    - 報酬以名目本金 1 倍計（不加槓桿）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ENGINE_VERSION = "1.0.0"
REQUIRED_COLUMNS = ("open", "high", "low", "close")
SECONDS_PER_YEAR = 365 * 24 * 3600


class LookaheadError(Exception):
    """信號函式用到了未來資料。"""


# ---------------------------------------------------------------- 驗證

def _validate_costs(fee, slippage) -> float:
    for name, v in (("fee", fee), ("slippage", slippage)):
        if v is None:
            raise ValueError(
                f"{name} 未設定。請在 config/costs.json 填入你的數值，或在呼叫時明確傳入。"
            )
        if not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 or v >= 0.05:
            raise ValueError(f"{name}={v!r} 不合理：需為 0 ≤ x < 0.05 的單邊比例（例如 0.0004）")
    return float(fee) + float(slippage)


def _validate_df(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df 的索引必須是 DatetimeIndex（K 線開盤時間，UTC）")
    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("df 的索引必須遞增且不重複")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"df 缺少欄位：{missing}")
    if len(df) < 3:
        raise ValueError("資料太短（< 3 根 K 線）")
    if df["open"].isna().any() or (df["open"] <= 0).any():
        raise ValueError("open 欄位有缺值或非正數")


def compute_signals(signal_fn, df: pd.DataFrame, params: dict) -> pd.Series:
    s = signal_fn(df, **params)
    if not isinstance(s, pd.Series):
        s = pd.Series(np.asarray(s), index=df.index)
    if len(s) != len(df) or not s.index.equals(df.index):
        raise ValueError("signal_fn 回傳的 Series 索引必須與 df 完全相同")
    s = s.fillna(0)
    bad = set(np.unique(s.to_numpy())) - {-1, 0, 1}
    if bad:
        raise ValueError(f"信號只能是 -1/0/1，發現：{sorted(bad)[:5]}")
    return s.astype(np.int8)


def check_lookahead(signal_fn, df: pd.DataFrame, params: dict,
                    full: pd.Series | None = None, cuts=(0.5, 0.7, 0.9)) -> None:
    """截斷檢查：用前 k 根資料算出的信號，必須等於用全部資料算出的前 k 根信號。"""
    if full is None:
        full = compute_signals(signal_fn, df, params)
    n = len(df)
    for f in cuts:
        k = int(n * f)
        if k < 2:
            continue
        part = compute_signals(signal_fn, df.iloc[:k], params)
        diff = np.flatnonzero(part.to_numpy() != full.iloc[:k].to_numpy())
        if diff.size:
            t = df.index[diff[0]]
            raise LookaheadError(
                f"前視偏差：截斷到第 {k} 根後，{t} 的信號從 {full.iloc[diff[0]]} 變成 "
                f"{part.iloc[diff[0]]}（共 {diff.size} 根不同）。信號用到了未來資料。"
            )


# ---------------------------------------------------------------- 核心模擬

def infer_bars_per_year(index: pd.DatetimeIndex) -> float:
    step = pd.Series(index).diff().median().total_seconds()
    return SECONDS_PER_YEAR / step


def funding_at_bar_end(index: pd.DatetimeIndex, funding: pd.Series) -> np.ndarray:
    """回傳每根 K 線「結束時點」的資金費率（無結算則為 0）。"""
    step = pd.Series(index).diff().median()
    ends = index + step
    return funding.reindex(ends).fillna(0.0).to_numpy(float)


def simulate(pos: np.ndarray, bar_ret: np.ndarray, fund_end: np.ndarray,
             cost_rate: float, index: pd.DatetimeIndex) -> pd.DataFrame:
    """給定每根 K 線期間的持倉，計算逐根損益明細。

    pos[t]：第 t 根 K 線期間（open[t] → open[t+1]）的持倉。
    """
    pos = np.asarray(pos, float)
    prev = np.concatenate([[0.0], pos[:-1]])
    changed = pos != prev
    entry_cost = np.where(changed, np.abs(pos), 0.0) * cost_rate
    exit_cost = np.where(changed, np.abs(prev), 0.0) * cost_rate
    gross = pos * bar_ret
    funding_pnl = -pos * fund_end
    net = gross + funding_pnl - entry_cost - exit_cost
    return pd.DataFrame(
        {"pos": pos, "bar_ret": bar_ret, "gross": gross, "funding": funding_pnl,
         "entry_cost": entry_cost, "exit_cost": exit_cost, "net": net},
        index=index,
    )


def _trade_pnls(detail: pd.DataFrame) -> np.ndarray:
    """每筆交易的損益（單利加總）：持倉段的毛利 + 資金費 − 進場成本 − 出場成本。"""
    pos = detail["pos"].to_numpy()
    n = len(pos)
    pnls = []
    t = 0
    while t < n:
        if pos[t] == 0:
            t += 1
            continue
        a = t
        while t + 1 < n and pos[t + 1] == pos[a]:
            t += 1
        b = t
        seg = detail.iloc[a:b + 1]
        pnl = seg["gross"].sum() + seg["funding"].sum() - detail["entry_cost"].iat[a]
        if b + 1 < n:
            pnl -= detail["exit_cost"].iat[b + 1]
        pnls.append(pnl)
        t = b + 1
    return np.asarray(pnls, float)


def summarize(detail: pd.DataFrame, bars_per_year: float) -> dict:
    net = detail["net"]
    equity = (1.0 + net).cumprod()
    trades = _trade_pnls(detail)
    wins, losses = trades[trades > 0].sum(), -trades[trades < 0].sum()
    if losses > 0:
        pf = float(wins / losses)
    else:
        pf = float("inf") if wins > 0 else float("nan")
    sd = net.std(ddof=1)
    sharpe = float(net.mean() / sd * np.sqrt(bars_per_year)) if sd > 0 else float("nan")
    dd = equity / equity.cummax() - 1.0
    years = len(net) / bars_per_year
    total = float(equity.iat[-1] - 1.0)
    cagr = float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else float("nan")
    return {
        "returns": net,
        "equity": equity,
        "total_return": total,
        "cagr": cagr,
        "profit_factor": pf,
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()),
        "trade_count": int(len(trades)),
        "win_rate": float((trades > 0).mean()) if len(trades) else float("nan"),
        "exposure": float((detail["pos"] != 0).mean()),
        "total_cost": float((detail["entry_cost"] + detail["exit_cost"]).sum()),
        "total_funding": float(detail["funding"].sum()),
        "open_at_end": bool(detail["pos"].iat[-1] != 0),
        "bars": int(len(net)),
        "start": str(detail.index[0]),
        "end": str(detail.index[-1]),
        "bars_per_year": float(bars_per_year),
        "engine_version": ENGINE_VERSION,
    }


# ---------------------------------------------------------------- 對外介面

def run(signal_fn, params: dict, df: pd.DataFrame, fee, slippage,
        funding: pd.Series | None = None, *, allow_no_funding: bool = False,
        check_lookahead_bias: bool = True, keep_detail: bool = False) -> dict:
    """跑一次回測。

    funding：以結算時點（UTC 整點）為索引的資金費率 Series。永續合約回測必填；
             若確定不需要（例如單元測試），需同時傳 allow_no_funding=True。
    """
    cost_rate = _validate_costs(fee, slippage)
    _validate_df(df)
    if funding is None and not allow_no_funding:
        raise ValueError("永續合約回測必須提供 funding（資金費率）。")

    sig = compute_signals(signal_fn, df, params)
    if check_lookahead_bias:
        check_lookahead(signal_fn, df, params, full=sig)

    pos = sig.shift(1).fillna(0).to_numpy(float)  # 第 t-1 根信號 → 第 t 根開盤成交
    o = df["open"].to_numpy(float)
    bar_ret = np.zeros(len(o))
    bar_ret[:-1] = o[1:] / o[:-1] - 1.0
    fund_end = (funding_at_bar_end(df.index, funding) if funding is not None
                else np.zeros(len(o)))

    detail = simulate(pos, bar_ret, fund_end, cost_rate, df.index)
    out = summarize(detail, infer_bars_per_year(df.index))
    out.update({"params": dict(params), "fee": float(fee), "slippage": float(slippage),
                "signals": sig})
    if keep_detail:
        out["detail"] = detail
    return out


def metrics_table(result: dict) -> pd.Series:
    """把 run() 的純量結果整理成一張表，方便印出。"""
    keys = ["start", "end", "bars", "total_return", "cagr", "profit_factor", "sharpe",
            "max_drawdown", "trade_count", "win_rate", "exposure", "total_cost",
            "total_funding", "open_at_end", "fee", "slippage", "engine_version"]
    return pd.Series({k: result[k] for k in keys})


# ---------------------------------------------------------------- 多腳（價差）部位

def run_spread(signal_fn, params: dict, df: pd.DataFrame, fee, slippage,
               funding: dict, *, legs: dict, check_lookahead_bias: bool = True,
               keep_detail: bool = False) -> dict:
    """價差回測：一個信號同時驅動多腳部位。

    df：同一時間索引，欄位以幣種為前綴，例如 ETHUSDT_open、BTCUSDT_open（至少要有 {sym}_open）。
    legs：{幣種: 權重}，例如 {"ETHUSDT": 1, "BTCUSDT": -1}。
          信號 s ∈ {-1,0,1}；第 i 腳持倉 = s × 權重（每腳名目本金 = |權重|）。
    funding：{幣種: 資金費率 Series}，每一腳都必須提供。
    成交、成本、資金費規則與 run() 相同，逐腳計算後加總；交易以價差信號的連續持倉段計。
    假設：每根 K 線結束時各腳名目本金恢復為固定值（與 run() 相同的固定名目假設），再平衡不另收成本。
    """
    cost_rate = _validate_costs(fee, slippage)
    if not isinstance(df.index, pd.DatetimeIndex) or not df.index.is_monotonic_increasing \
            or not df.index.is_unique:
        raise ValueError("df 的索引必須是遞增且不重複的 DatetimeIndex")
    for sym in legs:
        col = f"{sym}_open"
        if col not in df.columns:
            raise ValueError(f"df 缺少欄位 {col}")
        if df[col].isna().any() or (df[col] <= 0).any():
            raise ValueError(f"{col} 有缺值或非正數")
        if funding is None or sym not in funding:
            raise ValueError(f"缺少 {sym} 的資金費率")

    sig = compute_signals(signal_fn, df, params)
    if check_lookahead_bias:
        check_lookahead(signal_fn, df, params, full=sig)
    s = sig.shift(1).fillna(0).to_numpy(float)

    parts = []
    for sym, w in legs.items():
        o = df[f"{sym}_open"].to_numpy(float)
        bar_ret = np.zeros(len(o))
        bar_ret[:-1] = o[1:] / o[:-1] - 1.0
        parts.append(simulate(s * w, bar_ret, funding_at_bar_end(df.index, funding[sym]),
                              cost_rate, df.index))
    detail = pd.DataFrame({
        "pos": s,  # 價差部位（用來切分交易）
        "gross": sum(p["gross"] for p in parts),
        "funding": sum(p["funding"] for p in parts),
        "entry_cost": sum(p["entry_cost"] for p in parts),
        "exit_cost": sum(p["exit_cost"] for p in parts),
    }, index=df.index)
    detail["net"] = detail["gross"] + detail["funding"] - detail["entry_cost"] - detail["exit_cost"]
    out = summarize(detail, infer_bars_per_year(df.index))
    out.update({"params": dict(params), "fee": float(fee), "slippage": float(slippage),
                "signals": sig, "legs": dict(legs)})
    if keep_detail:
        out["detail"] = detail
        out["leg_detail"] = dict(zip(legs, parts))
    return out
