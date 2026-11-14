"""每日開盤區間突破（UTC 日內動能，4 小時線專用）。

賺誰的錢：日內較晚進場的追價資金。加密貨幣的日內動能有文獻支持：
當日前段的方向性移動，傾向在當日後段延續（資訊逐步擴散、日內交易者跟單）。
與已淘汰點子的差異：donchian 是多週通道、持有數週；session_seasonality 是固定時段無條件做多；
本點子每天依開盤區間是否被突破決定方向（可多可空），且只在突破日交易、當日收盤前出場。
規則：每個 UTC 日第一根 4 小時 K 線（00:00–04:00）定義區間 [L, H]、寬度 R。
     之後的 K 線收盤 > H + buffer×R 做多、< L − buffer×R 做空（每天只取第一次觸發），持有到當日結束（次日 00:00 開盤出場）。
     過濾：R ≥ min_ratio × 過去 14 天平均日振幅（只用已完成的日子）。
"""
import numpy as np
import pandas as pd

from strategies._util import step_hours


def signal(df: pd.DataFrame, buffer: float = 0.0, min_ratio: float = 0.0) -> pd.Series:
    if int(step_hours(df)) != 4:
        raise ValueError("daily_orb 只適用 4 小時線")
    idx = df.index
    day = idx.floor("D")
    hour = idx.hour
    first = hour == 0
    h0 = pd.Series(np.where(first, df["high"], np.nan), index=idx).groupby(day).transform("first")
    l0 = pd.Series(np.where(first, df["low"], np.nan), index=idx).groupby(day).transform("first")
    has_first = pd.Series(first, index=idx).groupby(day).transform("any")
    r = h0 - l0
    daily_range = (df["high"].groupby(day).max() - df["low"].groupby(day).min())
    avg_prev = daily_range.rolling(14).mean().shift(1)  # 只用已完成的前 14 天
    ok_range = r >= min_ratio * avg_prev.reindex(day).to_numpy()

    c = df["close"].to_numpy()
    hv, lv, rv = h0.to_numpy(), l0.to_numpy(), r.to_numpy()
    ok = (has_first.to_numpy() & ok_range.to_numpy())
    hr = np.asarray(hour)
    dv = np.asarray(day)
    out = np.zeros(len(c), dtype=np.int8)
    cur_day, pos = None, 0
    for t in range(len(c)):
        if dv[t] != cur_day:
            cur_day, pos = dv[t], 0
        if hr[t] == 0 or hr[t] >= 20 or not ok[t]:
            # 00:00 這根定義區間；20:00 這根的信號必須為 0，讓部位在次日 00:00 開盤出場
            out[t] = 0 if hr[t] >= 20 else pos
            continue
        if pos == 0:
            if c[t] > hv[t] + buffer * rv[t]:
                pos = 1
            elif c[t] < lv[t] - buffer * rv[t]:
                pos = -1
        out[t] = pos
    return pd.Series(out, index=df.index)
