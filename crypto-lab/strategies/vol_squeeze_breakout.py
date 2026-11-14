"""波動壓縮後突破（為 4 小時線設計）。

賺誰的錢：在低波動期賣出波動（掛區間單、賣選擇權）的一方。波動有群聚性，
長時間壓縮後的擴張常伴隨停損單連鎖觸發；壓縮期的區間突破比一般突破更可能延續。
與已淘汰 donchian_trend 的差異：donchian 在任何時候突破多週通道都進場、持有到趨勢反轉；
本點子只在「波動處於過去半年低檔」時的短期區間突破進場，持有有上限，賺的是波動擴張而非長期趨勢。
進場：前一根處於壓縮（80 小時布林寬度在過去 180 天的百分位 ≤ squeeze_pct），
     收盤突破過去 channel 根的最高（最低）價 → 做多（做空）。
出場：收盤反向穿越 80 小時均線，或持有滿 10 天。
"""
import numpy as np
import pandas as pd

from strategies._util import bars

BAND_H = 80
RANK_H = 180 * 24
MAX_HOLD_H = 240


def signal(df: pd.DataFrame, squeeze_pct: float = 0.1, channel: int = 12) -> pd.Series:
    c = df["close"]
    nb, nr, max_hold = bars(df, BAND_H), bars(df, RANK_H), bars(df, MAX_HOLD_H)
    mid = c.rolling(nb).mean()
    width = c.rolling(nb).std() / mid
    squeezed = (width.rolling(nr).rank(pct=True) <= squeeze_pct).shift(1, fill_value=False).to_numpy()
    hi = df["high"].rolling(channel).max().shift(1).to_numpy()
    lo = df["low"].rolling(channel).min().shift(1).to_numpy()
    cv, mv = c.to_numpy(), mid.to_numpy()
    out = np.zeros(len(c), dtype=np.int8)
    pos, held = 0, 0
    for t in range(len(c)):
        if np.isnan(mv[t]) or np.isnan(hi[t]):
            continue
        if pos != 0:
            held += 1
            if held >= max_hold or (pos == 1 and cv[t] < mv[t]) or (pos == -1 and cv[t] > mv[t]):
                pos = 0
        if pos == 0 and squeezed[t]:
            if cv[t] > hi[t]:
                pos, held = 1, 0
            elif cv[t] < lo[t]:
                pos, held = -1, 0
        out[t] = pos
    return pd.Series(out, index=df.index)
