"""槓桿堆積後突破（資料：未平倉量）。

賺誰的錢：突破時站錯邊的槓桿方。未平倉量在過去 7 天明顯堆積後，一旦價格突破區間，
反向部位被迫止損或強平，推動行情延續。
進場：前一根的「7 天未平倉量對數變化」相對過去 90 天的 z-score ≥ oi_z，
     且收盤突破過去 channel 根最高（最低）價 → 做多（做空）。
出場：收盤反向穿越 channel 根的中線，或持有滿 10 天。
需要欄位：oi（attach_features(..., ["metrics"])）。
"""
import numpy as np
import pandas as pd

from strategies._util import bars

CHG_H = 7 * 24
Z_H = 90 * 24
MAX_HOLD_H = 240


def signal(df: pd.DataFrame, oi_z: float = 1.0, channel: int = 12) -> pd.Series:
    nc, nz, max_hold = bars(df, CHG_H), bars(df, Z_H), bars(df, MAX_HOLD_H)
    chg = np.log(df["oi"]).diff(nc)
    mp = int(nz * 0.9)
    z = (chg - chg.rolling(nz, min_periods=mp).mean().shift(1)) / chg.rolling(nz, min_periods=mp).std().shift(1)
    built = (z >= oi_z).shift(1, fill_value=False).to_numpy()
    hi = df["high"].rolling(channel).max().shift(1).to_numpy()
    lo = df["low"].rolling(channel).min().shift(1).to_numpy()
    mid = (hi + lo) / 2
    c = df["close"].to_numpy()
    out = np.zeros(len(c), dtype=np.int8)
    pos, held = 0, 0
    for t in range(len(c)):
        if np.isnan(hi[t]):
            continue
        if pos != 0:
            held += 1
            if held >= max_hold or (pos == 1 and c[t] < mid[t]) or (pos == -1 and c[t] > mid[t]):
                pos = 0
        if pos == 0 and built[t]:
            if c[t] > hi[t]:
                pos, held = 1, 0
            elif c[t] < lo[t]:
                pos, held = -1, 0
        out[t] = pos
    return pd.Series(out, index=df.index)
