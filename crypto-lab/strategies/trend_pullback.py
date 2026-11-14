"""趨勢中的回檔進場。

賺誰的錢：在既有趨勢中因短線波動而恐慌停損或獲利了結的弱手。
與已淘汰的 donchian_trend 的差異：donchian 在「突破」時追價進場、持有到趨勢反轉；
這裡在「趨勢內的短線超跌／超漲」時逆短線進場、短線回到均值就出場，賺的是回檔修復，不是趨勢延續。
進場：收盤 > trend_len 根均線（上升趨勢）且 24 小時 z-score ≤ -threshold → 做多；下降趨勢對稱做空。
出場：z-score 回到 0 以上（空單為 0 以下），或持有滿 48 小時。
"""
import numpy as np
import pandas as pd

from strategies._util import bars

MAX_HOLD_H = 48  # 小時
Z_H = 24


def signal(df: pd.DataFrame, trend_len: int = 720, threshold: float = 2.0) -> pd.Series:
    c = df["close"]
    max_hold, nz = bars(df, MAX_HOLD_H), bars(df, Z_H)
    trend = (c - c.rolling(trend_len).mean()).to_numpy()
    z = ((c - c.rolling(nz).mean()) / c.rolling(nz).std()).to_numpy()
    out = np.zeros(len(c), dtype=np.int8)
    pos, held = 0, 0
    for t in range(len(c)):
        if np.isnan(trend[t]) or np.isnan(z[t]):
            continue
        if pos != 0:
            held += 1
            if held >= max_hold or (pos == 1 and z[t] >= 0) or (pos == -1 and z[t] <= 0):
                pos = 0
        if pos == 0:
            if trend[t] > 0 and z[t] <= -threshold:
                pos, held = 1, 0
            elif trend[t] < 0 and z[t] >= threshold:
                pos, held = -1, 0
        out[t] = pos
    return pd.Series(out, index=df.index)
