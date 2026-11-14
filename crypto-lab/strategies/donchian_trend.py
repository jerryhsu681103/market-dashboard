"""大級別突破後的跟隨（Donchian 通道趨勢）。

賺誰的錢：突破後才追進的晚到資金、被迫回補的空方，以及逆勢掛單者。
加密貨幣的趨勢延續有長期文獻支持（time-series momentum）。
進場：收盤突破過去 entry 根的最高價 → 做多；跌破最低價 → 做空。
出場：反向觸及過去 exit = entry × exit_ratio 根的通道。
"""
import numpy as np
import pandas as pd


def signal(df: pd.DataFrame, entry: int = 240, exit_ratio: float = 0.5) -> pd.Series:
    exit_n = max(2, int(entry * exit_ratio))
    hi_in = df["high"].rolling(entry).max().shift(1).to_numpy()
    lo_in = df["low"].rolling(entry).min().shift(1).to_numpy()
    hi_out = df["high"].rolling(exit_n).max().shift(1).to_numpy()
    lo_out = df["low"].rolling(exit_n).min().shift(1).to_numpy()
    c = df["close"].to_numpy()
    out = np.zeros(len(c), dtype=np.int8)
    pos = 0
    for t in range(len(c)):
        if np.isnan(hi_in[t]):
            continue
        if pos == 1 and c[t] < lo_out[t]:
            pos = 0
        elif pos == -1 and c[t] > hi_out[t]:
            pos = 0
        if c[t] > hi_in[t]:
            pos = 1
        elif c[t] < lo_in[t]:
            pos = -1
        out[t] = pos
    return pd.Series(out, index=df.index)
