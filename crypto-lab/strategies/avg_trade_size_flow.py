"""大單資金流（為 4 小時線設計）。

賺誰的錢：反應較慢的小額參與者。平均單筆成交額（quote_volume / trades）明顯高於近期水準時，
代表該根 K 線由大額參與者主導；假設大額參與者資訊較充分或拆單尚未完成，順著該根 K 線方向。
信號：log(平均單筆成交額) 相對過去 30 天的 z-score ≥ threshold 時，
     收盤 > 開盤 → 做多；收盤 < 開盤 → 做空。持有 hold 根。
失效情境：大單來自做市、對敲或強平（非資訊性）。
與 taker_imbalance 的差異：該點子看主動買賣「比例」，本點子看單筆「規模」，方向取自 K 線本身。
"""
import numpy as np
import pandas as pd

from strategies._util import bars, combine, hold_after

Z_H = 30 * 24


def signal(df: pd.DataFrame, threshold: float = 1.5, hold: int = 6) -> pd.Series:
    nz = bars(df, Z_H)
    ats = np.log((df["quote_volume"] / df["trades"]).where(df["trades"] > 0))
    mp = int(nz * 0.9)
    z = (ats - ats.rolling(nz, min_periods=mp).mean().shift(1)) / ats.rolling(nz, min_periods=mp).std().shift(1)
    big = z >= threshold
    long = hold_after(big & (df["close"] > df["open"]), hold)
    short = hold_after(big & (df["close"] < df["open"]), hold)
    return combine(long, short)
