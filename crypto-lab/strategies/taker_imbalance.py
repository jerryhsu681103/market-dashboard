"""主動買賣失衡延續。

賺誰的錢：資訊較慢的被動掛單方。主動買（賣）量突然異常偏高，代表有資訊或急迫性的
一方正在吃單，價格在接下來數小時傾向延續。
進場：taker 買量比例相對過去 WINDOW 根的 z-score ≥ threshold 做多、≤ -threshold 做空，持有 hold 根。
"""
import pandas as pd

from strategies._util import combine, hold_after

WINDOW = 168  # 一週


def signal(df: pd.DataFrame, threshold: float = 2.5, hold: int = 6) -> pd.Series:
    ratio = (df["taker_buy_volume"] / df["volume"]).where(df["volume"] > 0).fillna(0.5)
    mu = ratio.rolling(WINDOW).mean().shift(1)
    sd = ratio.rolling(WINDOW).std().shift(1)
    z = (ratio - mu) / sd
    long = hold_after(z >= threshold, hold)
    short = hold_after(z <= -threshold, hold)
    return combine(long, short)
