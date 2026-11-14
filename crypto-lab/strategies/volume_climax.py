"""量能高潮反轉。

賺誰的錢：最後一批追高（殺低）的資金。價格已大幅偏離均值，同時成交量創 lookback 根新高，
代表追價資金在此集中出清，之後缺乏後續買（賣）盤而回落。
與 liq_reversal 的差異：這裡看的是「累積偏離 + 量能創新高」，而不是單根爆量急跌。
進場：成交量 ≥ 過去 lookback 根最大量，且 (收盤 − 24 根均價) / ATR24 ≥ k 做空、≤ -k 做多；持有 HOLD 根。
"""
import pandas as pd

from strategies._util import combine, hold_after

HOLD = 12


def signal(df: pd.DataFrame, lookback: int = 168, k: float = 2.5) -> pd.Series:
    c, h, l = df["close"], df["high"], df["low"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(24).mean()
    ext = (c - c.rolling(24).mean()) / atr
    climax = df["volume"] >= df["volume"].rolling(lookback).max().shift(1)
    long = hold_after(climax & (ext <= -k), HOLD)
    short = hold_after(climax & (ext >= k), HOLD)
    return combine(long, short)
