"""現貨與永續主動買賣流分歧（資料：現貨 K 線）。

賺誰的錢：永續上的槓桿投機方。現貨主動淨買、永續主動淨賣時，現貨的真實需求在吸收槓桿賣壓，
空方後續被迫回補；反之，現貨淨賣、永續淨買代表槓桿多頭在接現貨的賣壓。
信號：過去 window 根的主動淨買比例 = (2×主動買成交額 − 成交額) / 成交額，現貨減永續，
     相對過去 30 天的 z-score ≥ threshold 做多、≤ -threshold 做空；持有 window 根。
需要欄位：spot_quote_volume、spot_taker_buy_quote_volume（attach_features(..., ["spot"])）。
"""
import pandas as pd

from strategies._util import bars, combine, hold_after

Z_H = 30 * 24


def _imb(buy: pd.Series, total: pd.Series, n: int) -> pd.Series:
    t = total.rolling(n).sum()
    return (2 * buy.rolling(n).sum() - t) / t.where(t > 0)


def signal(df: pd.DataFrame, threshold: float = 1.0, window: int = 6) -> pd.Series:
    d = (_imb(df["spot_taker_buy_quote_volume"], df["spot_quote_volume"], window)
         - _imb(df["taker_buy_quote_volume"], df["quote_volume"], window))
    nz = bars(df, Z_H)
    mp = int(nz * 0.9)
    z = (d - d.rolling(nz, min_periods=mp).mean().shift(1)) / d.rolling(nz, min_periods=mp).std().shift(1)
    return combine(hold_after(z >= threshold, window), hold_after(z <= -threshold, window))
