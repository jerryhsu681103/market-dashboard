"""現貨主導 vs 槓桿主導的行情（新資料：現貨 K 線）。

賺誰的錢：用槓桿追價的投機資金。現貨買盤（真實需求、需要全額付款）推動的行情較能延續；
主要由永續合約推動的行情是槓桿堆疊，容易在去槓桿時回吐。
信號：過去 window 根的價格方向，搭配同期間現貨成交額占比（現貨 /（現貨＋永續））相對過去 30 天的 z-score：
     z ≥ threshold（現貨主導）→ 順著價格方向；z ≤ -threshold（槓桿主導）→ 逆著價格方向。持有 window 根。
需要欄位：spot_quote_volume、quote_volume（lab.data.attach_features(..., ["spot"])）。
"""
import numpy as np
import pandas as pd

from strategies._util import bars, combine, hold_after

Z_H = 30 * 24


def signal(df: pd.DataFrame, threshold: float = 1.0, window: int = 6) -> pd.Series:
    sq = df["spot_quote_volume"].rolling(window).sum()
    pq = df["quote_volume"].rolling(window).sum()
    share = sq / (sq + pq)
    nz = bars(df, Z_H)
    z = (share - share.rolling(nz).mean().shift(1)) / share.rolling(nz).std().shift(1)
    direction = np.sign(np.log(df["close"]).diff(window))
    follow = z >= threshold
    fade = z <= -threshold
    want = pd.Series(np.where(follow, direction, np.where(fade, -direction, 0)), index=df.index).fillna(0)
    long = hold_after(want > 0, window)
    short = hold_after(want < 0, window)
    return combine(long, short)
