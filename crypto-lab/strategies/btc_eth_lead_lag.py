"""BTC 領先 ETH（交易 ETHUSDT）。

賺誰的錢：ETH 上反應較慢的參與者。BTC 是加密市場的定價中心，大型資訊先反映在 BTC，
ETH 的價格發現稍有落後；BTC 已大幅移動而 ETH 尚未跟上時，買入（賣出）落後的 ETH。
需要 df 有 BTCUSDT_close 欄位（與 ETH 同一時間索引）。
進場：過去 3 小時 BTC 報酬 − ETH 報酬 ≥ threshold 做多 ETH、≤ -threshold 做空 ETH；持有 hold 根。
"""
import numpy as np
import pandas as pd

from strategies._util import bars, combine, hold_after

LOOKBACK_H = 3  # 小時


def signal(df: pd.DataFrame, threshold: float = 0.01, hold: int = 6) -> pd.Series:
    lb = bars(df, LOOKBACK_H)
    btc = np.log(df["BTCUSDT_close"]).diff(lb)
    eth = np.log(df["close"]).diff(lb)
    gap = btc - eth
    long = hold_after(gap >= threshold, hold)
    short = hold_after(gap <= -threshold, hold)
    return combine(long, short)
