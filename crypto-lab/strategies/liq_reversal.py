"""清算瀑布後的反彈。

賺誰的錢：被強制平倉的一方。連環爆倉時賣壓（或買壓）來自強平引擎而非資訊，
價格被推過頭；在恐慌中提供流動性的人收取溢價。
進場：單根報酬超過過去 30 天波動的 k 倍，且成交量 ≥ 過去 30 天中位數的 3 倍
     → 大跌做多、大漲做空；持有 hold 根。
"""
import numpy as np
import pandas as pd

from strategies._util import combine, hold_after

WINDOW = 24 * 30


def signal(df: pd.DataFrame, k: float = 4.0, hold: int = 12) -> pd.Series:
    r = np.log(df["close"]).diff()
    sd = r.rolling(WINDOW).std().shift(1)
    vmed = df["volume"].rolling(WINDOW).median().shift(1)
    spike = df["volume"] >= 3 * vmed
    long = hold_after((r <= -k * sd) & spike, hold)
    short = hold_after((r >= k * sd) & spike, hold)
    return combine(long, short)
