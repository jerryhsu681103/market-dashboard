"""平倉驅動行情的反向（資料：未平倉量）。

賺誰的錢：追逐「空頭回補上漲／多頭平倉下跌」的晚到資金。價格移動伴隨未平倉量明顯下降，
代表行情由既有部位平倉推動，而非新資金進場；平倉推力耗盡後價格回吐。
信號：過去 24 小時價格上漲 且 同期未平倉量對數變化的 z-score（相對過去 30 天）≤ -k → 做空；
     價格下跌 且 z ≤ -k → 做多。持有 hold 根。
需要欄位：oi（attach_features(..., ["metrics"])）。
"""
import numpy as np
import pandas as pd

from strategies._util import bars, combine, hold_after

CHG_H = 24
Z_H = 30 * 24


def signal(df: pd.DataFrame, k: float = 1.0, hold: int = 6) -> pd.Series:
    nc, nz = bars(df, CHG_H), bars(df, Z_H)
    dp = np.log(df["close"]).diff(nc)
    chg = np.log(df["oi"]).diff(nc)
    mp = int(nz * 0.9)
    z = (chg - chg.rolling(nz, min_periods=mp).mean().shift(1)) / chg.rolling(nz, min_periods=mp).std().shift(1)
    unwinding = z <= -k
    return combine(hold_after(unwinding & (dp < 0), hold), hold_after(unwinding & (dp > 0), hold))
