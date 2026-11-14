"""資金費率極端值反向。

賺誰的錢：高槓桿擁擠的一方。費率極高代表多頭擁擠、付高額費率持倉，
後續容易因去槓桿回落；反向持倉同時收取資金費。
進場：最新已結算費率 ≥ threshold → 做空；≤ -threshold → 做多；持有 hold 根。
需要 df 有 funding_rate 欄位（lab.data.attach_funding）。
"""
import pandas as pd

from strategies._util import combine, hold_after


def signal(df: pd.DataFrame, threshold: float = 0.0005, hold: int = 24) -> pd.Series:
    f = df["funding_rate"]
    # 只在費率剛更新的那根觸發，避免同一筆費率重複觸發
    fresh = f.ne(f.shift(1))
    short = hold_after(fresh & (f >= threshold), hold)
    long = hold_after(fresh & (f <= -threshold), hold)
    return combine(long, short)
