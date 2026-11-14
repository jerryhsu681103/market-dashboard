"""假突破反轉（為 4 小時線設計）。

賺誰的錢：追突破進場、隨即被套的短線交易者，以及停損掛在區間外側被掃到的一方。
價格刺穿過去 channel 根的最高（最低）價，但同一根收盤又收回區間內，代表突破沒有後續買（賣）盤，
追價者與被掃停損者的部位需要反向了結，推動價格回到區間內。
信號：最高價 > 前 channel 根最高價 且 收盤 < 該最高價 → 做空；
     最低價 < 前 channel 根最低價 且 收盤 > 該最低價 → 做多。持有 hold 根。同時成立時空倉。
失效情境：真突破（趨勢啟動）前的第一次試探。
"""
import pandas as pd

from strategies._util import combine, hold_after


def signal(df: pd.DataFrame, channel: int = 18, hold: int = 3) -> pd.Series:
    hi = df["high"].rolling(channel).max().shift(1)
    lo = df["low"].rolling(channel).min().shift(1)
    failed_up = (df["high"] > hi) & (df["close"] < hi)
    failed_dn = (df["low"] < lo) & (df["close"] > lo)
    return combine(hold_after(failed_dn, hold), hold_after(failed_up, hold))
