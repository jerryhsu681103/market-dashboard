"""多日深跌後的承接（只做多，為 4 小時線設計）。

賺誰的錢：多日去槓桿過程中被迫賣出的一方（追繳保證金、基金贖回、停損）。
被迫賣壓結束、價格開始回升時，承接者收取流動性溢價。
與已淘汰 liq_reversal 的差異：liq_reversal 在「單根 K 線」爆量急跌當下逆勢接刀、持有數小時；
本點子看「過去 30 天高點的累積跌幅」，並等價格出現回升確認（收盤高於前一根高點）才進場，持有數天。
進場：收盤距 30 天最高收盤跌幅 ≥ drop，且收盤 > 前一根最高價 → 做多，持有 hold 根。
"""
import pandas as pd

from strategies._util import bars, hold_after

PEAK_H = 30 * 24


def signal(df: pd.DataFrame, drop: float = 0.2, hold: int = 12) -> pd.Series:
    c = df["close"]
    dd = c / c.rolling(bars(df, PEAK_H)).max() - 1
    turn_up = c > df["high"].shift(1)
    long = hold_after((dd <= -drop) & turn_up, hold)
    return long.astype(int)
