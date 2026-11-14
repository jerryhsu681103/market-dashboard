"""吸收後突破（為 4 小時線設計）。

賺誰的錢：掛單被吃光的一方。成交量明顯放大但 K 線振幅不大，代表有大額掛單在吸收主動單；
之後價格離開該根 K 線的高低點，表示其中一方的掛單已被消化完，被動方需要追價或停損。
吸收 K 線：成交量 ≥ vol_mult × 過去 30 天成交量中位數，且振幅（高−低）/收盤 ≤ 過去 30 天振幅中位數。
信號：吸收 K 線出現後 24 小時內，收盤 > 該 K 線最高價 → 做多；收盤 < 該 K 線最低價 → 做空。持有 hold 根。
失效情境：高量窄幅之後方向隨機；吸收方其實是持續出貨的大戶。
與 volume_climax 的差異：該點子要求價格已大幅偏離並反向；本點子要求振幅小、順突破方向。
"""
import pandas as pd

from strategies._util import bars, combine, hold_after

MED_H = 30 * 24
BREAK_H = 24


def signal(df: pd.DataFrame, vol_mult: float = 2.0, hold: int = 6) -> pd.Series:
    nm, nb = bars(df, MED_H), bars(df, BREAK_H)
    mp = int(nm * 0.9)
    rng = (df["high"] - df["low"]) / df["close"]
    vol_med = df["volume"].rolling(nm, min_periods=mp).median().shift(1)
    rng_med = rng.rolling(nm, min_periods=mp).median().shift(1)
    absorb = (df["volume"] >= vol_mult * vol_med) & (rng <= rng_med)
    # 最近一根吸收 K 線的高低點，只在其後 nb 根內有效（不含吸收 K 線當根）
    ref_hi = df["high"].where(absorb).ffill(limit=nb).shift(1)
    ref_lo = df["low"].where(absorb).ffill(limit=nb).shift(1)
    long = hold_after(df["close"] > ref_hi, hold)
    short = hold_after(df["close"] < ref_lo, hold)
    return combine(long, short)
