"""週末波動回歸。

賺誰的錢：週末流動性差時推動價格的一方。週末機構與做市商較少，
小資金就能推動價格；週一流動性回來後，價格傾向修正週末的過度移動。
進場：週日 23:00 UTC 那根收盤時，計算週末報酬（週六 00:00 開盤 → 週日 23:00 收盤）。
     超過 +threshold 做空、低於 -threshold 做多，週一 00:00 開盤成交，持有 hold 根。
"""
import pandas as pd

from strategies._util import combine, hold_after

WEEKEND_BARS = 48


def signal(df: pd.DataFrame, threshold: float = 0.03, hold: int = 24) -> pd.Series:
    idx = df.index
    trigger = pd.Series((idx.day_of_week == 6) & (idx.hour == 23), index=idx)
    start_open = df["open"].shift(WEEKEND_BARS - 1)
    # 確認回推 47 根正好是週六 00:00（資料若有缺口則不觸發）
    shifted_time = pd.Series(idx, index=idx).shift(WEEKEND_BARS - 1)
    aligned = shifted_time.eq(pd.Series(idx - pd.Timedelta(hours=WEEKEND_BARS - 1), index=idx))
    wret = df["close"] / start_open - 1
    valid = trigger & aligned
    long = hold_after(valid & (wret <= -threshold), hold)
    short = hold_after(valid & (wret >= threshold), hold)
    return combine(long, short)
