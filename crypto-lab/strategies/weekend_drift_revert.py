"""週末波動回歸。

賺誰的錢：週末流動性差時推動價格的一方。週末機構與做市商較少，
小資金就能推動價格；週一流動性回來後，價格傾向修正週末的過度移動。
進場：週日最後一根 K 線收盤時，計算週末報酬（週六 00:00 開盤 → 週日最後一根收盤）。
     超過 +threshold 做空、低於 -threshold 做多，週一 00:00 開盤成交，持有 hold 根。
"""
import pandas as pd

from strategies._util import bars, combine, hold_after, step_hours

WEEKEND_H = 48  # 週六 00:00 → 週一 00:00


def signal(df: pd.DataFrame, threshold: float = 0.03, hold: int = 24) -> pd.Series:
    idx = df.index
    step = pd.Timedelta(hours=step_hours(df))
    n = bars(df, WEEKEND_H)
    nxt = idx + step  # 觸發於週日最後一根：下一根是週一 00:00
    trigger = pd.Series((nxt.day_of_week == 0) & (nxt.hour == 0), index=idx)
    start_open = df["open"].shift(n - 1)
    # 確認回推 47 根正好是週六 00:00（資料若有缺口則不觸發）
    shifted_time = pd.Series(idx, index=idx).shift(n - 1)
    aligned = shifted_time.eq(pd.Series(idx - (n - 1) * step, index=idx))
    wret = df["close"] / start_open - 1
    valid = trigger & aligned
    long = hold_after(valid & (wret <= -threshold), hold)
    short = hold_after(valid & (wret >= threshold), hold)
    return combine(long, short)
