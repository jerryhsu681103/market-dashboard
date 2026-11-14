"""低波動環境的短線均值回歸。

賺誰的錢：平靜市場中急著用市價單進出的流動性需求者。波動低時沒有資訊衝擊，
短線偏離多半是流動性造成的，價格傾向回到原位。
與已淘汰的 liq_reversal 的差異：liq_reversal 只在「極端爆量急跌」時進場；
這裡反而排除高波動環境，只在平靜時期做小幅偏離的回歸。
進場：24 小時已實現波動在過去 30 天的百分位 ≤ vol_pct，且 6 小時報酬的 z-score ≤ -threshold 做多、≥ threshold 做空；持有 6 小時。
"""
import numpy as np
import pandas as pd

from strategies._util import bars, combine, hold_after

HOLD_H = 6  # 小時
REGIME_H = 24 * 30
VOL_H = 24
MOVE_H = 6


def signal(df: pd.DataFrame, threshold: float = 2.0, vol_pct: float = 0.4) -> pd.Series:
    hold, regime, n_vol, n_move = (bars(df, h) for h in (HOLD_H, REGIME_H, VOL_H, MOVE_H))
    r = np.log(df["close"]).diff()
    vol24 = r.rolling(n_vol).std()
    calm = vol24.rolling(regime).rank(pct=True) <= vol_pct
    sd = r.rolling(regime).std().shift(1)
    z6 = r.rolling(n_move).sum() / (sd * np.sqrt(n_move))
    long = hold_after(calm & (z6 <= -threshold), hold)
    short = hold_after(calm & (z6 >= threshold), hold)
    return combine(long, short)
