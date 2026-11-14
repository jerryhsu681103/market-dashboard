"""被套部位的擠壓（新資料：未平倉量、多空比）。

賺誰的錢：逆勢加碼、被套住的槓桿方。價格上漲、未平倉量增加，但帳戶多空比下降
（新增部位多為散戶開空）→ 空方被套，後續回補推動上漲；下跌時對稱（散戶逆勢接多被套）。
信號：過去 24 小時的價格變化、未平倉量變化 z-score、帳戶多空比變化 z-score（z 皆相對過去 30 天）：
     價格上漲 且 ΔOI z ≥ k 且 Δ多空比 z ≤ -k → 做多；價格下跌 且 ΔOI z ≥ k 且 Δ多空比 z ≥ k → 做空。持有 hold 根。
需要欄位：oi、global_ls_account（attach_features(..., ["metrics"])）。
"""
import numpy as np
import pandas as pd

from strategies._util import bars, combine, hold_after

CHG_H = 24
Z_H = 30 * 24


def _z(x: pd.Series, n: int) -> pd.Series:
    return (x - x.rolling(n, min_periods=int(n * 0.9)).mean().shift(1)) / x.rolling(n, min_periods=int(n * 0.9)).std().shift(1)


def signal(df: pd.DataFrame, k: float = 1.0, hold: int = 6) -> pd.Series:
    nc, nz = bars(df, CHG_H), bars(df, Z_H)
    dp = np.log(df["close"]).diff(nc)
    doi = _z(np.log(df["oi"]).diff(nc), nz)
    dls = _z(np.log(df["global_ls_account"]).diff(nc), nz)
    long = hold_after((dp > 0) & (doi >= k) & (dls <= -k), hold)
    short = hold_after((dp < 0) & (doi >= k) & (dls >= k), hold)
    return combine(long, short)
