"""大戶與散戶部位分歧（新資料：多空比）。

賺誰的錢：散戶。Binance 公布兩種多空比：大戶持倉多空比（依部位大小，top trader position）
與全市場帳戶多空比（依帳戶數，以散戶為主）。大戶相對散戶更偏多時，跟著大戶；更偏空時，反向。
信號：log(大戶持倉多空比) − log(全市場帳戶多空比) 的 z-score（相對過去 30 天）。
     z ≥ threshold 做多、≤ -threshold 做空，持有 hold 根。
需要欄位：top_ls_position、global_ls_account（attach_features(..., ["metrics"])）。2020-09 之前沒有資料，不交易。
"""
import numpy as np
import pandas as pd

from strategies._util import bars, combine, hold_after

Z_H = 30 * 24


def signal(df: pd.DataFrame, threshold: float = 1.5, hold: int = 6) -> pd.Series:
    d = np.log(df["top_ls_position"]) - np.log(df["global_ls_account"])
    nz = bars(df, Z_H)
    z = (d - d.rolling(nz, min_periods=int(nz * 0.9)).mean().shift(1)) / d.rolling(nz, min_periods=int(nz * 0.9)).std().shift(1)
    long = hold_after(z >= threshold, hold)
    short = hold_after(z <= -threshold, hold)
    return combine(long, short)
