"""ETH/BTC 比值回歸（市場中性價差：做多一腳、做空另一腳）。

賺誰的錢：在兩幣之間追逐相對強弱的資金。ETH 與 BTC 由相同的大盤資金驅動，
短中期的相對價格偏離多半來自資金輪動與槓桿擁擠，而非基本面改變，偏離後傾向回歸。
市場中性：同時做多 ETH、做空 BTC（或相反），抵銷大盤漲跌造成的回撤。
失效情境：敘事驅動的長期相對趨勢（例如 2021 年 ETH 大幅跑贏、2022 年合併前後），偏離持續擴大。
需要 df 有 ETHUSDT_close、BTCUSDT_close。
進場：log(ETH/BTC) 相對過去 lookback 根的 z-score ≤ -entry_z → +1（做多 ETH、做空 BTC）；≥ entry_z → -1。
出場：z-score 穿越 0，或持有滿 lookback 根（偏離一個窗口都未回歸，視為結構改變）。
"""
import numpy as np
import pandas as pd


def signal(df: pd.DataFrame, lookback: int = 90, entry_z: float = 2.0) -> pd.Series:
    ratio = np.log(df["ETHUSDT_close"]) - np.log(df["BTCUSDT_close"])
    z = ((ratio - ratio.rolling(lookback).mean()) / ratio.rolling(lookback).std()).to_numpy()
    out = np.zeros(len(z), dtype=np.int8)
    pos, held = 0, 0
    for t in range(len(z)):
        if np.isnan(z[t]):
            continue
        if pos != 0:
            held += 1
            if held >= lookback or (pos == 1 and z[t] >= 0) or (pos == -1 and z[t] <= 0):
                pos = 0
        if pos == 0:
            if z[t] <= -entry_z:
                pos, held = 1, 0
            elif z[t] >= entry_z:
                pos, held = -1, 0
        out[t] = pos
    return pd.Series(out, index=df.index)
