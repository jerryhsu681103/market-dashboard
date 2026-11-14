"""EMA 金叉做多、死叉做空（煙霧測試用，不是研究點子）。"""
import numpy as np
import pandas as pd


def signal(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    c = df["close"]
    ema_f = c.ewm(span=fast, adjust=False).mean()
    ema_s = c.ewm(span=slow, adjust=False).mean()
    s = pd.Series(np.where(ema_f > ema_s, 1, -1), index=df.index)
    s.iloc[:slow] = 0  # 暖機期不交易
    return s
