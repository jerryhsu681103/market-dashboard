"""時段季節性（只做多）。

賺誰的錢：時段間資金流的結構性差異（例如美股開盤、亞洲時段散戶、歐美機構調倉）
造成的固定時段漂移；在漂移時段持有、其餘時間空倉。
進場：只在 UTC 時段 [start, start+length) 的 K 線持倉做多。
（只看時間戳，不使用任何價格資訊。）
"""
import numpy as np
import pandas as pd


def signal(df: pd.DataFrame, start: int = 13, length: int = 4) -> pd.Series:
    # 第 t 根的信號決定第 t+1 根的持倉，所以看下一根的開盤小時
    next_hour = (df.index.hour + 1) % 24
    in_window = ((next_hour - start) % 24) < length
    return pd.Series(np.where(in_window, 1, 0), index=df.index)
