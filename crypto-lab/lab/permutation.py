"""K 線置換：產生「假行情」。

做法（對數價格空間）：
    - 把每根 K 線拆成「跳空」gap = log(open[t]) - log(close[t-1])
      與「K 線形狀」(high-open, low-open, close-open)。
    - 形狀與跳空各自隨機打亂後重新拼回。
    - 起點價格與終點價格不變；每根 K 線的開高低收相對關係不變；成交量跟著形狀一起移動。
    - 時間索引不動：週末、時段等時間標籤留在原本的時間上（虛無假設 = 漲跌與時間無關）。
    - 資金費率不打亂（含 df 裡 funding_ 開頭的欄位），留在原本的結算時間上。點子若以資金費率當信號，需在日誌註明「不確定」。

start_index：只打亂這根之後的 K 線（第 4 關前推置換用：保留第一個訓練窗口）。
多幣種點子用 permute_many，所有幣種套用同一組打亂順序，保留幣種間的相關性。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE = ["open", "high", "low", "close"]


def _perm_indices(n_bars: int, start_index: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    k = n_bars - start_index - 1  # 被打亂的 K 線數
    if k < 2:
        raise ValueError("可打亂的 K 線太少")
    return rng.permutation(k), rng.permutation(k)


def _apply(df: pd.DataFrame, start_index: int, p_shape: np.ndarray, p_gap: np.ndarray) -> pd.DataFrame:
    lo, lh, ll, lc = (np.log(df[c].to_numpy(float)) for c in PRICE)
    s = start_index
    gap = lo[s + 1:] - lc[s:-1]
    h_rel, l_rel, c_rel = (x[s + 1:] - lo[s + 1:] for x in (lh, ll, lc))

    gap, h_rel, l_rel, c_rel = gap[p_gap], h_rel[p_shape], l_rel[p_shape], c_rel[p_shape]
    new_lc = lc[s] + np.cumsum(gap + c_rel)
    new_lo = np.concatenate([[lc[s]], new_lc[:-1]]) + gap

    out = df.copy()
    out.iloc[s + 1:, out.columns.get_loc("open")] = np.exp(new_lo)
    out.iloc[s + 1:, out.columns.get_loc("high")] = np.exp(new_lo + h_rel)
    out.iloc[s + 1:, out.columns.get_loc("low")] = np.exp(new_lo + l_rel)
    out.iloc[s + 1:, out.columns.get_loc("close")] = np.exp(new_lc)
    # 成交量等 K 線屬性跟著形狀移動；funding_* 欄位留在原本的時間上，不打亂
    others = [c for c in df.columns if c not in PRICE and not c.startswith("funding")]
    for c in others:
        col = out.columns.get_loc(c)
        out.iloc[s + 1:, col] = df[c].to_numpy()[s + 1:][p_shape]
    return out


def permute_bars(df: pd.DataFrame, seed: int, start_index: int = 0) -> pd.DataFrame:
    p_shape, p_gap = _perm_indices(len(df), start_index, seed)
    return _apply(df, start_index, p_shape, p_gap)


def permute_many(dfs: dict[str, pd.DataFrame], seed: int, start_index: int = 0) -> dict[str, pd.DataFrame]:
    idx = next(iter(dfs.values())).index
    for k, d in dfs.items():
        if not d.index.equals(idx):
            raise ValueError(f"{k} 的時間索引與其他幣種不一致，請先對齊")
    p_shape, p_gap = _perm_indices(len(idx), start_index, seed)
    return {k: _apply(d, start_index, p_shape, p_gap) for k, d in dfs.items()}
