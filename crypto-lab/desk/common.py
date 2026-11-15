"""隔夜研究組（desk）共用：幣池、資料視窗、指標。

- 所有回測數字仍由 lab/engine.py 產生；這裡只負責載入資料與計算（只用過去資料的）指標。
- 評估視窗：近 4 年（EVAL_START 起）。EVAL_START 之前的 1 年只用來讓指標暖機，不計入任何成績。
- 幣池以 2022-11 當時的主流幣為準（憑記憶選定，屬估計），刻意包含中途下市的 MATICUSDT，
  以免只挑「現在還活著的幣」造成倖存者偏差。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab.data import DATA, attach_funding, resample

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
           "MATICUSDT", "DOTUSDT", "LTCUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT"]
ALTS = [s for s in SYMBOLS if s != "BTCUSDT"]
EVAL_START = pd.Timestamp("2022-11-15", tz="UTC")
WARM_START = pd.Timestamp("2021-11-15", tz="UTC")
XS_HOURS = (72, 168, 336, 720)  # 橫斷面排名用的回看長度（小時）


def step_hours(df: pd.DataFrame) -> float:
    d = pd.Series(df.index[: min(len(df), 101)]).diff().dropna()
    return d.median().total_seconds() / 3600


def bars(df: pd.DataFrame, hours: float) -> int:
    return max(1, int(round(hours / step_hours(df))))


def load_universe(timeframe: str = "4h", end: pd.Timestamp | None = None) -> dict:
    """回傳 {symbol: (df, funding)}。df 已併入 funding_rate、btc_* 與橫斷面排名欄位。

    併入的欄位在第 t 列都只用到第 t 根收盤（含）之前的資訊。
    """
    raw = {}
    for s in SYMBOLS:
        df = resample(pd.read_parquet(DATA / f"{s}_1h.parquet"), timeframe)
        df = df[df.index >= WARM_START]
        if end is not None:
            df = df[df.index < end]
        funding = pd.read_parquet(DATA / f"{s}_funding.parquet")["funding_rate"]
        raw[s] = (df, funding)

    btc = raw["BTCUSDT"][0]
    close = pd.DataFrame({s: d["close"] for s, (d, _) in raw.items()})
    ranks = {}
    for h in XS_HOURS:
        n = bars(btc, h)
        ret = close / close.shift(n) - 1.0
        r = ret.rank(axis=1, pct=True)
        r[ret.notna().sum(axis=1) < 6] = np.nan  # 可排名的幣太少時不排名
        ranks[h] = r

    out = {}
    for s, (df, funding) in raw.items():
        d = attach_funding(df, funding)
        for c in ("open", "high", "low", "close", "volume"):
            d[f"btc_{c}"] = btc[c].reindex(d.index)
        for h in XS_HOURS:
            d[f"xs_rank_{h}"] = ranks[h][s].reindex(d.index)
        out[s] = (d, funding)
    return out


# ---------------------------------------------------------------- 指標（全部只用過去資料）

def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
                     axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).rolling(n, min_periods=n).mean()


def rsi(x: pd.Series, n: int = 14) -> pd.Series:
    d = x.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def stateful(index, long_entry, short_entry, long_exit, short_exit, close, atr_v,
             stop_k: float = 0.0, trail: bool = True, max_hold: int = 0) -> pd.Series:
    """事件進場、規則出場的持倉狀態機（逐根往前走，只看當根及之前）。

    止損：以「收盤價」判斷——多單收盤跌破 參考價 − stop_k × 進場時 ATR 即出場（下一根開盤成交）。
    trail=True 時參考價為進場後最高（最低）收盤價，否則為進場當根收盤價。
    出場的那一根不會同時反手進場。
    """
    n = len(index)
    le, se = np.asarray(long_entry, bool), np.asarray(short_entry, bool)
    lx, sx = np.asarray(long_exit, bool), np.asarray(short_exit, bool)
    c, a = np.asarray(close, float), np.asarray(atr_v, float)
    out = np.zeros(n, np.int8)
    pos, ref, a0, held = 0, 0.0, 0.0, 0
    for t in range(n):
        if pos != 0:
            held += 1
            if trail:
                ref = max(ref, c[t]) if pos > 0 else min(ref, c[t])
            stop = stop_k > 0 and ((pos > 0 and c[t] < ref - stop_k * a0) or
                                   (pos < 0 and c[t] > ref + stop_k * a0))
            sig = (pos > 0 and lx[t]) or (pos < 0 and sx[t])
            if stop or sig or (max_hold and held >= max_hold):
                pos = 0
        elif np.isfinite(a[t]) and a[t] > 0 and np.isfinite(c[t]):
            if le[t] and not se[t]:
                pos, ref, a0, held = 1, c[t], a[t], 0
            elif se[t] and not le[t]:
                pos, ref, a0, held = -1, c[t], a[t], 0
        out[t] = pos
    return pd.Series(out, index=index)
