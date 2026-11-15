"""夜巡候選清單（2026-11-15 晚）。理由、付錢的人、適用狀態、參數網格都在看到任何結果之前寫定。

每個訊號函式：signal(df, **params) -> Series(-1/0/1)，第 t 根的訊號只用第 t 根收盤（含）之前的資料，
由 lab/engine.py 在第 t+1 根開盤成交。以小時表示的長度（*_h）會依 K 線週期換算成根數。

與 FACTORY_RULES 的關係：其中數條與工廠已淘汰的 BTC 單幣點子同源（overlap 欄標示）。
今晚是使用者指定的新題目（12 幣等權組合），視為新的測試並照實計入測試次數，不是替死掉的點子翻案。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import adx, atr, bars, ema, rsi, sma, stateful

Z = lambda df: pd.Series(0, index=df.index)  # noqa: E731


def _ls(long_flag, short_flag) -> pd.Series:
    return long_flag.fillna(False).astype(int) - short_flag.fillna(False).astype(int)


def _hold(event: pd.Series, n: int) -> pd.Series:
    return event.fillna(False).astype(float).rolling(n, min_periods=1).max().astype(bool)


# ---------------------------------------------------------------- A. 趨勢／動量

def tsmom_ls(df, look_h, band):
    r = df["close"] / df["close"].shift(bars(df, look_h)) - 1
    return _ls(r > band, r < -band)


def tsmom_long_btcfilter(df, look_h, btc_ma_h):
    r = df["close"] / df["close"].shift(bars(df, look_h)) - 1
    ok = df["btc_close"] > sma(df["btc_close"], bars(df, btc_ma_h))
    return _ls((r > 0) & ok, r < -9)


def donchian_atr(df, entry_h, stop_k):
    n = bars(df, entry_h)
    c = df["close"]
    hi, lo = df["high"].rolling(n).max().shift(1), df["low"].rolling(n).min().shift(1)
    xhi, xlo = df["high"].rolling(n // 2).max().shift(1), df["low"].rolling(n // 2).min().shift(1)
    return stateful(df.index, c > hi, c < lo, c < xlo, c > xhi, c, atr(df, 14), stop_k, True)


def ema_adx(df, slow_h, adx_th):
    slow = bars(df, slow_h)
    f, s, a = ema(df["close"], max(2, slow // 4)), ema(df["close"], slow), adx(df, 14)
    return _ls((f > s) & (a > adx_th), (f < s) & (a > adx_th))


def squeeze_breakout(df, squeeze_pct, channel_h):
    c = df["close"]
    bw = c.rolling(20).std() / c.rolling(20).mean()
    pct = bw.rolling(bars(df, 720)).rank(pct=True)
    squeezed = (pct < squeeze_pct).astype(float).rolling(6, min_periods=1).max() > 0
    n = bars(df, channel_h)
    hi, lo = df["high"].rolling(n).max().shift(1), df["low"].rolling(n).min().shift(1)
    mid = (hi + lo) / 2
    return stateful(df.index, squeezed & (c > hi), squeezed & (c < lo), c < mid, c > mid,
                    c, atr(df, 14), 3.0, True)


def trend_pullback(df, trend_h, rsi_th):
    c = df["close"]
    e = ema(c, bars(df, trend_h))
    r = rsi(c, 14)
    up, dn = (c > e) & (e > e.shift(6)), (c < e) & (e < e.shift(6))
    return stateful(df.index, up & (r < rsi_th), dn & (r > 100 - rsi_th), r > 55, r < 45,
                    c, atr(df, 14), 3.0, False, 30)


def vwap_trend(df, vwap_h, band):
    n = bars(df, vwap_h)
    v = df["quote_volume"].rolling(n).sum() / df["volume"].rolling(n).sum()
    return _ls(df["close"] > v * (1 + band), df["close"] < v * (1 - band))


def volume_breakout(df, vol_k, hold_h):
    c = df["close"]
    n = bars(df, 80)
    hi, lo = df["high"].rolling(n).max().shift(1), df["low"].rolling(n).min().shift(1)
    big = df["volume"] > vol_k * df["volume"].rolling(bars(df, 720)).mean().shift(1)
    h = bars(df, hold_h)
    return _ls(_hold(big & (c > hi), h), _hold(big & (c < lo), h))


def near_high_momo(df, look_h, prox):
    top = df["high"].rolling(bars(df, look_h)).max()
    return _ls(df["close"] >= (1 - prox) * top, df["close"] < 0)


def dual_trend_long(df, slow_h, fast_ratio):
    slow = bars(df, slow_h)
    s, f = sma(df["close"], slow), sma(df["close"], max(2, int(slow * fast_ratio)))
    return _ls((df["close"] > s) & (f > s), df["close"] < 0)


def vol_expansion_momo(df, look_h, volpct):
    c = df["close"]
    r = c / c.shift(bars(df, look_h)) - 1
    rv = c.pct_change().rolling(bars(df, 120)).std()
    hot = rv.rolling(bars(df, 2160)).rank(pct=True) > volpct
    return _ls(hot & (r > 0), hot & (r < 0))


# ---------------------------------------------------------------- B. 過激後回歸

def overshoot_revert(df, k, hold_h):
    r = df["close"] / df["open"] - 1
    sd = r.rolling(bars(df, 720)).std().shift(1)
    h = bars(df, hold_h)
    return _ls(_hold(r < -k * sd, h), _hold(r > k * sd, h))


def rsi2_dip(df, rsi_th, trend_h):
    c = df["close"]
    up = c > sma(c, bars(df, trend_h))
    r2 = rsi(c, 2)
    return stateful(df.index, up & (r2 < rsi_th), Z(df) > 0, c > sma(c, 5), Z(df) > 0,
                    c, atr(df, 14), 4.0, False, 12)


def bb_revert_range(df, z, adx_max):
    c = df["close"]
    zs = (c - c.rolling(20).mean()) / c.rolling(20).std()
    quiet = adx(df, 14) < adx_max
    return stateful(df.index, quiet & (zs < -z), quiet & (zs > z), zs > 0, zs < 0,
                    c, atr(df, 14), 3.0, False, 30)


def wick_reversal(df, k, hold_h):
    a = atr(df, 14).shift(1)
    body_lo, body_hi = df[["open", "close"]].min(axis=1), df[["open", "close"]].max(axis=1)
    big = df["volume"] > 2 * df["volume"].rolling(bars(df, 720)).mean().shift(1)
    h = bars(df, hold_h)
    return _ls(_hold(big & (body_lo - df["low"] > k * a), h),
               _hold(big & (df["high"] - body_hi > k * a), h))


def volume_climax_fade(df, look_h, k):
    c = df["close"]
    n = bars(df, look_h)
    r = c / c.shift(n) - 1
    rz = r / r.rolling(bars(df, 2160)).std().shift(1)
    big = df["volume"] > k * df["volume"].rolling(bars(df, 720)).mean().shift(1)
    h = bars(df, 24)
    return _ls(_hold(big & (rz < -1.5), h), _hold(big & (rz > 1.5), h))


# ---------------------------------------------------------------- C. 資金費

def _fund_avg(df, hours=24):
    return df["funding_rate"].rolling(bars(df, hours)).mean()


def funding_extreme_fade(df, z, hold_h):
    f = _fund_avg(df)
    n = bars(df, 2160)
    fz = (f - f.rolling(n).mean()) / f.rolling(n).std()
    h = bars(df, hold_h)
    return _ls(_hold(fz < -z, h), _hold(fz > z, h))


def funding_carry(df, th, trend_h):
    f = _fund_avg(df)
    r = df["close"] / df["close"].shift(bars(df, trend_h)) - 1
    return _ls((f < -th) & (r >= 0), (f > th) & (r <= 0))


def momo_uncrowded(df, look_h, fund_cap):
    f = _fund_avg(df)
    r = df["close"] / df["close"].shift(bars(df, look_h)) - 1
    return _ls((r > 0) & (f < fund_cap), (r < 0) & (f > -fund_cap))


# ---------------------------------------------------------------- D. 跨幣：BTC 先動、強弱輪動

def btc_lead_alt(df, th, hold_h):
    rb = df["btc_close"] / df["btc_open"] - 1
    ra = df["close"] / df["open"] - 1
    h = bars(df, hold_h)
    return _ls(_hold((rb > th) & (ra < rb / 2), h), _hold((rb < -th) & (ra > rb / 2), h))


def btc_breakout_alt(df, entry_h, stop_k):
    n = bars(df, entry_h)
    b = df["btc_close"]
    hi = df["btc_high"].rolling(n).max().shift(1)
    xlo = df["btc_low"].rolling(n // 2).min().shift(1)
    return stateful(df.index, b > hi, Z(df) > 0, b < xlo, Z(df) > 0,
                    df["close"], atr(df, 14), stop_k, True)


def rel_strength_vs_btc(df, ma_h, btc_ma_h):
    ratio = df["close"] / df["btc_close"]
    strong = ratio > sma(ratio, bars(df, ma_h))
    ok = df["btc_close"] > sma(df["btc_close"], bars(df, btc_ma_h))
    return _ls(strong & ok, df["close"] < 0)


def _daily_hold(df, sig):
    """只在每天最後一根（收盤於 UTC 00:00）更新訊號，其餘時間沿用，降低換手。"""
    step = pd.Series(df.index).diff().median()
    last = pd.Series((df.index + step).hour == 0, index=df.index)
    return sig.where(last).ffill().fillna(0).astype(int)


def xs_momentum(df, look_h, top):
    r = df[f"xs_rank_{look_h}"]
    return _daily_hold(df, _ls(r > 1 - top, r <= top))


def xs_reversal(df, look_h, top):
    r = df[f"xs_rank_{look_h}"]
    return _daily_hold(df, _ls(r <= top, r > 1 - top))


# ---------------------------------------------------------------- 登錄表

ALL, ALT = "all", "alts"
C = {}


def _reg(fn, grid, family, regime, why, payer, universe=ALL, overlap="", tfs=("4h",)):
    C[fn.__name__] = dict(fn=fn, grid=grid, family=family, regime=regime, why=why, payer=payer,
                          universe=universe, overlap=overlap, tfs=tfs)


_reg(tsmom_ls, {"look_h": [240, 480, 960], "band": [0.0, 0.05, 0.10]}, "趨勢", "走趨勢",
     "過去 10–40 天的漲跌方向延續：資訊擴散慢、追價資金分批進場",
     "太早反手的逆勢者與被迫回補的槓桿方")
_reg(tsmom_long_btcfilter, {"look_h": [240, 480, 960], "btc_ma_h": [1200, 2400, 4800]}, "趨勢", "走趨勢（只做多）",
     "只在 BTC 站上長均線時順勢做多：山寨的多頭行情幾乎都發生在 BTC 多頭裡",
     "熊市裡抄底山寨的人（我們不參與那一段）")
_reg(donchian_atr, {"entry_h": [240, 480, 960], "stop_k": [2.0, 3.0, 4.0]}, "趨勢", "走趨勢",
     "創 10–40 天新高/新低後跟隨，ATR 移動止損：突破觸發止損單與晚到的追價",
     "在區間邊緣放空/抄底被止損的人", overlap="工廠 donchian_trend（BTC 單幣已淘汰）")
_reg(ema_adx, {"slow_h": [200, 400, 800], "adx_th": [15, 20, 25]}, "趨勢", "走趨勢",
     "均線多空排列且 ADX 顯示有趨勢才持倉：把震盪期濾掉",
     "震盪期被均線來回打臉的人不是我們，是趨勢期的逆勢者")
_reg(squeeze_breakout, {"squeeze_pct": [0.15, 0.25, 0.35], "channel_h": [48, 96, 192]}, "突破", "窄區間→波動放大",
     "布林帶寬縮到近 30 天低檔後的通道突破：低波動期賣波動的一方在擴張時被迫止損",
     "賣波動、掛區間單的人", overlap="工廠 vol_squeeze_breakout（BTC 單幣已淘汰）")
_reg(trend_pullback, {"trend_h": [800, 1600, 3200], "rsi_th": [30, 35, 40]}, "趨勢回踩", "強趨勢",
     "趨勢內 RSI 回落到低檔時進場：短線恐慌止損的弱手把籌碼低價交出",
     "趨勢中被洗出場的短線止損單", overlap="工廠 trend_pullback（BTC 單幣已淘汰）")
_reg(vwap_trend, {"vwap_h": [96, 240, 480], "band": [0.0, 0.01, 0.02]}, "趨勢", "走趨勢",
     "價格站上/跌破滾動 VWAP：VWAP 是近期參與者的平均成本，站上代表多數人賺錢、賣壓小",
     "成本在 VWAP 另一側、被套後砍倉的人")
_reg(volume_breakout, {"vol_k": [1.5, 2.0, 3.0], "hold_h": [24, 72, 168]}, "突然放量", "波動放大",
     "放量突破近 80 小時高低點後跟隨：放量代表有新資金而非假突破",
     "突破時被迫平倉的反向槓桿", tfs=("4h", "1h"))
_reg(near_high_momo, {"look_h": [720, 1440, 2160], "prox": [0.03, 0.06, 0.10]}, "趨勢", "走趨勢（只做多）",
     "貼近 30–90 天高點時持有：錨定效應讓人不敢追高，價格對好消息反應不足",
     "在前高附近提早獲利了結的人")
_reg(dual_trend_long, {"slow_h": [1200, 2400, 4800], "fast_ratio": [0.1, 0.25, 0.5]}, "趨勢", "走趨勢（只做多）",
     "價格與快均線都在慢均線上才持有，否則空手：無聊的長線趨勢過濾，目的在躲大回撤",
     "熊市裡死拿的人承擔了我們躲掉的回撤")
_reg(vol_expansion_momo, {"look_h": [240, 480, 960], "volpct": [0.5, 0.6, 0.7]}, "趨勢", "波動特別大",
     "只在實現波動處於高檔時順勢：高波動期的方向更持久（去槓桿/追價是連鎖的）",
     "高波動期逆勢接刀的人")
_reg(overshoot_revert, {"k": [2.0, 3.0, 4.0], "hold_h": [8, 24, 48]}, "過激回歸", "波動特別大",
     "單根漲跌超過 k 倍標準差後反向：強平與恐慌造成的非資訊性價格衝擊會回吐",
     "被強平、市價砍倉的人", overlap="工廠 liq_reversal（BTC 單幣已淘汰）", tfs=("4h", "1h"))
_reg(rsi2_dip, {"rsi_th": [5, 10, 15], "trend_h": [800, 1600, 3200]}, "趨勢回踩", "強趨勢（只做多）",
     "多頭趨勢中 RSI(2) 極低時短買、反彈即走：趨勢內的急跌多為流動性事件",
     "急跌時市價賣出的人")
_reg(bb_revert_range, {"z": [1.5, 2.0, 2.5], "adx_max": [15, 20, 25]}, "過激回歸", "在震盪",
     "ADX 低（無趨勢）時，價格偏離 20 根均線 z 倍標準差就反向、回到均線出場",
     "震盪市裡追突破的人", overlap="工廠 lowvol_meanrev（BTC 單幣已淘汰）")
_reg(wick_reversal, {"k": [1.0, 1.5, 2.0], "hold_h": [8, 24, 48]}, "清算帶動", "波動特別大",
     "放量長下影（上影）後反向：影線是強平單被吃完後價格彈回的痕跡",
     "被清算的槓桿方", overlap="工廠 liq_reversal（BTC 單幣已淘汰）", tfs=("4h", "1h"))
_reg(volume_climax_fade, {"look_h": [72, 168, 336], "k": [2.0, 3.0, 4.0]}, "過激回歸", "波動特別大",
     "多日大漲/大跌末段放出巨量後反向：追價資金集中出清",
     "最後一批追價的人", overlap="工廠 volume_climax（BTC 單幣已淘汰）")
_reg(funding_extreme_fade, {"z": [1.5, 2.0, 2.5], "hold_h": [24, 72, 168]}, "資金費", "資金費極端",
     "資金費相對自身 90 天分佈走到極端時反向：擁擠的槓桿方要去化，且反向持倉收資金費",
     "擁擠一側付資金費的槓桿方", overlap="工廠 funding_fade（BTC 單幣已淘汰）")
_reg(funding_carry, {"th": [0.0002, 0.0003, 0.0005], "trend_h": [240, 480, 960]}, "資金費", "資金費極端",
     "資金費高且價格不再上漲時放空收費（反之亦然）：只在趨勢不對著幹時收 carry",
     "願意付高資金費維持槓桿的人")
_reg(momo_uncrowded, {"look_h": [240, 480, 960], "fund_cap": [0.0002, 0.0003, 0.0005]}, "趨勢", "走趨勢",
     "順勢，但資金費顯示同向已擁擠時不參與：避開趨勢末段的擠兌",
     "趨勢末段高槓桿追進去的人")
_reg(btc_lead_alt, {"th": [0.01, 0.015, 0.02], "hold_h": [4, 8, 24]}, "BTC 先動", "波動放大", universe=ALT,
     why="BTC 單根大動而山寨沒跟上時，順 BTC 方向做山寨：山寨的價格發現落後",
     payer="山寨盤口上反應慢的掛單方", overlap="工廠 btc_eth_lead_lag（ETH 單幣已淘汰）", tfs=("4h", "1h"))
_reg(btc_breakout_alt, {"entry_h": [240, 480, 960], "stop_k": [2.0, 3.0, 4.0]}, "BTC 先動", "走趨勢（只做多）",
     universe=ALT, why="BTC 創 10–40 天新高後做多山寨：資金先進 BTC、再外溢到山寨（多日級別的時差）",
     payer="BTC 突破後還在放空山寨的人")
_reg(rel_strength_vs_btc, {"ma_h": [240, 480, 960], "btc_ma_h": [1200, 2400, 4800]}, "強弱輪動", "走趨勢（只做多）",
     universe=ALT, why="BTC 多頭中只持有相對 BTC 走強的山寨：資金輪動有持續性",
     payer="持有弱勢山寨等補漲的人")
_reg(xs_momentum, {"look_h": [168, 336, 720], "top": [0.25, 0.34, 0.42]}, "強弱輪動", "不看方向（多空對沖）",
     "做多近 7–30 天最強的幣、放空最弱的幣，每天調一次：強弱排名有持續性，且對沖掉大盤",
     "抄底弱幣、放空強幣的人")
_reg(xs_reversal, {"look_h": [72, 168, 336], "top": [0.25, 0.34, 0.42]}, "過激回歸", "不看方向（多空對沖）",
     "做多近 3–14 天最弱的幣、放空最強的幣，每天調一次：短期相對漲跌有過度反應",
     "追逐短期強勢幣的人")

TESTS = [(name, tf) for name, c in C.items() for tf in c["tfs"]]
