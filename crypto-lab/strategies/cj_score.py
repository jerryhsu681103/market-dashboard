"""多因子共振分數進場（candle-json v2 的機械打分規則）。

來源：github.com/Allencake/candle-json-for-ai-researcher 的 tools/lens.py（MIT）。
原專案把分數交給 AI 主觀解讀；這裡只取「機械分數 + 紙面風險地圖」當成規則化策略來驗證，不含任何 AI 判斷。

賺誰的錢：趨勢、動量、方向強度、量能、大週期五項同時轉向時才進場的晚到追價資金與被迫回補的逆勢方。
失效情境：五項指標都是同一段價格的不同寫法（高度共線），共振只是趨勢已走完一段的確認，進場後即回吐。

分數（多空各 0–100，權重沿用原專案，不掃描）：
    趨勢 25：EMA20>EMA50（10）、EMA50>EMA200（5）、收盤>EMA20（10）
    動量 20：50<RSI<70（10）、MACD 柱>0（5）、MACD 柱上升（5）
    強度 15：DI+>DI−（7），且 ADX≥20 再加（8）
    量能 15：陽線且量比≥1.2（15），量比≥1.0（8）
    大週期 25：上一根已收完的 4 小時收盤 > 其 EMA50（25）
    空方為鏡像（RSI 取 30–50）。
進場：分數由下往上穿越 threshold、且高於對向分數、且 ATR 百分位 < 90（非高波動）。
出場：止損 = 進場收盤 ∓ stop_atr × ATR；目標 1 = 1.5R（到達後止損移到進場價）；目標 2 = 3R。

與原專案的差異（引擎規則造成，不是調參）：
    - 原專案的紙面帳本假設「觸價即成交」；引擎一律在下一根開盤成交。
      這裡在 K 線收盤後用該根高低點判斷是否觸及止損／目標，下一根開盤出場，成交價會比紙面帳本差或好。
    - EMA／RMA 用 pandas ewm（首值起算），原專案用 SMA 起算；暖機後數值一致。
"""
import numpy as np
import pandas as pd

EMA_FAST, EMA_MID, EMA_SLOW = 20, 50, 200
RSI_LEN, ADX_LEN, ATR_LEN = 14, 14, 14
VOL_LEN, ATR_RANK_LEN, HIGH_VOL_PCT = 20, 100, 90
HTF, HTF_EMA = "4h", 50
T1_R, T2_R = 1.5, 3.0


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def scores(df: pd.DataFrame) -> pd.DataFrame:
    """回傳每根 K 線的多方分數、空方分數、ATR、是否高波動。只用當根及之前的資料。"""
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    e20, e50, e200 = _ema(c, EMA_FAST), _ema(c, EMA_MID), _ema(c, EMA_SLOW)

    ch = c.diff()
    ag, al = _rma(ch.clip(lower=0), RSI_LEN), _rma((-ch).clip(lower=0), RSI_LEN)
    rsi = 100 - 100 / (1 + ag / al.where(al != 0))
    rsi = rsi.where(al != 0, 100.0).where(ag.notna() & al.notna())

    macd = _ema(c, 12) - _ema(c, 26)
    hist = macd - _ema(macd, 9)
    hist_up = hist > hist.shift(1)

    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = _rma(tr, ATR_LEN)
    up, dn = h.diff(), -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    mdm = dn.where((dn > up) & (dn > 0), 0.0)
    dip = 100 * _rma(pdm, ADX_LEN) / atr
    dim = 100 * _rma(mdm, ADX_LEN) / atr
    adx = 100 * _rma((dip - dim).abs() / (dip + dim).where((dip + dim) != 0), ADX_LEN)

    relv = v / v.rolling(VOL_LEN).mean()

    a = atr.to_numpy()
    rank = np.full(len(a), np.nan)
    if len(a) > ATR_RANK_LEN:
        win = np.lib.stride_tricks.sliding_window_view(a, ATR_RANK_LEN)[:-1]  # 第 i 列 = a[i : i+100]，對應第 i+100 根
        cur = a[ATR_RANK_LEN:]
        r = 100 * (win <= cur[:, None]).sum(axis=1) / ATR_RANK_LEN
        r = np.where(np.isnan(win[:, 0]) | np.isnan(cur), np.nan, r)
        rank[ATR_RANK_LEN:] = r

    # 大週期：上一個「已收完」的 4 小時區塊收盤 vs 其 EMA50
    blk = df.index.floor(HTF)
    hc = c.groupby(blk).last()
    if df.index[0] != blk[0]:
        hc = hc.iloc[1:]  # 第一個區塊不完整
    he = _ema(hc, HTF_EMA)
    prev = blk - pd.Timedelta(HTF)
    p_hc = hc.reindex(prev).to_numpy()
    p_he = he.reindex(prev).to_numpy()
    htf_up, htf_dn, htf_flat = p_hc > p_he, p_hc < p_he, p_hc == p_he

    bull, bear = (c > o).to_numpy(), (c < o).to_numpy()
    rv = relv.to_numpy()
    e20a, e50a, e200a, ca = e20.to_numpy(), e50.to_numpy(), e200.to_numpy(), c.to_numpy()
    rs, hi, hu = rsi.to_numpy(), hist.to_numpy(), hist_up.to_numpy()
    dp, dm, ad = dip.to_numpy(), dim.to_numpy(), adx.to_numpy()

    with np.errstate(invalid="ignore"):
        long_s = (
            10 * (e20a > e50a) + 5 * (e50a > e200a) + 10 * (ca > e20a)
            + 10 * ((rs > 50) & (rs < 70)) + 5 * (hi > 0) + 5 * hu
            + 7 * (dp > dm) + 8 * ((dp > dm) & (ad >= 20))
            + np.where(bull & (rv >= 1.2), 15, np.where(bull & (rv >= 1.0), 8, 0))
            + np.where(htf_up, 25, np.where(htf_flat, 10, 0))
        ).astype(float)
        short_s = (
            10 * (e20a < e50a) + 5 * (e50a < e200a) + 10 * (ca < e20a)
            + 10 * ((rs > 30) & (rs < 50)) + 5 * (hi < 0) + 5 * ~hu
            + 7 * (dm > dp) + 8 * ((dm > dp) & (ad >= 20))
            + np.where(bear & (rv >= 1.2), 15, np.where(bear & (rv >= 1.0), 8, 0))
            + np.where(htf_dn, 25, np.where(htf_flat, 10, 0))
        ).astype(float)

    valid = (e200.notna() & e200.shift(5).notna() & adx.notna() & atr.notna() & rsi.notna()
             & hist.notna() & hist.shift(1).notna() & relv.notna()).to_numpy() & ~np.isnan(rank)
    long_s[~valid] = np.nan
    short_s[~valid] = np.nan
    return pd.DataFrame({"long": long_s, "short": short_s, "atr": a, "high_vol": rank >= HIGH_VOL_PCT}, index=df.index)


def signal(df: pd.DataFrame, threshold: int = 70, stop_atr: float = 1.5) -> pd.Series:
    s = scores(df)
    ls, ss, atr, hv = s["long"].to_numpy(), s["short"].to_numpy(), s["atr"].to_numpy(), s["high_vol"].to_numpy()
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    out = np.zeros(len(c), dtype=np.int8)
    pos, stop, t1, t2, t1_hit = 0, 0.0, 0.0, 0.0, False
    prev_l = prev_s = 0.0
    for t in range(len(c)):
        if np.isnan(ls[t]):
            continue
        if pos == 1:
            if l[t] <= stop or h[t] >= t2:
                pos = 0
            elif not t1_hit and h[t] >= t1:
                t1_hit, stop = True, entry
        elif pos == -1:
            if h[t] >= stop or l[t] <= t2:
                pos = 0
            elif not t1_hit and l[t] <= t1:
                t1_hit, stop = True, entry
        if pos == 0 and not hv[t]:
            side = 0
            if ls[t] >= threshold and prev_l < threshold and ls[t] > ss[t]:
                side = 1
            elif ss[t] >= threshold and prev_s < threshold and ss[t] > ls[t]:
                side = -1
            if side:
                risk = stop_atr * atr[t]
                pos, entry, t1_hit = side, c[t], False
                stop, t1, t2 = entry - side * risk, entry + side * T1_R * risk, entry + side * T2_R * risk
        prev_l, prev_s = ls[t], ss[t]
        out[t] = pos
    return pd.Series(out, index=df.index)
