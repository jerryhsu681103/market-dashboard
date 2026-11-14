"""從 data.binance.vision 下載 USDT 本位永續合約 1 小時 K 線與資金費率。

用法：
    python -m lab.download_data                       # BTCUSDT、ETHUSDT，最早月份 → 上個月
    python -m lab.download_data --daily               # 另外補上本月至昨天的日檔（紙面交易用）
    python -m lab.download_data --symbols SOLUSDT

輸出：
    data/{symbol}_1h.parquet        K 線（索引 = 開盤時間 UTC）
    data/{symbol}_funding.parquet   資金費率（索引 = 結算時間，取整到整點 UTC）
    data/quality_{symbol}.txt       資料品質報告（連續性、重複、缺失、OHLC 異常）
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.binance.vision/data/futures/um"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
FIRST_MONTH = date(2019, 9, 1)  # U 本位永續合約最早上線月份

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]

session = requests.Session()


def _months(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _fetch(rel: str) -> bytes | None:
    """下載 zip 並以 .CHECKSUM 驗證 SHA256；檔案不存在回傳 None。已下載過則讀本地快取。"""
    cache = RAW / rel
    if cache.exists():
        return cache.read_bytes()
    r = session.get(f"{BASE}/{rel}", timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    c = session.get(f"{BASE}/{rel}.CHECKSUM", timeout=60)
    if c.status_code == 200:
        expected = c.text.split()[0].strip().lower()
        actual = hashlib.sha256(r.content).hexdigest()
        if expected != actual:
            raise RuntimeError(f"SHA256 不符：{rel}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(r.content)
    return r.content


def _read_zip_csv(blob: bytes, names: list[str] | None) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    first = raw.split(b"\n", 1)[0]
    has_header = not first[:1].isdigit()
    if names is None:
        return pd.read_csv(io.BytesIO(raw))
    return pd.read_csv(io.BytesIO(raw), header=0 if has_header else None, names=names)


def _to_utc(ms_or_us: pd.Series) -> pd.DatetimeIndex:
    """毫秒或微秒時間戳 → UTC。Binance 部分資料集在 2025 年後改用微秒。"""
    v = ms_or_us.astype("int64")
    v = v.where(v < 10**14, v // 1000)  # 微秒轉毫秒
    return pd.DatetimeIndex(pd.to_datetime(v, unit="ms", utc=True))


def download_klines(symbol: str, last_month: date, daily: bool) -> pd.DataFrame:
    frames, started = [], False
    for y, m in _months(FIRST_MONTH, last_month):
        rel = f"monthly/klines/{symbol}/1h/{symbol}-1h-{y}-{m:02d}.zip"
        blob = _fetch(rel)
        if blob is None:
            if started:
                print(f"  ⚠ 缺少月檔：{rel}", file=sys.stderr)
            continue
        started = True
        frames.append(_read_zip_csv(blob, KLINE_COLS))
        if m == 12:
            print(f"  {symbol} klines 已下載至 {y}-12")
    if daily:
        today = datetime.now(timezone.utc).date()
        d = today.replace(day=1)
        while d < today:
            rel = f"daily/klines/{symbol}/1h/{symbol}-1h-{d.isoformat()}.zip"
            blob = _fetch(rel)
            if blob is not None:
                frames.append(_read_zip_csv(blob, KLINE_COLS))
            d += timedelta(days=1)
    if not frames:
        raise RuntimeError(f"{symbol}：找不到任何 K 線檔")
    df = pd.concat(frames, ignore_index=True)
    df.index = _to_utc(df["open_time"])
    df.index.name = "open_time"
    df = df.drop(columns=["open_time", "close_time", "ignore"])
    return df.astype({"open": float, "high": float, "low": float, "close": float,
                      "volume": float, "quote_volume": float, "trades": "int64",
                      "taker_buy_volume": float, "taker_buy_quote_volume": float})


def download_funding(symbol: str, last_month: date) -> pd.Series:
    frames = []
    for y, m in _months(FIRST_MONTH, last_month):
        rel = f"monthly/fundingRate/{symbol}/{symbol}-fundingRate-{y}-{m:02d}.zip"
        blob = _fetch(rel)
        if blob is not None:
            frames.append(_read_zip_csv(blob, None))
    if not frames:
        raise RuntimeError(f"{symbol}：找不到任何資金費率檔")
    f = pd.concat(frames, ignore_index=True)
    idx = _to_utc(f["calc_time"]).floor("h")  # calc_time 帶有幾毫秒抖動
    s = pd.Series(f["last_funding_rate"].astype(float).to_numpy(), index=idx, name="funding_rate")
    s.index.name = "funding_time"
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def quality_report(symbol: str, df: pd.DataFrame, funding: pd.Series) -> tuple[pd.DataFrame, str]:
    lines = [f"# {symbol} 資料品質報告（產生於 {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC）", ""]
    dup = df.index.duplicated(keep="last")
    lines.append(f"重複行：{int(dup.sum())}" + ("（已保留最後一筆）" if dup.any() else ""))
    df = df[~dup].sort_index()

    full = pd.date_range(df.index[0], df.index[-1], freq="h")
    missing = full.difference(df.index)
    lines.append(f"區間：{df.index[0]} → {df.index[-1]}，應有 {len(full)} 根，實有 {len(df)} 根")
    lines.append(f"缺失的小時：{len(missing)}")
    if len(missing):
        gaps = pd.Series(missing).diff().ne(pd.Timedelta("1h")).cumsum()
        for _, g in pd.Series(missing).groupby(gaps.to_numpy()):
            lines.append(f"  - {g.iloc[0]} → {g.iloc[-1]}（{len(g)} 小時）")
    off = (df.index.minute != 0) | (df.index.second != 0)
    lines.append(f"非整點時間戳：{int(off.sum())}")

    bad_ohlc = (df["high"] < df[["open", "close"]].max(axis=1)) | \
               (df["low"] > df[["open", "close"]].min(axis=1)) | (df["low"] <= 0)
    lines.append(f"OHLC 異常（high/low 不包住 open/close，或價格 ≤ 0）：{int(bad_ohlc.sum())}")
    lines.append(f"成交量為 0 的 K 線：{int((df['volume'] == 0).sum())}")

    lines.append("")
    lines.append(f"資金費率：{len(funding)} 筆，{funding.index[0]} → {funding.index[-1]}")
    fgap = pd.Series(funding.index).diff().dropna()
    lines.append(f"結算間隔分佈：{fgap.value_counts().head(5).to_dict()}")
    big = fgap[fgap > pd.Timedelta("8h")]
    lines.append(f"超過 8 小時的結算缺口：{len(big)}")
    return df, "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    ap.add_argument("--daily", action="store_true", help="補上本月至昨天的日檔")
    args = ap.parse_args(argv)

    today = datetime.now(timezone.utc).date()
    last_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    DATA.mkdir(exist_ok=True)
    for sym in args.symbols:
        print(f"== {sym}")
        df = download_klines(sym, last_month, args.daily)
        funding = download_funding(sym, last_month)
        df, report = quality_report(sym, df, funding)
        df.to_parquet(DATA / f"{sym}_1h.parquet")
        funding.to_frame().to_parquet(DATA / f"{sym}_funding.parquet")
        (DATA / f"quality_{sym}.txt").write_text(report, encoding="utf-8")
        print(report)


if __name__ == "__main__":
    main()
