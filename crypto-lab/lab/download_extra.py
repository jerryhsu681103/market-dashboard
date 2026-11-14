"""下載額外資料集並彙整成 1 小時（使用者於 2026-11-14 同意新增）。

用法：
    python -m lab.download_extra                       # 全部：metrics spot premium depth
    python -m lab.download_extra --sets metrics spot

輸出（索引 = 1 小時 K 線開盤時間 UTC；每列只含該小時內 [t, t+1h) 的觀測，收盤時已知）：
    data/{symbol}_metrics_1h.parquet   未平倉量、多空比（該小時最後一筆 5 分鐘快照）
    data/{symbol}_spot_1h.parquet      現貨 1 小時 K 線
    data/{symbol}_premium_1h.parquet   溢價指數 1 小時 K 線（永續相對指數價）
    data/{symbol}_depth_1h.parquet     訂單簿 ±1%～±5% 名目深度（該小時內快照平均，USD）
    data/quality_extra_{symbol}.txt    各資料集涵蓋期間與缺失小時
"""
from __future__ import annotations

import argparse
import hashlib
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from .download_data import KLINE_COLS, _months, _to_utc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
HOST = "https://data.binance.vision/data"

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=3)
session.mount("https://", adapter)

START = {  # 各資料集最早檔案
    "metrics": {"BTCUSDT": date(2020, 9, 1), "ETHUSDT": date(2021, 12, 1)},
    "depth": {"BTCUSDT": date(2023, 1, 1), "ETHUSDT": date(2023, 1, 1)},
}


def _get(rel: str, cache: bool = True) -> bytes | None:
    path = RAW / rel
    if cache and path.exists():
        return path.read_bytes()
    r = session.get(f"{HOST}/{rel}", timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    c = session.get(f"{HOST}/{rel}.CHECKSUM", timeout=60)
    if c.status_code == 200 and c.text.split()[0].strip().lower() != hashlib.sha256(r.content).hexdigest():
        raise RuntimeError(f"SHA256 不符：{rel}")
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    return r.content


def _csv(blob: bytes, names=None) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    if names is None:
        return pd.read_csv(io.BytesIO(raw))
    has_header = not raw[:1].isdigit()
    return pd.read_csv(io.BytesIO(raw), header=0 if has_header else None, names=names)


def _days(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _hourly_last(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("1h", label="left", closed="left").last().dropna(how="all")


# ---------------------------------------------------------------- 各資料集

def metrics(symbol: str, end: date) -> pd.DataFrame:
    def one(d):
        blob = _get(f"futures/um/daily/metrics/{symbol}/{symbol}-metrics-{d.isoformat()}.zip")
        return None if blob is None else _csv(blob)
    with ThreadPoolExecutor(8) as ex:
        frames = [f for f in ex.map(one, _days(START["metrics"][symbol], end)) if f is not None]
    m = pd.concat(frames, ignore_index=True)
    m.index = pd.DatetimeIndex(pd.to_datetime(m["create_time"], utc=True))
    m = m.sort_index()
    m = m[~m.index.duplicated(keep="last")]
    out = pd.DataFrame({
        "oi": m["sum_open_interest"], "oi_value": m["sum_open_interest_value"],
        "top_ls_account": m["count_toptrader_long_short_ratio"],
        "top_ls_position": m["sum_toptrader_long_short_ratio"],
        "global_ls_account": m["count_long_short_ratio"],
        "taker_ls_vol": m["sum_taker_long_short_vol_ratio"],
    }).astype(float)
    return _hourly_last(out)


def _monthly_klines(prefix: str, symbol: str, first: date, end_month: date) -> pd.DataFrame:
    frames = []
    for y, mth in _months(first, end_month):
        blob = _get(f"{prefix}/{symbol}/1h/{symbol}-1h-{y}-{mth:02d}.zip")
        if blob is not None:
            frames.append(_csv(blob, KLINE_COLS))
    k = pd.concat(frames, ignore_index=True)
    k.index = _to_utc(k["open_time"])
    k = k.sort_index()
    return k[~k.index.duplicated(keep="last")]


def spot(symbol: str, end: date) -> pd.DataFrame:
    k = _monthly_klines("spot/monthly/klines", symbol, date(2017, 8, 1), end)
    cols = ["open", "high", "low", "close", "volume", "quote_volume", "trades",
            "taker_buy_volume", "taker_buy_quote_volume"]
    return k[cols].astype(float)


def premium(symbol: str, end: date) -> pd.DataFrame:
    k = _monthly_klines("futures/um/monthly/premiumIndexKlines", symbol, date(2020, 1, 1), end)
    return k[["open", "high", "low", "close"]].astype(float)


def depth(symbol: str, end: date) -> pd.DataFrame:
    """訂單簿深度原始檔很大（每天約 0.5 MB），不保留快取，下載後立即彙整成 1 小時平均。"""
    def one(d):
        blob = _get(f"futures/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{d.isoformat()}.zip", cache=False)
        if blob is None:
            return None
        x = _csv(blob)
        x["ts"] = pd.to_datetime(x["timestamp"], utc=True).dt.floor("h")
        p = x.pivot_table(index="ts", columns="percentage", values="notional", aggfunc="mean")
        # 例如 -1 → depth_m1、0.2 → depth_p0_2（2026-01-15 起新增 ±0.2% 檔位）
        p.columns = [f"depth_{'m' if float(c) < 0 else 'p'}"
                     f"{format(abs(float(c)), 'g').replace('.', '_')}" for c in p.columns]
        return p
    with ThreadPoolExecutor(8) as ex:
        frames = [f for f in ex.map(one, _days(START["depth"][symbol], end)) if f is not None]
    out = pd.concat(frames).sort_index()
    out.index.name = None
    return out[~out.index.duplicated(keep="last")]


# ---------------------------------------------------------------- 主程式

def _coverage(name: str, df: pd.DataFrame) -> list[str]:
    full = pd.date_range(df.index[0], df.index[-1], freq="h")
    missing = full.difference(df.index)
    lines = [f"## {name}", f"區間：{df.index[0]} → {df.index[-1]}，應有 {len(full)} 小時，實有 {len(df)}，缺 {len(missing)}"]
    if len(missing):
        gaps = pd.Series(missing).diff().ne(pd.Timedelta("1h")).cumsum()
        groups = list(pd.Series(missing).groupby(gaps.to_numpy()))
        for _, g in groups[:15]:
            lines.append(f"  - {g.iloc[0]} → {g.iloc[-1]}（{len(g)} 小時）")
        if len(groups) > 15:
            lines.append(f"  - ……另有 {len(groups) - 15} 段")
    lines.append(f"欄位：{', '.join(df.columns)}")
    lines.append("")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    ap.add_argument("--sets", nargs="+", default=["metrics", "spot", "premium", "depth"])
    a = ap.parse_args(argv)
    today = datetime.now(timezone.utc).date()
    last_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    last_day = today.replace(day=1) - timedelta(days=1)  # 與 K 線一致：到上個月底
    funcs = {"metrics": lambda s: metrics(s, last_day), "spot": lambda s: spot(s, last_month),
             "premium": lambda s: premium(s, last_month), "depth": lambda s: depth(s, last_day)}
    for sym in a.symbols:
        for name in a.sets:
            print(f"== {sym} {name}", flush=True)
            df = funcs[name](sym)
            df.index.name = "open_time"
            df.to_parquet(DATA / f"{sym}_{name}_1h.parquet")
            print("\n".join(_coverage(name, df)), flush=True)
        write_report(sym)


def write_report(sym: str) -> None:
    """依目前 data/ 裡已有的資料集重寫品質報告（只下載部分資料集時不會蓋掉其他資料集的紀錄）。"""
    report = [f"# {sym} 額外資料集品質報告（產生於 {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC）", ""]
    for name in ("metrics", "spot", "premium", "depth"):
        p = DATA / f"{sym}_{name}_1h.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            report += _coverage(name, d)
            nulls = d.isna().groupby(d.index.year).mean()
            nulls = nulls.loc[:, nulls.max() > 0.05]
            if not nulls.empty:
                report += [f"{name} 單一欄位缺值比例（依年度，僅列出某年缺值 > 5% 的欄位）：",
                           nulls.round(3).to_string(), ""]
            # 溢價指數本來就可以為負，不檢查
            bad = {} if name == "premium" else {c: int((d[c] <= 0).sum()) for c in d.columns if (d[c] <= 0).any()}
            if bad:
                report += [f"{name} 中 ≤ 0 的值（載入時視為缺值）：{bad}", ""]
    (DATA / f"quality_extra_{sym}.txt").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
