"""資料載入與樣本內／樣本外切分。

- load_insample(symbol)：前 4 年（第 1、2 關只能用這個）。
- load_full(symbol, idea, stage, reason)：含樣本外資料，只允許第 3、4 關使用；
  每個點子在每一關只能讀一次，讀取會寫進 oos_access_log.csv，第二次讀取直接報錯。
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OOS_LOG = ROOT / "oos_access_log.csv"
INSAMPLE_YEARS = 4


def _read(symbol: str) -> tuple[pd.DataFrame, pd.Series]:
    kp, fp = DATA / f"{symbol}_1h.parquet", DATA / f"{symbol}_funding.parquet"
    if not kp.exists() or not fp.exists():
        raise FileNotFoundError(f"找不到 {symbol} 資料，請先執行：python -m lab.download_data")
    df = pd.read_parquet(kp)
    funding = pd.read_parquet(fp)["funding_rate"]
    return df, funding


def insample_end(df: pd.DataFrame) -> pd.Timestamp:
    """樣本內結束時間（不含）：第一根 K 線 + 4 年。"""
    return df.index[0] + pd.DateOffset(years=INSAMPLE_YEARS)


def load_insample(symbol: str) -> tuple[pd.DataFrame, pd.Series]:
    df, funding = _read(symbol)
    end = insample_end(df)
    return df[df.index < end], funding[funding.index <= end]


def _logged(idea: str, stage: int) -> bool:
    if not OOS_LOG.exists():
        return False
    with OOS_LOG.open(encoding="utf-8") as f:
        return any(r["idea"] == idea and int(r["stage"]) == stage for r in csv.DictReader(f))


def load_full(symbol: str, idea: str, stage: int, reason: str,
              end: str | pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """讀取含樣本外的完整資料。stage 只能是 3 或 4（或 'recheck' 用 stage=6）。"""
    if stage not in (3, 4, 6):
        raise PermissionError("樣本外資料只允許第 3、4 關及每週複查（stage=6）使用")
    if stage in (3, 4) and _logged(idea, stage):
        raise PermissionError(f"點子 {idea!r} 已在第 {stage} 關看過樣本外資料，不得重看")
    df, funding = _read(symbol)
    if end is not None:
        end = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end)
        df, funding = df[df.index < end], funding[funding.index < end]
    new = not OOS_LOG.exists()
    with OOS_LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp_utc", "idea", "stage", "symbol", "data_end", "reason"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), idea, stage,
                    symbol, str(df.index[-1]), reason])
    return df, funding
