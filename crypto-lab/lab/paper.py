"""紙面交易：追蹤 watchlist.csv 裡已晉級策略的晉級後表現（不改參數、不重新優化）。

用法：
    python -m lab.download_data --daily     # 先更新到昨天
    python -m lab.paper                     # 輸出 paper/summary.csv 與各策略權益曲線

做法：用完整資料跑 engine.run（讓指標暖機），只截取 promoted_date 之後的逐根明細，
再交給 engine.summarize 計算晉級後績效。所有數字仍來自 engine.py。
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd

from . import engine
from .costs import load_costs
from .data import load_full

ROOT = Path(__file__).resolve().parents[1]
WATCHLIST = ROOT / "watchlist.csv"
OUT = ROOT / "paper"


def track(row: pd.Series, fee, slippage) -> dict:
    mod = importlib.import_module(row["module"])
    params = json.loads(row["params_json"])
    df, funding = load_full(row["symbol"], idea=row["name"], stage=6, reason="paper tracking")
    r = engine.run(mod.signal, params, df, fee, slippage, funding, keep_detail=True)
    start = pd.Timestamp(row["promoted_date"], tz="UTC")
    detail = r["detail"][r["detail"].index >= start]
    if len(detail) < 3:
        return {"name": row["name"], "status": "晉級後資料不足"}
    s = engine.summarize(detail, r["bars_per_year"])
    s["equity"].to_csv(OUT / f"{row['name']}_equity.csv", header=["equity"])
    return {"name": row["name"], "since": str(start.date()), "bars": s["bars"],
            "total_return": s["total_return"], "sharpe": s["sharpe"],
            "profit_factor": s["profit_factor"], "max_drawdown": s["max_drawdown"],
            "trade_count": s["trade_count"],
            "backtest_sharpe": row.get("backtest_sharpe"), "backtest_pf": row.get("backtest_pf"),
            "status": "ok"}


def main():
    fee, slippage = load_costs()
    wl = pd.read_csv(WATCHLIST)
    if wl.empty:
        print("watchlist.csv 為空，沒有紙面交易中的策略。")
        return
    OUT.mkdir(exist_ok=True)
    rows = [track(r, fee, slippage) for _, r in wl.iterrows()]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
