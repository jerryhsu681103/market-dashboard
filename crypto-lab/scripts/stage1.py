"""第 1 關：樣本內參數掃描 + 參數高原 + 門檻判定。

用法：python -m scripts.stage1 donchian_trend funding_fade liq_reversal [--symbol BTCUSDT]
輸出：reports/heatmaps/*.png、reports/stage1/<idea>.csv、research_log.csv 追加一行／點子。
"""
import argparse
import csv
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from lab import research
from lab.costs import load_costs
from lab.data import attach_funding, load_insample

ROOT = Path(__file__).resolve().parents[1]

IDEAS = {
    "donchian_trend": {
        "grid": {"entry": [72, 120, 168, 240, 336, 480, 720], "exit_ratio": [0.25, 0.5, 0.75, 1.0]},
        "rationale": "大級別突破後的跟隨：晚到資金追價、空方被迫回補",
    },
    "funding_fade": {
        "grid": {"threshold": [0.0002, 0.0003, 0.0005, 0.0008, 0.001], "hold": [8, 24, 48, 72, 120]},
        "rationale": "資金費率極端時反向：擁擠槓桿去化，且反向持倉收取資金費",
    },
    "liq_reversal": {
        "grid": {"k": [3.0, 3.5, 4.0, 5.0, 6.0], "hold": [3, 6, 12, 24, 48]},
        "rationale": "清算瀑布後反彈：強平造成的非資訊性賣壓被推過頭",
    },
}

GATE = {"min_sharpe": 1.0, "min_pf": 1.2, "min_trades": 100, "too_good_sharpe": 4.0, "too_good_dd": -0.03}


def judge(best: pd.Series | None, plateau: dict) -> tuple[str, str]:
    if best is None:
        return "dead", f"沒有任何參數組合的交易次數 ≥ {GATE['min_trades']}"
    if best["sharpe"] > GATE["too_good_sharpe"] or best["max_drawdown"] > GATE["too_good_dd"]:
        return "review", "成績好得不正常，需先查前視偏差"
    fails = []
    if best["sharpe"] < GATE["min_sharpe"]:
        fails.append(f"Sharpe {best['sharpe']:.2f} < {GATE['min_sharpe']}")
    if best["profit_factor"] < GATE["min_pf"]:
        fails.append(f"PF {best['profit_factor']:.2f} < {GATE['min_pf']}")
    if plateau["is_spike"]:
        fails.append(f"參數尖峰：鄰近中位數 / 最優 = {plateau['ratio']:.2f} < {plateau['threshold']}")
    if fails:
        return "dead", "；".join(fails)
    return "pass", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ideas", nargs="+")
    ap.add_argument("--symbol", default="BTCUSDT")
    a = ap.parse_args()
    fee, slip = load_costs()
    df, funding = load_insample(a.symbol)
    df = attach_funding(df, funding)
    today = datetime.now(timezone.utc).date().isoformat()
    log_path = ROOT / "research_log.csv"
    n_prev = len(pd.read_csv(log_path))

    for i, name in enumerate(a.ideas, 1):
        spec = IDEAS[name]
        fn = importlib.import_module(f"strategies.{name}").signal
        grid = spec["grid"]
        res = research.grid_search(fn, grid, df, fee, slip, funding)
        out_csv = ROOT / "reports" / "stage1" / f"{name}_{a.symbol}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out_csv, index=False)

        x, y = list(grid)
        hm = research.save_heatmap(res, x, y, ROOT / "reports" / "heatmaps" / f"{name}_{a.symbol}_stage1.png",
                                   title=f"{name} {a.symbol} in-sample Sharpe")
        eligible = res[res["trade_count"] >= GATE["min_trades"]]
        best = eligible.loc[eligible["sharpe"].idxmax()] if not eligible.empty else None
        plateau = research.plateau_check(eligible if not eligible.empty else res, grid)
        status, reason = judge(best, plateau)

        summary = {"symbol": a.symbol, "best_params": {k: next(v for v in grid[k] if v == best[k]) for k in grid} if best is not None else None,
                   "sharpe": None if best is None else round(float(best["sharpe"]), 4),
                   "pf": None if best is None else round(float(best["profit_factor"]), 4),
                   "total_return": None if best is None else round(float(best["total_return"]), 4),
                   "max_dd": None if best is None else round(float(best["max_drawdown"]), 4),
                   "trades": None if best is None else int(best["trade_count"]),
                   "plateau_ratio": round(plateau["ratio"], 4)}
        print(f"\n== {name}  →  {status}  {reason}")
        print(json.dumps(summary, ensure_ascii=False, default=float))

        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                today, name, spec["rationale"], json.dumps(grid), json.dumps(summary, ensure_ascii=False, default=float),
                "", "", "", "", str(hm.relative_to(ROOT)),
                {"pass": "alive_stage1", "dead": "dead_stage1", "review": "review_stage1"}[status],
                reason, n_prev + i,
                f"python -m scripts.stage1 {name} --symbol {a.symbol}  ({out_csv.relative_to(ROOT)})",
            ])


if __name__ == "__main__":
    main()
