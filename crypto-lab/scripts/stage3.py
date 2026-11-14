"""第 3 關：滾動前推（訓練 4 年 → 交易 30 天 → 往前滾動）。

用法：
    python -m scripts.stage3 <idea> --timeframe 4h --dry-run   # 彩排：只用樣本內資料（訓練 2 年），不讀樣本外
    python -m scripts.stage3 <idea> --timeframe 4h             # 正式：讀取樣本外（每個點子只能一次，記錄在 oos_access_log.csv）

規則：
  - 第一個測試窗口從樣本內結束時點（2024-01-01）開始；訓練窗口 = 之前 4 年（同樣長度往前滾動）
  - 每個訓練窗口用與第 1 關相同的參數網格與選擇規則（交易次數 ≥ 100 中 Sharpe 最高）
  - 淘汰：前推 Sharpe < 0.5，或 PF < 1.1，或前推 Sharpe < 樣本內最優值的 50%
"""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lab import research
from lab.costs import load_costs
from lab.data import attach_features, attach_funding, insample_end, load_full, load_insample
from scripts.stage1 import GATE, IDEAS, grid_for

ROOT = Path(__file__).resolve().parents[1]
GATE3 = {"min_sharpe": 0.5, "min_pf": 1.1, "min_ratio_to_insample": 0.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("idea")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    spec, tf = IDEAS[a.idea], a.timeframe
    if "legs" in spec or spec.get("attach"):
        raise NotImplementedError("價差／多幣種點子的第 3 關尚未實作")
    symbol = spec.get("symbol", a.symbol)
    fn = importlib.import_module(f"strategies.{a.idea}").signal
    grid = grid_for(spec, tf)
    fee, slip = load_costs()

    log = pd.read_csv(ROOT / "research_log.csv", dtype=str)
    row = log[(log["idea"] == a.idea) & (log["timeframe"] == tf)]
    if len(row) != 1 or row["status"].iat[0] != "alive_stage2":
        raise RuntimeError(f"{a.idea} [{tf}] 不是 alive_stage2，不能進第 3 關")
    insample_sharpe = json.loads(row["stage1"].iat[0])["sharpe"]

    if a.dry_run:
        df, funding = load_insample(symbol, tf)
        train_bars = len(df) // 2  # 彩排：前 2 年訓練、後 2 年前推，全部在樣本內
    else:
        df, funding = load_full(symbol, idea=f"{a.idea}@{tf}", stage=3,
                                reason="第 3 關滾動前推", timeframe=tf)
        train_bars = int((df.index < insample_end(df)).sum())
    df = attach_funding(df, funding)
    if spec.get("features"):
        df = attach_features(df, symbol, spec["features"], tf)
    test_bars = int(round(30 * 24 / (pd.Series(df.index).diff().median().total_seconds() / 3600)))

    wf = research.walk_forward(fn, grid, df, fee, slip, funding, train_bars=train_bars,
                               test_bars=test_bars, objective="sharpe", min_trades=GATE["min_trades"])

    fails = []
    if not (wf["sharpe"] >= GATE3["min_sharpe"]):
        fails.append(f"前推 Sharpe {wf['sharpe']:.2f} < {GATE3['min_sharpe']}")
    if not (wf["profit_factor"] >= GATE3["min_pf"]):
        fails.append(f"PF {wf['profit_factor']:.2f} < {GATE3['min_pf']}")
    if not (wf["sharpe"] >= GATE3["min_ratio_to_insample"] * insample_sharpe):
        fails.append(f"前推 Sharpe {wf['sharpe']:.2f} < 樣本內 {insample_sharpe:.2f} × 50%")
    status = "alive_stage3" if not fails else "dead_stage3"

    net, det = wf["returns"], wf["detail"]
    yearly = pd.DataFrame({
        "return": net.groupby(net.index.year).apply(lambda x: float((1 + x).prod() - 1)),
        "sharpe": net.groupby(net.index.year).apply(
            lambda x: float(x.mean() / x.std() * np.sqrt(wf["bars_per_year"])) if x.std() > 0 else float("nan")),
        "exposure": (det["pos"] != 0).groupby(det.index.year).mean(),
    })
    windows = pd.DataFrame(wf["windows"])
    summary = {
        "idea": a.idea, "symbol": symbol, "timeframe": tf, "dry_run": a.dry_run,
        "test_start": wf["start"], "test_end": wf["end"], "n_windows": len(windows),
        "train_bars": train_bars, "test_bars": test_bars,
        "sharpe": round(wf["sharpe"], 4), "profit_factor": round(wf["profit_factor"], 4),
        "total_return": round(wf["total_return"], 4), "cagr": round(wf["cagr"], 4),
        "max_drawdown": round(wf["max_drawdown"], 4), "trade_count": wf["trade_count"],
        "win_rate": round(wf["win_rate"], 4), "exposure": round(wf["exposure"], 4),
        "total_cost": round(wf["total_cost"], 4), "total_funding": round(wf["total_funding"], 4),
        "insample_sharpe": insample_sharpe, "status": status, "fails": fails,
        "param_changes": int((windows["params"].astype(str) != windows["params"].astype(str).shift()).sum() - 1),
        "command": f"python -m scripts.stage3 {a.idea} --timeframe {tf}" + (" --dry-run" if a.dry_run else ""),
    }
    out = ROOT / "reports" / "stage3"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{a.idea}_{symbol}_{tf}_stage3" + ("_dryrun" if a.dry_run else "")
    (out / f"{stem}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    windows.to_csv(out / f"{stem}_windows.csv", index=False)
    yearly.to_csv(out / f"{stem}_yearly.csv")
    wf["equity"].to_csv(out / f"{stem}_equity.csv", header=["equity"])

    if not a.dry_run:
        log = pd.read_csv(ROOT / "research_log.csv", dtype=str)
        m = (log["idea"] == a.idea) & (log["timeframe"] == tf)
        log.loc[m, "stage3"] = json.dumps({k: summary[k] for k in (
            "test_start", "test_end", "n_windows", "sharpe", "profit_factor", "total_return",
            "max_drawdown", "trade_count")}, ensure_ascii=False)
        log.loc[m, "status"] = status
        if fails:
            log.loc[m, "death_reason"] = "；".join(fails)
        log.loc[m, "scripts"] = log.loc[m, "scripts"] + f" ; {summary['command']}  (reports/stage3/{stem}.json)"
        log.to_csv(ROOT / "research_log.csv", index=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n分年度：\n" + yearly.round(3).to_string())
    print("\n各窗口參數：\n" + windows.to_string(index=False))


if __name__ == "__main__":
    main()
