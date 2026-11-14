"""第 4 關：前推置換檢驗。

用法：
    python -m scripts.stage4 <idea> --timeframe 4h --dry-run [--n-perm 8]   # 彩排：只用樣本內
    python -m scripts.stage4 <idea> --timeframe 4h                          # 正式：讀樣本外（每個點子一次）

做法：只打亂第一個訓練窗口之後的 K 線（外生特徵與資金費率固定），每條假行情完整跑一次滾動前推。
門檻：pass_line 基本門檻 ÷ 累計進入第 4 關的點子數（lab.research.stage4_threshold）。
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("idea")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    spec, tf = IDEAS[a.idea], a.timeframe
    if "legs" in spec or spec.get("attach"):
        raise NotImplementedError("價差／多幣種點子的第 4 關尚未實作")
    symbol = spec.get("symbol", a.symbol)
    fn = importlib.import_module(f"strategies.{a.idea}").signal
    grid = grid_for(spec, tf)
    fee, slip = load_costs()
    log_path = ROOT / "research_log.csv"

    if a.dry_run:
        df, funding = load_insample(symbol, tf)
        train_bars = len(df) // 2
    else:
        log = pd.read_csv(log_path, dtype=str)
        row = log[(log["idea"] == a.idea) & (log["timeframe"] == tf)]
        if len(row) != 1 or row["status"].iat[0] != "alive_stage3":
            raise SystemExit(f"{a.idea} [{tf}] 不是 alive_stage3，不能進第 4 關")
        df, funding = load_full(symbol, idea=f"{a.idea}@{tf}", stage=4,
                                reason="第 4 關前推置換檢驗", timeframe=tf)
        train_bars = int((df.index < insample_end(df)).sum())
    df = attach_funding(df, funding)
    feat_cols = []
    if spec.get("features"):
        before = set(df.columns)
        df = attach_features(df, symbol, spec["features"], tf)
        feat_cols = [c for c in df.columns if c not in before]
    step_h = pd.Series(df.index).diff().median().total_seconds() / 3600
    test_bars = int(round(30 * 24 / step_h))
    years_forward = (len(df) - train_bars) * step_h / (365 * 24)
    thr = research.stage4_threshold(log_path, years_forward)

    r = research.permutation_test(
        fn, grid, df, fee, slip, funding, kind="walkforward", n_perm=a.n_perm, seed=a.seed,
        objective="sharpe", fixed_cols=feat_cols,
        wf_kw={"train_bars": train_bars, "test_bars": test_bars, "min_trades": GATE["min_trades"]})
    status = "watchlist" if r["p_value"] < thr["threshold"] else "dead_stage4"
    perm = r["perm_scores"][np.isfinite(r["perm_scores"])]
    summary = {
        "idea": a.idea, "symbol": symbol, "timeframe": tf, "dry_run": a.dry_run,
        "real_forward_sharpe": round(r["real"], 4), "p_value": round(r["p_value"], 5),
        "threshold": thr["threshold"], "threshold_base": thr["base"],
        "n_tested_stage4": thr["n_tested_stage4"], "years_forward": round(years_forward, 2),
        "n_perm": r["n_perm"], "n_ge_real": r["n_ge_real"], "seed": r["seed"],
        "perm_sharpe_mean": round(float(perm.mean()), 4) if len(perm) else None,
        "perm_sharpe_p99": round(float(np.quantile(perm, 0.99)), 4) if len(perm) else None,
        "fixed_cols": feat_cols, "status": status,
        "command": f"python -m scripts.stage4 {a.idea} --timeframe {tf} --n-perm {a.n_perm} --seed {a.seed}"
                   + (" --dry-run" if a.dry_run else ""),
    }
    out = ROOT / "reports" / "permutation"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{a.idea}_{symbol}_{tf}_stage4" + ("_dryrun" if a.dry_run else "")
    research.save_permutation_plot(r, out / f"{stem}.png",
                                   title=f"{a.idea} {symbol} {tf}  stage 4  p = {r['p_value']:.4f}")
    (out / f"{stem}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not a.dry_run:
        log = pd.read_csv(log_path, dtype=str)
        m = (log["idea"] == a.idea) & (log["timeframe"] == tf)
        log.loc[m, "stage4_p"] = f"{r['p_value']:.5f}"
        log.loc[m, "stage4_threshold"] = f"{thr['threshold']:.5f}"
        log.loc[m, "status"] = status
        if status == "dead_stage4":
            log.loc[m, "death_reason"] = f"前推置換 p = {r['p_value']:.4f} ≥ 門檻 {thr['threshold']:.4f}"
        log.loc[m, "scripts"] = log.loc[m, "scripts"] + f" ; {summary['command']}  (reports/permutation/{stem}.json)"
        log.to_csv(log_path, index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
