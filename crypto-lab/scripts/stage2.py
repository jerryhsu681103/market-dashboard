"""第 2 關：樣本內置換檢驗。

用法：python -m scripts.stage2 <idea> --timeframe 4h [--n-perm 1000] [--seed 20261114]
做法：
  - 資料、參數網格、最優組合選擇規則（交易次數 ≥ 100 中 Sharpe 最高）與第 1 關完全相同
  - 每條假行情：保持起點終點價格、打亂 K 線漲跌順序；外生特徵（點子使用的新資料集欄位）與資金費率固定在原時間
  - p = (假行情成績 ≥ 真實成績的次數 + 1) / (n_perm + 1)，p ≥ 1% 淘汰
輸出：reports/permutation/<idea>_<symbol>_<tf>_stage2.png 與 .json；更新 research_log.csv 該點子的列。
"""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lab import research
from lab.costs import load_costs
from lab.data import load_feature
from scripts.stage1 import GATE, IDEAS, grid_for, load_idea_data

ROOT = Path(__file__).resolve().parents[1]
PASS_P = 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("idea")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261114)
    a = ap.parse_args()

    spec, tf = IDEAS[a.idea], a.timeframe
    fn = importlib.import_module(f"strategies.{a.idea}").signal
    grid = grid_for(spec, tf)
    fee, slip = load_costs()
    df, funding, symbol, run_fn = load_idea_data(spec, tf, a.symbol)
    fixed = [c for name in spec.get("features", []) for c in
             (load_feature(symbol, name, "1h").columns if name not in ("spot", "premium") else
              [f"{'spot_' if name == 'spot' else 'prem_'}{c}" for c in load_feature(symbol, name, "1h").columns])]

    r = research.permutation_test(fn, grid, df, fee, slip, funding, kind="insample", n_perm=a.n_perm,
                                  seed=a.seed, objective="sharpe", min_trades=GATE["min_trades"],
                                  fixed_cols=fixed, run_fn=run_fn)
    status = "alive_stage2" if r["p_value"] < PASS_P else "dead_stage2"
    perm = r["perm_scores"][np.isfinite(r["perm_scores"])]
    summary = {
        "idea": a.idea, "symbol": symbol, "timeframe": tf, "real_sharpe": round(r["real"], 4),
        "p_value": round(r["p_value"], 5), "n_perm": r["n_perm"], "n_ge_real": r["n_ge_real"],
        "perm_no_eligible_combo": r["n_perm_no_eligible"], "seed": r["seed"],
        "perm_sharpe_mean": round(float(perm.mean()), 4) if len(perm) else None,
        "perm_sharpe_p95": round(float(np.quantile(perm, 0.95)), 4) if len(perm) else None,
        "perm_sharpe_p99": round(float(np.quantile(perm, 0.99)), 4) if len(perm) else None,
        "perm_sharpe_max": round(float(perm.max()), 4) if len(perm) else None,
        "fixed_cols": fixed, "min_trades": r["min_trades"], "status": status,
        "command": f"python -m scripts.stage2 {a.idea} --timeframe {tf} --n-perm {a.n_perm} --seed {a.seed}",
    }
    out = ROOT / "reports" / "permutation"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{a.idea}_{symbol}_{tf}_stage2"
    research.save_permutation_plot(r, out / f"{stem}.png",
                                   title=f"{a.idea} {symbol} {tf}  stage 2  p = {r['p_value']:.4f}")
    (out / f"{stem}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(out / f"{stem}_scores.npy", r["perm_scores"])

    log_path = ROOT / "research_log.csv"
    log = pd.read_csv(log_path, dtype=str)
    m = (log["idea"] == a.idea) & (log["timeframe"] == tf)
    if m.sum() != 1:
        raise RuntimeError(f"research_log 中找不到唯一一列 {a.idea} [{tf}]")
    log.loc[m, "stage2_p"] = f"{r['p_value']:.5f}"
    log.loc[m, "status"] = status
    if status == "dead_stage2":
        log.loc[m, "death_reason"] = f"置換檢驗 p = {r['p_value']:.4f} ≥ {PASS_P}"
    log.loc[m, "scripts"] = log.loc[m, "scripts"] + f" ; {summary['command']}  (reports/permutation/{stem}.json)"
    log.to_csv(log_path, index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
