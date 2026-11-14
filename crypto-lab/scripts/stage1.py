"""第 1 關：樣本內參數掃描 + 參數高原 + 門檻判定。

用法：python -m scripts.stage1 donchian_trend funding_fade liq_reversal [--symbol BTCUSDT] [--timeframe 4h]
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
        "bar_params": ["entry"],
        "grid": {"entry": [72, 120, 168, 240, 336, 480, 720], "exit_ratio": [0.25, 0.5, 0.75, 1.0]},
        "rationale": "大級別突破後的跟隨：晚到資金追價、空方被迫回補",
    },
    "funding_fade": {
        "bar_params": ["hold"],
        "grid": {"threshold": [0.0002, 0.0003, 0.0005, 0.0008, 0.001], "hold": [8, 24, 48, 72, 120]},
        "rationale": "資金費率極端時反向：擁擠槓桿去化，且反向持倉收取資金費",
    },
    "liq_reversal": {
        "bar_params": ["hold"],
        "grid": {"k": [3.0, 3.5, 4.0, 5.0, 6.0], "hold": [3, 6, 12, 24, 48]},
        "rationale": "清算瀑布後反彈：強平造成的非資訊性賣壓被推過頭",
    },
    "taker_imbalance": {
        "bar_params": ["hold"],
        "grid": {"threshold": [1.5, 2.0, 2.5, 3.0, 3.5], "hold": [1, 3, 6, 12, 24]},
        "rationale": "主動買賣失衡延續：急迫的吃單方帶有資訊，被動掛單方被吃",
    },
    "weekend_drift_revert": {
        "bar_params": ["hold"],
        "grid": {"threshold": [0.01, 0.02, 0.03, 0.05, 0.08], "hold": [6, 12, 24, 48, 72]},
        "rationale": "週末波動回歸：週末流動性差被推動的價格，週一流動性回來後修正",
    },
    "session_seasonality": {
        "bar_params": [],
        "grid_by_tf": {"4h": {"start": [0, 4, 8, 12, 16, 20], "length": [4, 8, 12]}},
        "grid": {"start": list(range(0, 24, 2)), "length": [2, 4, 6, 8, 12]},
        "rationale": "時段季節性：時段間資金流的結構性差異造成固定時段漂移（只做多）",
    },
    "volume_climax": {
        "bar_params": ["lookback"],
        "grid": {"lookback": [72, 168, 336, 720], "k": [1.5, 2.0, 2.5, 3.0, 4.0]},
        "rationale": "量能高潮反轉：累積偏離加量能創新高，追價資金集中出清後回落",
    },
    "trend_pullback": {
        "bar_params": ["trend_len"],
        "grid": {"trend_len": [240, 480, 720, 1200, 1680], "threshold": [1.0, 1.5, 2.0, 2.5, 3.0]},
        "rationale": "趨勢中的回檔進場：趨勢內短線恐慌停損的弱手，賺回檔修復",
    },
    "lowvol_meanrev": {
        "bar_params": [],
        "grid": {"threshold": [1.5, 2.0, 2.5, 3.0, 3.5], "vol_pct": [0.2, 0.3, 0.4, 0.5, 0.6]},
        "rationale": "低波動環境短線均值回歸：平靜時期的偏離多為流動性需求造成",
    },
    "btc_eth_lead_lag": {
        "bar_params": ["hold"],
        "grid": {"threshold": [0.005, 0.01, 0.015, 0.02, 0.03], "hold": [1, 3, 6, 12, 24]},
        "rationale": "BTC 領先 ETH：ETH 價格發現落後，買入（賣出）尚未跟上的 ETH",
        "symbol": "ETHUSDT",
        "attach": ["BTCUSDT"],
    },
}

def grid_for(spec: dict, timeframe: str) -> dict:
    """以 1 小時線網格為準，機械換算到其他週期：以「根數」計的參數除以週期小時數（四捨五入、至少 1、去重）。

    換算規則在看到結果之前就固定，不針對新週期另外調參。
    """
    if timeframe == "1h":
        return spec["grid"]
    if timeframe in spec.get("grid_by_tf", {}):
        return spec["grid_by_tf"][timeframe]
    h = int(timeframe.rstrip("h"))
    out = {}
    for k, vals in spec["grid"].items():
        if k in spec["bar_params"]:
            vals = list(dict.fromkeys(max(1, int(round(v / h))) for v in vals))
        out[k] = vals
    return out


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
    if plateau["is_spike"] and plateau["best"] > 0:  # 最優值 ≤ 0 時高原比沒有意義
        fails.append(f"參數尖峰：鄰近中位數 / 最優 = {plateau['ratio']:.2f} < {plateau['threshold']}")
    if fails:
        return "dead", "；".join(fails)
    return "pass", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ideas", nargs="+")
    ap.add_argument("--symbol", default="BTCUSDT")  # 點子未指定 symbol 時使用
    ap.add_argument("--timeframe", default="1h")
    a = ap.parse_args()
    fee, slip = load_costs()
    today = datetime.now(timezone.utc).date().isoformat()
    log_path = ROOT / "research_log.csv"
    n_prev = len(pd.read_csv(log_path))

    for i, name in enumerate(a.ideas, 1):
        spec = IDEAS[name]
        fn = importlib.import_module(f"strategies.{name}").signal
        grid = grid_for(spec, a.timeframe)
        tf = a.timeframe
        sfx = "" if tf == "1h" else f"_{tf}"
        symbol = spec.get("symbol", a.symbol)
        df, funding = load_insample(symbol, tf)
        df = attach_funding(df, funding)
        for other in spec.get("attach", []):
            odf, _ = load_insample(other, tf)
            if not odf.index.equals(df.index):
                raise ValueError(f"{other} 與 {symbol} 時間索引不一致")
            df[f"{other}_close"] = odf["close"]
        res = research.grid_search(fn, grid, df, fee, slip, funding)
        out_csv = ROOT / "reports" / "stage1" / f"{name}_{symbol}{sfx}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out_csv, index=False)

        x, y = list(grid)
        hm = research.save_heatmap(res, x, y, ROOT / "reports" / "heatmaps" / f"{name}_{symbol}{sfx}_stage1.png",
                                   title=f"{name} {symbol} {tf} in-sample Sharpe")
        eligible = res[res["trade_count"] >= GATE["min_trades"]]
        best = eligible.loc[eligible["sharpe"].idxmax()] if not eligible.empty else None
        plateau = research.plateau_check(eligible if not eligible.empty else res, grid)
        status, reason = judge(best, plateau)

        summary = {"symbol": symbol, "timeframe": tf, "best_params": {k: next(v for v in grid[k] if v == best[k]) for k in grid} if best is not None else None,
                   "sharpe": None if best is None else round(float(best["sharpe"]), 4),
                   "pf": None if best is None else round(float(best["profit_factor"]), 4),
                   "total_return": None if best is None else round(float(best["total_return"]), 4),
                   "max_dd": None if best is None else round(float(best["max_drawdown"]), 4),
                   "trades": None if best is None else int(best["trade_count"]),
                   "plateau_ratio": round(plateau["ratio"], 4)}
        print(f"\n== {name} [{tf}]  →  {status}  {reason}")
        print(json.dumps(summary, ensure_ascii=False, default=float))

        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                today, name, spec["rationale"], json.dumps(grid), json.dumps(summary, ensure_ascii=False, default=float),
                "", "", "", "", str(hm.relative_to(ROOT)),
                {"pass": "alive_stage1", "dead": "dead_stage1", "review": "review_stage1"}[status],
                reason, n_prev + i,
                f"python -m scripts.stage1 {name} --symbol {symbol} --timeframe {tf}  ({out_csv.relative_to(ROOT)})",
                tf,
            ])


if __name__ == "__main__":
    main()
