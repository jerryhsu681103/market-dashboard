"""存活者的加驗：逐年、逐幣、多空拆分、拿掉最賺的幣、費用加倍（凍結參數與 walk-forward 兩種口徑）。

用法：python -m desk.robust near_high_momo@4h donchian_atr@4h
輸出：reports/desk/<日期>/robust_<名稱>.json
"""
import json
import sys

import numpy as np
import pandas as pd

from lab.costs import load_costs

from .common import EVAL_START
from .evaluate import OUT, Book


def main():
    fee, slip = load_costs()
    day = sorted(p.name for p in OUT.iterdir() if p.is_dir())[-1]
    for arg in sys.argv[1:]:
        name, tf = arg.split("@")
        res = json.loads((OUT / day / f"{arg}.json").read_text(encoding="utf-8"))
        bk = Book(name, tf, fee, slip)
        ci = bk.combos.index(res["frozen_params"])
        det = bk.details(bk.pos[ci], EVAL_START)
        st = bk.stats(det, EVAL_START)
        net, tr = st["_net"], st["_trades"]
        yearly = ((1 + net).groupby(net.index.year).prod() - 1).round(4).to_dict()
        by_sym = (tr.groupby("sym")["pnl"].agg(["sum", "count"]) / [len(bk.syms), 1]).round(4)
        by_side = tr.groupby("side")["pnl"].agg(["sum", "count"])
        by_side["sum"] /= len(bk.syms)
        top = by_sym["sum"].idxmax()
        rest = {s: d for s, d in det.items() if s != top}
        st_wo = bk.stats(rest, EVAL_START)  # 仍除以原幣數：等於把最賺的那一份改放現金
        x2 = bk.stats(bk.details(bk.pos[ci], EVAL_START, cost_mult=2.0), EVAL_START)
        out = {
            "name": arg, "frozen_params": res["frozen_params"], "window": [st["start"], st["end"]],
            "yearly_return": yearly,
            "by_symbol": {s: {"pnl_contrib": float(r["sum"]), "trades": int(r["count"])} for s, r in by_sym.iterrows()},
            "by_side": {("long" if k > 0 else "short"): {"pnl_contrib": round(float(r["sum"]), 4), "trades": int(r["count"])}
                        for k, r in by_side.iterrows()},
            "top_symbol": top, "without_top_symbol": {k: round(st_wo[k], 4) for k in ("sharpe", "total_return", "max_drawdown")},
            "frozen_cost_x2_4y": {k: round(x2[k], 4) for k in ("sharpe", "total_return", "max_drawdown")},
            "frozen_4y": {k: round(st[k], 4) for k in ("sharpe", "total_return", "max_drawdown", "trades", "win_rate",
                                                       "profit_factor", "longest_losing_streak", "worst_month", "exposure")},
            "median_trade_pnl_on_slice": float(np.median(tr["pnl"])),
            "avg_holding_bars": float(((tr["exit"] - tr["entry"]) / (bk.master[1] - bk.master[0])).mean() + 1),
        }
        (OUT / day / f"robust_{arg}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
