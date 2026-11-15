"""驗偽：12 幣等權組合回測 + walk-forward + 費用加倍 + 熊市/震盪檢查 + 五項打分。

用法：
    python -m desk.evaluate                 # 跑 candidates.TESTS 全部
    python -m desk.evaluate tsmom_ls@4h     # 只跑指定的

數字來源：
    - 每個幣每組參數的訊號與持倉：lab.engine.run（含前視截斷檢查）
    - 逐根損益（手續費、滑點、資金費）：lab.engine.simulate
    - 組合層的 Sharpe／回撤／報酬：lab.engine.summarize（組合逐根淨值 = 各幣淨損益 ÷ 幣數）
    - 參數高原：lab.research.plateau_check
    這個檔案只做「切片、拼接、加總、計數」，不另寫損益算法。

組合假設：本金等分給幣池每個幣（各 1/N，1 倍名目、不加槓桿）；沒訊號或已下市的那一份放現金。
死拿基準：同一幣池、同樣成本與資金費、每個幣一直做多（下市即出場）。

淘汰線（看到結果前寫定）：
    G0 前視截斷檢查不過
    G1 近 4 年最優組合：Sharpe < 0.5 或 交易 < 50
    G2 參數尖峰（最優點鄰居中位數 < 最優的 60%）或 參數面太窄（Sharpe > 0 的組合不到 2/3）
    G3 walk-forward（訓練 2 年→交易 91 天滾動，全是沒見過的資料）：Sharpe < 0.5 或 報酬 ≤ 0 或 交易 < 50
    G4 walk-forward 期間跑不贏死拿：Sharpe ≤ 死拿 或 最大回撤比死拿深
    G5 手續費+滑點加倍後：walk-forward Sharpe < 0.25 或 報酬 ≤ 0
    G6 熊市或震盪市年化報酬 < −20%（撐不住）
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from lab import engine
from lab.costs import load_costs
from lab.research import _combos, plateau_check

from .candidates import ALT, C, TESTS
from .common import ALTS, EVAL_START, SYMBOLS, load_universe, sma

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "desk"
TRAIN = pd.DateOffset(years=2)
TEST_DAYS = 91
MIN_TRAIN_TRADES = 30


@lru_cache(maxsize=2)
def universe(tf: str):
    return load_universe(tf)


def trade_list(detail: pd.DataFrame) -> pd.DataFrame:
    """逐筆交易（與 engine._trade_pnls 相同定義：持倉段毛利 + 資金費 − 進出場成本）。"""
    p = detail["pos"].to_numpy()
    n = len(p)
    if n == 0:
        return pd.DataFrame(columns=["entry", "exit", "side", "pnl"])
    starts = np.flatnonzero(np.diff(np.concatenate([[0.0], p])) != 0)
    if p[0] != 0 and (len(starts) == 0 or starts[0] != 0):
        starts = np.concatenate([[0], starts])
    ends = np.concatenate([starts[1:] - 1, [n - 1]])
    cs = np.concatenate([[0.0], np.cumsum(detail["gross"].to_numpy() + detail["funding"].to_numpy())])
    ec, xc = detail["entry_cost"].to_numpy(), detail["exit_cost"].to_numpy()
    rows = []
    for a, b in zip(starts, ends):
        if p[a] == 0:
            continue
        pnl = cs[b + 1] - cs[a] - ec[a] - (xc[b + 1] if b + 1 < n else 0.0)
        rows.append((detail.index[a], detail.index[b], int(np.sign(p[a])), pnl))
    return pd.DataFrame(rows, columns=["entry", "exit", "side", "pnl"])


def regimes(tf: str) -> pd.Series:
    """以 BTC 判定市場狀態（只用於事後分段統計，不進訊號）。"""
    btc = universe(tf)["BTCUSDT"][0]
    per_day = int(round(pd.Timedelta("1D") / pd.Series(btc.index).diff().median()))
    c = btc["close"]
    s200, r60 = sma(c, 200 * per_day), c / c.shift(60 * per_day) - 1
    lab = pd.Series("range", index=btc.index)
    lab[(c > s200) & (r60 > 0.10)] = "bull"
    lab[(c < s200) & (r60 < -0.10)] = "bear"
    return lab


class Book:
    """一個候選在一個週期上的全部持倉與損益材料。"""

    def __init__(self, name: str, tf: str, fee: float, slip: float):
        cfg = C[name]
        self.name, self.tf, self.cfg = name, tf, cfg
        self.cost = fee + slip
        self.syms = ALTS if cfg["universe"] == ALT else SYMBOLS
        uni = universe(tf)
        self.df = {s: uni[s][0] for s in self.syms}
        self.master = uni["BTCUSDT"][0].index
        self.master = self.master[self.master >= EVAL_START]
        self.bpy = engine.infer_bars_per_year(self.master)
        self.bar_ret, self.fund = {}, {}
        for s in self.syms:
            d, f = uni[s]
            o = d["open"].to_numpy(float)
            br = np.zeros(len(o))
            br[:-1] = o[1:] / o[:-1] - 1.0
            self.bar_ret[s], self.fund[s] = br, engine.funding_at_bar_end(d.index, f)
        self.combos = _combos(cfg["grid"])
        self.pos = []  # [combo][sym] -> ndarray
        for ci, p in enumerate(self.combos):
            row = {}
            for s in self.syms:
                d, f = uni[s]
                r = engine.run(cfg["fn"], p, d, fee, slip, f, check_lookahead_bias=(ci == 0),
                               keep_detail=True)
                row[s] = r["detail"]["pos"].to_numpy(float)
            self.pos.append(row)

    def details(self, pos_by_sym: dict, start, end=None, cost_mult: float = 1.0) -> dict:
        out = {}
        for s in self.syms:
            idx = self.df[s].index
            d = engine.simulate(pos_by_sym[s], self.bar_ret[s], self.fund[s],
                                self.cost * cost_mult, idx)
            m = idx >= start
            if end is not None:
                m &= idx < end
            out[s] = d[m]
        return out

    def stats(self, details: dict, start, end=None) -> dict:
        idx = self.master[self.master >= start]
        if end is not None:
            idx = idx[idx < end]
        cols = ["gross", "funding", "entry_cost", "exit_cost", "net"]
        tot = pd.DataFrame(0.0, index=idx, columns=cols)
        expo = pd.Series(0.0, index=idx)
        for s, d in details.items():
            tot = tot.add(d[cols].reindex(idx).fillna(0.0))
            expo = expo.add(d["pos"].abs().reindex(idx).fillna(0.0))
        tot /= len(self.syms)
        tot.insert(0, "pos", 0.0)
        sm = engine.summarize(tot, self.bpy)
        tr = pd.concat([trade_list(d).assign(sym=s) for s, d in details.items()], ignore_index=True)
        tr = tr.sort_values("exit").reset_index(drop=True)
        pnl = tr["pnl"].to_numpy()
        wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
        streak = best = 0
        for x in pnl:
            streak = streak + 1 if x < 0 else 0
            best = max(best, streak)
        net = sm["returns"]
        monthly = (1 + net).groupby(net.index.strftime("%Y-%m")).prod() - 1
        return {
            "sharpe": sm["sharpe"], "total_return": sm["total_return"], "cagr": sm["cagr"],
            "max_drawdown": sm["max_drawdown"], "trades": int(len(tr)),
            "profit_factor": float(wins / losses) if losses > 0 else float("nan"),
            "win_rate": float((pnl > 0).mean()) if len(pnl) else float("nan"),
            "longest_losing_streak": int(best),
            "worst_month": float(monthly.min()) if len(monthly) else float("nan"),
            "worst_month_label": str(monthly.idxmin()) if len(monthly) else "",
            "exposure": float(expo.mean() / len(self.syms)),
            "cost_paid": float((tot["entry_cost"] + tot["exit_cost"]).sum()),
            "funding_pnl": float(tot["funding"].sum()),
            "start": str(idx[0]), "end": str(idx[-1]),
            "_net": net, "_trades": tr,
        }


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _ann_by_regime(net: pd.Series, reg: pd.Series, bpy: float) -> dict:
    r = reg.reindex(net.index).fillna("range")
    return {k: {"ann_return": float(net[r == k].mean() * bpy) if (r == k).any() else None,
                "share": float((r == k).mean())} for k in ("bull", "bear", "range")}


def score(wf: dict, reg_final: dict) -> dict:
    def band(x, cuts):  # cuts 遞增，回傳 1..5
        return 1 + sum(x >= c for c in cuts)
    worst_reg = min(v["ann_return"] for k, v in reg_final.items()
                    if k in ("bear", "range") and v["ann_return"] is not None)
    return {
        "walk_forward": band(wf["sharpe"], [0, 0.5, 1.0, 1.5]),
        "max_drawdown": band(wf["max_drawdown"], [-0.40, -0.30, -0.20, -0.10]),
        "trade_count": band(wf["trades"], [50, 100, 200, 400]),
        "losing_streak": 6 - band(wf["longest_losing_streak"], [5, 8, 11, 15]),
        "bear_and_range": band(worst_reg, [-0.20, -0.05, 0.05, 0.15]),
    }


def evaluate(name: str, tf: str) -> dict:
    fee, slip = load_costs()
    cfg = C[name]
    res = {"name": name, "timeframe": tf, "family": cfg["family"], "regime": cfg["regime"],
           "why": cfg["why"], "payer": cfg["payer"], "overlap": cfg["overlap"],
           "universe": cfg["universe"], "grid": cfg["grid"], "fee": fee, "slippage": slip}
    try:
        bk = Book(name, tf, fee, slip)
    except engine.LookaheadError as e:
        res.update(status="dead", gate="G0", death=f"前視偏差：{e}")
        return res
    end = bk.master[-1] + (bk.master[1] - bk.master[0])
    res["symbols"] = len(bk.syms)

    # ---- 近 4 年：每組參數
    full, rows = [], []
    for p, pos in zip(bk.combos, bk.pos):
        st = bk.stats(bk.details(pos, EVAL_START), EVAL_START)
        full.append(st)
        rows.append({**p, "sharpe": st["sharpe"], "total_return": st["total_return"],
                     "max_drawdown": st["max_drawdown"], "trade_count": st["trades"]})
    table = pd.DataFrame(rows)
    res["grid_table"] = rows
    ok = table[table["trade_count"] >= 50]
    res["breadth_positive"] = float((table["sharpe"] > 0).mean())
    bi = int(ok["sharpe"].idxmax()) if len(ok) and ok["sharpe"].notna().any() else int(table["sharpe"].fillna(-9).idxmax())
    res["full_best"] = {"params": bk.combos[bi], **_clean(full[bi])}
    pl = plateau_check(table.fillna({"sharpe": -9.0}), cfg["grid"])
    res["plateau"] = {"ratio": pl["ratio"], "is_spike": pl["is_spike"], "neighbor_median": pl["neighbor_median"]}

    # ---- 死拿基準
    hold = {s: np.ones(len(bk.df[s])) for s in bk.syms}
    res["hold_full"] = _clean(bk.stats(bk.details(hold, EVAL_START), EVAL_START))

    # ---- walk-forward
    wf_start = EVAL_START + TRAIN
    wf_pos = {s: np.zeros(len(bk.df[s])) for s in bk.syms}
    windows, ws = [], wf_start
    while ws < end:
        we = min(ws + pd.Timedelta(days=TEST_DAYS), end)
        ts = ws - TRAIN
        best_ci, best_sh = None, -np.inf
        for ci, st in enumerate(full):
            net = st["_net"]
            seg = net[(net.index >= ts) & (net.index < ws)]
            tr = st["_trades"]
            ntr = int(((tr["exit"] >= ts) & (tr["exit"] < ws)).sum())
            sd = seg.std(ddof=1)
            sh = seg.mean() / sd * np.sqrt(bk.bpy) if sd > 0 else -np.inf
            if ntr >= MIN_TRAIN_TRADES and sh > best_sh:
                best_ci, best_sh = ci, sh
        if best_ci is not None:
            for s in bk.syms:
                idx = bk.df[s].index
                m = (idx >= ws) & (idx < we)
                wf_pos[s][m] = bk.pos[best_ci][s][m]
        windows.append({"test_start": str(ws), "test_end": str(we),
                        "params": bk.combos[best_ci] if best_ci is not None else None,
                        "train_sharpe": None if best_ci is None else float(best_sh)})
        ws = we
    wf = bk.stats(bk.details(wf_pos, wf_start), wf_start)
    wf2 = bk.stats(bk.details(wf_pos, wf_start, cost_mult=2.0), wf_start)
    res["wf"], res["wf_cost_x2"], res["wf_windows"] = _clean(wf), _clean(wf2), windows
    res["hold_wf"] = _clean(bk.stats(bk.details(hold, wf_start), wf_start))

    # ---- 凍結參數：用最後 2 年（到資料末端）選出的那一組，之後紙面帳用這組、不再調
    ts = end - TRAIN
    cand = []
    for ci, st in enumerate(full):
        net = st["_net"]
        seg = net[net.index >= ts]
        tr = st["_trades"]
        sd = seg.std(ddof=1)
        if int((tr["exit"] >= ts).sum()) >= MIN_TRAIN_TRADES and sd > 0:
            cand.append((seg.mean() / sd, ci))
    fi = max(cand)[1] if cand else bi
    res["frozen_params"] = bk.combos[fi]
    res["frozen_full"] = _clean(full[fi])

    # ---- 市場狀態
    reg = regimes(tf)
    res["regime_wf"] = _ann_by_regime(wf["_net"], reg, bk.bpy)
    res["regime_frozen_4y"] = _ann_by_regime(full[fi]["_net"], reg, bk.bpy)
    res["scores"] = score(wf, res["regime_frozen_4y"])
    res["score_total"] = int(sum(res["scores"].values()))

    # ---- 淘汰判定（依序，掛在第一關就記那一關）
    fb, hw = res["full_best"], res["hold_wf"]
    worst_reg = min(v["ann_return"] for k, v in res["regime_frozen_4y"].items()
                    if k in ("bear", "range") and v["ann_return"] is not None)
    checks = [
        ("G1", not (fb["sharpe"] >= 0.5 and fb["trades"] >= 50),
         f"近 4 年最優 Sharpe {fb['sharpe']:.2f}、交易 {fb['trades']} 筆，不到 0.5／50 筆"),
        ("G2", pl["is_spike"] or res["breadth_positive"] < 2 / 3,
         f"參數面不穩：鄰居/最優 {pl['ratio']:.2f}，Sharpe>0 的組合佔 {res['breadth_positive']:.0%}"),
        ("G3", not (wf["sharpe"] >= 0.5 and wf["total_return"] > 0 and wf["trades"] >= 50),
         f"walk-forward Sharpe {wf['sharpe']:.2f}、報酬 {wf['total_return']:.1%}、交易 {wf['trades']} 筆"),
        ("G4", not (wf["sharpe"] > hw["sharpe"] and wf["max_drawdown"] >= hw["max_drawdown"]),
         f"跑不贏死拿：Sharpe {wf['sharpe']:.2f} 對 {hw['sharpe']:.2f}，回撤 {wf['max_drawdown']:.1%} 對 {hw['max_drawdown']:.1%}"),
        ("G5", not (wf2["sharpe"] >= 0.25 and wf2["total_return"] > 0),
         f"費用加倍後 walk-forward Sharpe {wf2['sharpe']:.2f}、報酬 {wf2['total_return']:.1%}"),
        ("G6", worst_reg < -0.20, f"熊市/震盪年化最差 {worst_reg:.1%}"),
    ]
    res["gates"] = {g: (not bad) for g, bad, _ in checks}
    dead = next(((g, why) for g, bad, why in checks if bad), None)
    res.update(status="dead" if dead else "alive", gate=dead[0] if dead else "",
               death=dead[1] if dead else "")
    return res


def _job(arg):
    name, tf = arg
    try:
        r = evaluate(name, tf)
    except Exception as e:  # 留下問題，繼續做剩下的
        r = {"name": name, "timeframe": tf, "status": "error", "gate": "ERR", "death": repr(e)}
    return r


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    tests = [tuple(a.split("@")) for a in argv] if argv else TESTS
    day = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    out = OUT / day
    out.mkdir(parents=True, exist_ok=True)
    tests = sorted(tests, key=lambda t: t[1])  # 同週期排一起，共用資料快取
    with Pool(4) as pool:
        for r in pool.imap_unordered(_job, tests, chunksize=1):
            (out / f"{r['name']}@{r['timeframe']}.json").write_text(
                json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            wf = r.get("wf", {})
            print(f"{r['name']}@{r['timeframe']:3s} {r['status']:5s} {r.get('gate', ''):3s} "
                  f"wfSharpe={wf.get('sharpe', float('nan')):.2f} {r.get('death', '')}", flush=True)


if __name__ == "__main__":
    main()
