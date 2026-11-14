"""研究流程工具：參數掃描、參數高原、滾動前推、置換檢驗。

所有數字都經由 engine.run / engine.simulate 產生，這裡只負責調度。
signal_fn 必須定義在模組頂層（strategies/ 底下），才能被多行程平行執行。
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import engine
from .permutation import permute_bars

OBJECTIVES = ("sharpe", "profit_factor", "total_return")


# ---------------------------------------------------------------- 參數掃描

def _combos(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]


def grid_search(signal_fn, grid: dict, df: pd.DataFrame, fee, slippage, funding,
                valid=None, run_fn=None) -> pd.DataFrame:
    """掃描所有參數組合，回傳每組的績效表。valid(params)->bool 可排除不合理組合（例如 fast>=slow）。

    第一組參數會做前視截斷檢查；其餘組合略過以節省時間（同一個信號函式）。
    run_fn：預設 engine.run；價差點子傳 functools.partial(engine.run_spread, legs=...)，funding 傳 {幣種: Series}。
    """
    rows, checked = [], False
    for p in _combos(grid):
        if valid is not None and not valid(p):
            continue
        r = (run_fn or engine.run)(signal_fn, p, df, fee, slippage, funding,
                                   check_lookahead_bias=not checked)
        checked = True
        rows.append({**p, **{k: r[k] for k in ("sharpe", "profit_factor", "total_return",
                                                "max_drawdown", "trade_count")}})
    if not rows:
        raise ValueError("沒有任何有效的參數組合")
    return pd.DataFrame(rows)


def best_params(results: pd.DataFrame, grid: dict, objective: str = "sharpe",
                min_trades: int = 1) -> dict:
    ok = results[results["trade_count"] >= min_trades]
    if ok.empty or ok[objective].isna().all():
        return {}
    row = ok.loc[ok[objective].idxmax()]
    # 取回 grid 中的原始值（DataFrame 可能把 int 轉成 float）
    return {k: next(v for v in grid[k] if v == row[k]) for k in grid}


# ---------------------------------------------------------------- 參數高原

def plateau_check(results: pd.DataFrame, grid: dict, objective: str = "sharpe",
                  ratio: float = 0.6) -> dict:
    """最優點周圍一圈（每個參數 ±1 格，含斜角）的成績中位數 ≥ 最優值 × ratio 才算高原。"""
    keys = list(grid)
    pos = {k: {v: i for i, v in enumerate(grid[k])} for k in keys}
    lookup = {tuple(pos[k][r[k]] for k in keys): r[objective] for _, r in results.iterrows()}
    best_row = results.loc[results[objective].idxmax()]
    center = tuple(pos[k][best_row[k]] for k in keys)
    neigh = []
    for d in itertools.product((-1, 0, 1), repeat=len(keys)):
        if all(x == 0 for x in d):
            continue
        c = tuple(a + b for a, b in zip(center, d))
        if c in lookup:
            v = lookup[c]
            neigh.append(v if np.isfinite(v) else -np.inf)
    best = float(best_row[objective])
    med = float(np.median(neigh)) if neigh else float("nan")
    is_plateau = bool(neigh) and best > 0 and med >= best * ratio
    return {"best": best, "best_params": {k: best_row[k] for k in keys},
            "neighbor_count": len(neigh), "neighbor_median": med,
            "ratio": med / best if best else float("nan"),
            "threshold": ratio, "is_spike": not is_plateau}


def save_heatmap(results: pd.DataFrame, x: str, y: str, path: str | Path,
                 objective: str = "sharpe", title: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    piv = results.pivot_table(index=y, columns=x, values=objective, aggfunc="max")
    fig, ax = plt.subplots(figsize=(1 + 0.6 * piv.shape[1], 1 + 0.45 * piv.shape[0]))
    im = ax.imshow(piv.to_numpy(), cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(piv.shape[1]), [str(v) for v in piv.columns], rotation=45)
    ax.set_yticks(range(piv.shape[0]), [str(v) for v in piv.index])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or objective)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="w")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- 滾動前推

def walk_forward(signal_fn, grid: dict, df: pd.DataFrame, fee, slippage, funding,
                 train_bars: int = 4 * 365 * 24, test_bars: int = 30 * 24,
                 objective: str = "sharpe", valid=None, min_trades: int = 1) -> dict:
    """訓練 train_bars → 交易 test_bars → 往前滾動。

    每個測試窗口用「訓練段 + 測試段」計算信號（讓指標暖機），只取測試段的持倉。
    所有測試段的持倉拼接後，由 engine.simulate 統一計算成本與資金費（窗口交界的換倉也會收費）。
    """
    cost_rate = engine._validate_costs(fee, slippage)
    n = len(df)
    if n <= train_bars:
        raise ValueError("資料長度不足一個訓練窗口")
    engine.check_lookahead(signal_fn, df.iloc[:train_bars],
                           _combos(grid)[0] if valid is None else
                           next(p for p in _combos(grid) if valid(p)))
    pos_parts, windows = [], []
    start = train_bars
    while start < n:
        end = min(start + test_bars, n)
        train = df.iloc[start - train_bars:start]
        f_train = funding[funding.index <= train.index[-1]]
        res = grid_search(signal_fn, grid, train, fee, slippage, f_train, valid=valid)
        params = best_params(res, grid, objective, min_trades)
        seg = df.iloc[start - train_bars:end]
        if params:
            sig = engine.compute_signals(signal_fn, seg, params)
            pos = sig.shift(1).fillna(0).to_numpy(float)[train_bars:]
        else:
            pos = np.zeros(end - start)
        pos_parts.append(pos)
        windows.append({"train_start": str(train.index[0]), "test_start": str(df.index[start]),
                        "test_end": str(df.index[end - 1]), "params": params,
                        f"train_{objective}": float(res[objective].max())})
        start = end

    test = df.iloc[train_bars:]
    pos = np.concatenate(pos_parts)
    o = df["open"].to_numpy(float)
    bar_ret = np.zeros(n)
    bar_ret[:-1] = o[1:] / o[:-1] - 1.0
    fund_end = engine.funding_at_bar_end(test.index, funding)
    detail = engine.simulate(pos, bar_ret[train_bars:], fund_end, cost_rate, test.index)
    out = engine.summarize(detail, engine.infer_bars_per_year(df.index))
    out.update({"windows": windows, "fee": float(fee), "slippage": float(slippage)})
    return out


# ---------------------------------------------------------------- 置換檢驗

def _score(res: pd.DataFrame, objective: str, min_trades: int) -> float:
    """與第 1 關相同的選擇規則：交易次數 ≥ min_trades 的組合中目標值最高者；沒有合格組合則為 -inf。"""
    ok = res[res["trade_count"] >= min_trades][objective]
    ok = ok[np.isfinite(ok)]
    return float(ok.max()) if len(ok) else float("-inf")


def _perm_worker(args):
    (kind, signal_fn, grid, df, fee, slippage, funding, seed, start_index, objective, valid, wf_kw,
     min_trades, fixed_cols, run_fn) = args
    pdf = permute_bars(df, seed=seed, start_index=start_index, fixed_cols=fixed_cols)
    if kind == "insample":
        res = grid_search(signal_fn, grid, pdf, fee, slippage, funding, valid=valid, run_fn=run_fn)
        return _score(res, objective, min_trades)
    return float(walk_forward(signal_fn, grid, pdf, fee, slippage, funding,
                              objective=objective, valid=valid, **wf_kw)[objective])


def permutation_test(signal_fn, grid: dict, df: pd.DataFrame, fee, slippage, funding, *,
                     kind: str = "insample", n_perm: int = 1000, seed: int = 20260101,
                     objective: str = "sharpe", valid=None, wf_kw: dict | None = None,
                     workers: int | None = None, min_trades: int = 1, fixed_cols=(),
                     run_fn=None) -> dict:
    """第 2 關（kind='insample'）或第 4 關（kind='walkforward'）置換檢驗。

    p = (假行情成績 ≥ 真實成績的次數 + 1) / (n_perm + 1)。
    +1 是標準的保守修正，避免 1000 次全輸時得到 p = 0。
    第 4 關只打亂第一個訓練窗口之後的資料（start_index = train_bars）。
    min_trades：真實與假行情都用相同的選擇規則（交易次數 ≥ min_trades 的組合中取最高）。
    fixed_cols：不隨 K 線打亂、固定在原時間上的外生特徵欄位（資金費率 funding_* 一律固定）。
    """
    wf_kw = wf_kw or {}
    fixed_cols = tuple(fixed_cols)
    if kind == "insample":
        real_res = grid_search(signal_fn, grid, df, fee, slippage, funding, valid=valid, run_fn=run_fn)
        real = _score(real_res, objective, min_trades)
        start_index = 0
    elif kind == "walkforward":
        real = float(walk_forward(signal_fn, grid, df, fee, slippage, funding,
                                  objective=objective, valid=valid, **wf_kw)[objective])
        start_index = wf_kw.get("train_bars", 4 * 365 * 24)
    else:
        raise ValueError("kind 只能是 'insample' 或 'walkforward'")

    seeds = [seed + i for i in range(n_perm)]
    jobs = [(kind, signal_fn, grid, df, fee, slippage, funding, s, start_index, objective, valid, wf_kw,
             min_trades, fixed_cols, run_fn) for s in seeds]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        perm = np.array(list(ex.map(_perm_worker, jobs, chunksize=4)), float)
    perm = np.where(np.isfinite(perm), perm, -np.inf)
    k = int((perm >= real).sum())
    return {"real": real, "perm_scores": perm, "n_perm": n_perm, "n_ge_real": k,
            "p_value": (k + 1) / (n_perm + 1), "seed": seed, "kind": kind, "objective": objective,
            "min_trades": min_trades, "fixed_cols": list(fixed_cols),
            "n_perm_no_eligible": int(np.isneginf(perm).sum())}


def save_permutation_plot(result: dict, path: str | Path, title: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    perm = result["perm_scores"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(perm[np.isfinite(perm)], bins=50, color="#7f8c8d")
    ax.axvline(result["real"], color="#c0392b", lw=2, label=f"real = {result['real']:.3f}")
    ax.set_xlabel(result["objective"])
    ax.set_ylabel("count")
    ax.set_title(title or f"{result['kind']} permutation  p = {result['p_value']:.4f}")
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- 多重檢定

def stage4_threshold(log_path: str | Path, years_forward: float) -> dict:
    """第 4 關門檻 = 基本門檻 ÷ 累計進入第 4 關的點子數（Bonferroni）。"""
    base = 0.05 if years_forward < 2 else 0.01
    n = 1
    p = Path(log_path)
    if p.exists():
        log = pd.read_csv(p)
        if "stage4_p" in log.columns:
            n = max(1, int(log["stage4_p"].notna().sum()) + 1)  # +1 = 本次
    return {"base": base, "n_tested_stage4": n, "threshold": base / n}
