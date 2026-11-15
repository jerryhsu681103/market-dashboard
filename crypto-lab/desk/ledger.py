"""紙面帳：只記帳、不下單。沒有任何交易所連線、金鑰或下單程式碼。

用法：
    python -m desk.ledger            # 讀已下載的行情，把新的紙面單接著記到日誌
    python -m desk.ledger --resume   # 只有使用者說可以之後才用：解除停記，從日誌接著記

檔案（paper/desk/）：
    journal.jsonl   只增不改的日誌，每筆事件一行（ENTRY／EXIT／HALT／RESUME）。中斷後從這裡接著記。
    ledger.csv      由日誌整理出的逐筆交易表（可重建）
    status.json     目前權益、未平倉、下一根待成交、值守異常
    HALT            存在就代表已停記，等使用者處理

記帳規則：
    - 每個策略一本獨立的紙面帳，本金見 config/desk.json；本金等分 12 份（每幣一份、1 倍名目、不加槓桿）。
      若初始止損距離會讓單筆虧損超過本金 1%，該筆名目等比例縮小（回測沒有這個上限，紙面帳較保守）。
    - 訊號與損益一律由 lab.engine.run 產生：第 t 根收盤的訊號，在第 t+1 根開盤價記成交，含手續費、滑點、資金費。
    - paper_start 那一刻策略已經持有的部位不追記（那不是按卡片規則的新進場）；等它出場後的下一次新訊號才開始記。
    - 行情來源：data.binance.vision 的 USDT 永續 1 小時 K 線日檔（合成 4 小時線），約落後一天。
      資金費月檔下個月才公布，未公布期間資金費暫記 0，status.json 會標明。
值守（任何一條成立就寫 HALT 並停記）：
    行情停更／時間戳過期、K 線缺漏、單日紙面回撤 ≤ −3%、連續 3 筆虧損出場、帳本與引擎重算對不上。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from lab import engine
from lab.costs import load_costs

from .candidates import C
from .common import SYMBOLS, atr, bars, load_universe
from .evaluate import trade_list

ROOT = Path(__file__).resolve().parents[1]
CFG = Path(os.environ.get("DESK_CFG", ROOT / "config" / "desk.json"))   # 環境變數只供演練用
OUT = Path(os.environ.get("DESK_OUT", ROOT / "paper" / "desk"))
JOURNAL, HALT = OUT / "journal.jsonl", OUT / "HALT"


def explain(name: str, params: dict, df: pd.DataFrame, t) -> dict:
    """第 t 根（訊號根）收盤時的規則水位，寫進每一筆的理由。"""
    r = df.loc[:t]
    c = float(r["close"].iloc[-1])
    if name == "donchian_atr":
        n = bars(df, params["entry_h"])
        a = float(atr(r, 14).iloc[-1])
        return {"close": c, "channel_high": float(r["high"].iloc[-n - 1:-1].max()),
                "channel_low": float(r["low"].iloc[-n - 1:-1].min()),
                "exit_high": float(r["high"].iloc[-n // 2 - 1:-1].max()),
                "exit_low": float(r["low"].iloc[-n // 2 - 1:-1].min()),
                "atr14": a, "stop_distance": params["stop_k"] * a, "stop_frac": params["stop_k"] * a / c}
    if name == "near_high_momo":
        top = float(r["high"].iloc[-bars(df, params["look_h"]):].max())
        line = (1 - params["prox"]) * top
        return {"close": c, "high_60d": top, "line": line, "stop_frac": max(c - line, 0.0) / c}
    return {"close": c}


def reason(name: str, kind: str, side: int, lv: dict) -> str:
    if name == "donchian_atr":
        if kind == "ENTRY":
            return (f"收盤 {lv['close']:.6g} " + ("突破 40 日高 %.6g" % lv["channel_high"] if side > 0 else "跌破 40 日低 %.6g" % lv["channel_low"])
                    + f"；初始止損距離 4×ATR = {lv['stop_distance']:.6g}（收盤價判斷、隨最佳收盤價移動）")
        hit = lv["close"] < lv["exit_low"] if side > 0 else lv["close"] > lv["exit_high"]
        return "收盤穿越 20 日反向通道，出場" if hit else "收盤觸及 4×ATR 移動止損，出場"
    if name == "near_high_momo":
        if kind == "ENTRY":
            return f"收盤 {lv['close']:.6g} ≥ 60 日高點 {lv['high_60d']:.6g} 的 97%（{lv['line']:.6g}）"
        return f"收盤 {lv['close']:.6g} 跌回 60 日高點 97% 線（{lv['line']:.6g}）之下，出場"
    return ""


def read_journal() -> list[dict]:
    if not JOURNAL.exists():
        return []
    return [json.loads(x) for x in JOURNAL.read_text(encoding="utf-8").splitlines() if x.strip()]


def append(events: list[dict]) -> None:
    with JOURNAL.open("a", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def build(cfg: dict, uni: dict, fee, slip):
    """由引擎重算每本帳自 paper_start 起的事件與逐根權益。"""
    t0 = pd.Timestamp(cfg["paper_start_utc"])
    notional = cfg["capital_usdt"] / len(SYMBOLS)
    max_loss = cfg["capital_usdt"] * cfg["max_loss_per_trade_pct"]
    events, equity, open_pos, pending, watching = [], {}, [], [], []
    for st in cfg["strategies"]:
        name, params = st["name"], st["params"]
        net_usdt = None
        for s in SYMBOLS:
            df, funding = uni[s]
            if df.index[-1] < t0 - pd.Timedelta(days=5):
                continue  # 已下市
            r = engine.run(C[name]["fn"], params, df, fee, slip, funding, keep_detail=True)
            full = r["detail"]
            d = full[full.index >= t0]
            before = int(full["pos"][full.index < t0].iloc[-1])  # paper_start 前一根的持倉
            sig_next = int(r["signals"].iloc[-1])
            cur = int(full["pos"].iloc[-1])
            trades = trade_list(d) if len(d) else trade_list(full.iloc[:0])
            # paper_start 前就已持有、一路延續下來的那一段：不是按規則的新進場，不追記
            carried_end = None
            if len(d) and before != 0 and len(trades) and trades["entry"].iloc[0] == d.index[0] \
                    and trades["side"].iloc[0] == before:
                carried_end = trades["exit"].iloc[0]
                trades = trades.iloc[1:]
            still_carried = (len(d) == 0 and cur != 0) or (carried_end is not None and carried_end == d.index[-1])
            if still_carried:
                watching.append({"strategy": name, "symbol": s, "side": cur,
                                 "note": "起記前已成立的持倉訊號，不追記；等它出場後的下一次新訊號才記"})
            lv = explain(name, params, df, df.index[-1])
            if sig_next != cur:
                if cur != 0 and not still_carried:
                    pending.append({"strategy": name, "symbol": s, "action": "EXIT", "side": cur,
                                    "reason": reason(name, "EXIT", cur, lv)})
                if sig_next != 0:
                    pending.append({"strategy": name, "symbol": s, "action": "ENTRY", "side": sig_next,
                                    "levels": lv, "reason": reason(name, "ENTRY", sig_next, lv)})
            if len(d) == 0:
                continue
            net = d["net"].copy()
            if carried_end is not None:
                net[net.index <= carried_end] = 0.0
                nxt = d.index[d.index > carried_end]
                if len(nxt):
                    net[nxt[0]] += d["exit_cost"][nxt[0]]
            opens = df["open"]
            scale = pd.Series(1.0, index=d.index)
            for _, tr in trades.iterrows():
                sig_bar = df.index[df.index.get_loc(tr["entry"]) - 1]
                lv = explain(name, params, df, sig_bar)
                # 單筆上限：初始止損被打到時的虧損不超過本金的 max_loss_per_trade_pct，超過就縮小這一筆
                k = min(1.0, max_loss / (notional * lv["stop_frac"])) if lv.get("stop_frac", 0) > 0 else 1.0
                scale[(d.index >= tr["entry"]) & (d.index <= tr["exit"])] = k
                nxt = d.index[d.index > tr["exit"]]
                if len(nxt):
                    net[nxt[0]] += d["exit_cost"][nxt[0]] * (1.0 - k)  # 出場成本也按縮小後的部位算
                tr = tr.copy()
                tr["pnl"] *= k
                base = {"strategy": name, "role": st["role"], "symbol": s, "side": int(tr["side"]),
                        "notional_usdt": round(notional * k, 2)}
                events.append({**base, "type": "ENTRY", "time": tr["entry"].isoformat(),
                               "price": float(opens[tr["entry"]]), "levels": lv,
                               "reason": reason(name, "ENTRY", int(tr["side"]), lv),
                               "id": f"{name}|{s}|{tr['entry'].isoformat()}|ENTRY"})
                loc = df.index.get_loc(tr["exit"])
                if loc + 1 < len(df):  # 已平倉：出場價 = 下一根開盤
                    xt = df.index[loc + 1]
                    lvx = explain(name, params, df, tr["exit"])
                    events.append({**base, "type": "EXIT", "time": xt.isoformat(), "price": float(opens[xt]),
                                   "pnl_usdt": round(float(tr["pnl"]) * notional, 4),
                                   "entry_time": tr["entry"].isoformat(),
                                   "reason": reason(name, "EXIT", int(tr["side"]), lvx),
                                   "id": f"{name}|{s}|{xt.isoformat()}|EXIT"})
                else:
                    open_pos.append({**base, "entry_time": tr["entry"].isoformat(),
                                     "entry_price": float(opens[tr["entry"]]),
                                     "last_close": float(df["close"].iloc[-1]),
                                     "pnl_usdt_so_far": round(float(tr["pnl"]) * notional, 4)})
            flat_after = (d["pos"] == 0)
            x = (net.where(flat_after, net * scale)) * notional  # 空手那幾根只剩（已調整的）出場成本
            net_usdt = x if net_usdt is None else net_usdt.add(x, fill_value=0.0)
        equity[name] = (net_usdt.sort_index().cumsum() if net_usdt is not None else pd.Series(dtype=float))
    events.sort(key=lambda e: (e["time"], e["id"]))
    return events, equity, open_pos, pending, watching


def watch(cfg: dict, uni: dict, events: list[dict], equity: dict, since: pd.Timestamp, now: pd.Timestamp):
    """回傳 (停記時間或 None, 停記原因, 其他異常列表)。只檢查 since 之後的紀錄。"""
    w, notes, halts = cfg["watch"], [], []
    cap = cfg["capital_usdt"]
    last_end = max(d.index[-1] for d, _ in uni.values()) + pd.Timedelta(hours=4)
    age_h = (now - last_end).total_seconds() / 3600
    if age_h > w["stale_hours_halt"]:
        halts.append((last_end, f"行情停更：最後一根收盤於 {last_end}，距今 {age_h:.0f} 小時"))
    t0 = pd.Timestamp(cfg["paper_start_utc"])
    for s, (d, f) in uni.items():
        if d.index[-1] < t0:
            continue
        seg = d.index[d.index >= t0 - pd.Timedelta(days=3)]
        gaps = pd.Series(seg).diff().dropna()
        if (gaps != pd.Timedelta(hours=4)).any():
            halts.append((seg[-1], f"{s} K 線缺漏或時間戳不連續"))
        if d.index[-1] + pd.Timedelta(hours=4) < last_end:
            halts.append((d.index[-1], f"{s} 行情落後其他幣（最後 {d.index[-1]}）"))
        if f.index[-1] < d.index[-1] - pd.Timedelta(hours=9):
            notes.append(f"{s} 資金費只公布到 {f.index[-1]:%Y-%m-%d %H:%M}，之後暫記 0（估計值）")
    for name, eq in equity.items():
        if eq.empty:
            continue
        eq = eq[eq.index >= since]
        if eq.empty:
            continue
        day_pnl = eq.diff().fillna(eq.iloc[0] if since <= t0 else 0.0).groupby(eq.index.date).cumsum()
        bad = day_pnl[day_pnl <= w["daily_drawdown_halt"] * cap]
        if len(bad):
            halts.append((bad.index[0] + pd.Timedelta(hours=4), f"{name} 單日紙面回撤 {bad.iloc[0] / cap:.2%}，達 −3% 線"))
        streak = 0
        for e in events:
            if e["strategy"] != name or e["type"] != "EXIT" or pd.Timestamp(e["time"]) < since:
                continue
            streak = streak + 1 if e["pnl_usdt"] < 0 else 0
            if e["pnl_usdt"] < -cfg["max_loss_per_trade_pct"] * cap:
                notes.append(f"{name} {e['symbol']} 單筆虧損 {e['pnl_usdt']:.2f} USDT，超過單筆上限")
            if streak >= w["consecutive_losses_halt"]:
                halts.append((pd.Timestamp(e["time"]), f"{name} 連續 {streak} 筆虧損出場"))
                break
    if not halts:
        return None, "", sorted(set(notes))
    halts.sort(key=lambda h: h[0])
    return halts[0][0], "；".join(dict.fromkeys(h[1] for h in halts)), sorted(set(notes))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="解除停記（只有使用者同意後才用）")
    args = ap.parse_args(argv)
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    journal = read_journal()
    if args.resume and HALT.exists():
        HALT.unlink()
        ev = {"type": "RESUME", "time": now.isoformat(), "id": f"RESUME|{now.isoformat()}"}
        append([ev])
        journal.append(ev)
    if HALT.exists():
        print("已停記，等使用者處理：\n" + HALT.read_text(encoding="utf-8"))
        return 2

    fee, slip = load_costs()
    uni = load_universe(cfg["timeframe"])
    events, equity, open_pos, pending, watching = build(cfg, uni, fee, slip)
    t0 = pd.Timestamp(cfg["paper_start_utc"])
    resumes = [pd.Timestamp(e["time"]) for e in journal if e["type"] == "RESUME"]
    since = max([t0] + resumes)
    halt_at, why, notes = watch(cfg, uni, events, equity, since, now)

    seen = {e["id"] for e in journal}
    blocked = [(pd.Timestamp(a["time"]), pd.Timestamp(b["time"])) for a, b in
               zip([e for e in journal if e["type"] == "HALT"], [e for e in journal if e["type"] == "RESUME"])]
    new = []
    for e in events:  # 依時間順序；出場只在對應的進場有記到時才記
        if e["id"] in seen or (halt_at is not None and pd.Timestamp(e["time"]) > halt_at) \
                or any(a < pd.Timestamp(e["time"]) <= b for a, b in blocked):
            continue
        if e["type"] == "EXIT" and f"{e['strategy']}|{e['symbol']}|{e['entry_time']}|ENTRY" not in seen:
            continue
        new.append(e)
        seen.add(e["id"])
    append(new)
    journal += new

    # 對帳：日誌裡已平倉損益 vs 引擎重算（停記期間略過的單不在帳上，直接比對同一批 id）
    by_id = {e["id"]: e for e in events}
    mismatch = [e["id"] for e in journal if e["type"] == "EXIT"
                and (e["id"] not in by_id or abs(by_id[e["id"]]["pnl_usdt"] - e["pnl_usdt"]) > 0.01)]
    mismatch += [e["id"] for e in journal if e["type"] == "ENTRY"
                 and (e["id"] not in by_id or abs(by_id[e["id"]]["price"] - e["price"]) > 1e-9)]
    if mismatch and halt_at is None:
        halt_at, why = now, f"帳本對不上：{len(mismatch)} 筆紀錄與引擎重算不一致（{mismatch[:3]}）"

    if halt_at is not None:
        ev = {"type": "HALT", "time": pd.Timestamp(halt_at).isoformat(), "reason": why,
              "written_at": now.isoformat(), "id": f"HALT|{pd.Timestamp(halt_at).isoformat()}"}
        if ev["id"] not in {e["id"] for e in journal}:
            append([ev])
        HALT.write_text(f"{pd.Timestamp(halt_at).isoformat()}\n{why}\n解除方式：使用者確認後執行 python -m desk.ledger --resume\n",
                        encoding="utf-8")

    trades = pd.DataFrame([e for e in journal if e["type"] in ("ENTRY", "EXIT")])
    if len(trades):
        trades.drop(columns=["levels"], errors="ignore").to_csv(OUT / "ledger.csv", index=False)
    last_bar = max(d.index[-1] for d, _ in uni.values())
    status = {
        "written_at_utc": now.isoformat(), "source": "data.binance.vision USDT 永續 1h 日檔（合成 4h）",
        "last_closed_bar_open_utc": last_bar.isoformat(), "paper_start_utc": cfg["paper_start_utc"],
        "halted": halt_at is not None, "halt_reason": why, "anomalies": notes,
        "books": {n: {"realized_plus_open_pnl_usdt": round(float(eq.iloc[-1]), 2) if len(eq) else 0.0,
                      "bars_recorded": int(len(eq))} for n, eq in equity.items()},
        "journal_events": len(read_journal()), "new_events_this_run": len(new),
        "open_positions": open_pos,
        "pending_next_bar": pending,
        "not_recorded_carried_signals": watching,
        "pending_note": "待成交＝訊號已成立、下一根開盤價尚未公布；價格公布後才記為成交，不預填價格。",
    }
    (OUT / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in status.items() if k not in ("open_positions", "not_recorded_carried_signals")}, ensure_ascii=False, indent=1))
    return 2 if halt_at is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
