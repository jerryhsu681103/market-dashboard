"""把 reports/desk/<日期>/*.json 整理成總表（markdown）。用法：python -m desk.report"""
import json

from .candidates import C, TESTS
from .evaluate import OUT

GATE = {"G0": "前視", "G1": "近4年無邊", "G2": "參數不穩", "G3": "walk-forward 不過", "G4": "輸給死拿",
        "G5": "費用加倍垮掉", "G6": "熊市/震盪撐不住", "": "—", "ERR": "執行錯誤"}


def main():
    day = sorted(p.name for p in OUT.iterdir() if p.is_dir())[-1]
    rs = [json.loads((OUT / day / f"{n}@{tf}.json").read_text(encoding="utf-8")) for n, tf in TESTS]
    rs.sort(key=lambda r: (r["status"] != "alive", -r.get("score_total", 0)))
    L = [f"# 夜巡＋驗偽總表（{day}）", "",
         f"- 測試數：{len(rs)}（{len(C)} 個候選；其中 4 個另測 1 小時線）。存活：{sum(r['status'] == 'alive' for r in rs)}。",
         "- 資料：Binance USDT 永續 1 小時 K 線與資金費，來源 data.binance.vision（月檔＋日檔，SHA256 校驗），"
         f"評估視窗 {rs[0]['full_best']['start'][:10]} → {rs[0]['full_best']['end'][:16]} UTC。",
         f"- 成本：手續費 {rs[0]['fee']:.4%}＋滑點 {rs[0]['slippage']:.4%}（單邊，config/costs.json）＋資金費。"
         "2026-11 的資金費月檔尚未公布，11 月 1–14 日資金費記為 0（低估成本，影響很小但屬估計）。",
         "- 產生方式：`python -m desk.evaluate && python -m desk.report`；淘汰線寫在 desk/evaluate.py 開頭，跑之前已 commit。",
         "- 死拿基準（同幣池、同成本、一直做多）：walk-forward 期間 "
         f"Sharpe {rs[0]['hold_wf']['sharpe']:.2f}、報酬 {rs[0]['hold_wf']['total_return']:.1%}、最大回撤 {rs[0]['hold_wf']['max_drawdown']:.1%}；"
         f"近 4 年 Sharpe {rs[0]['hold_full']['sharpe']:.2f}、報酬 {rs[0]['hold_full']['total_return']:.1%}、最大回撤 {rs[0]['hold_full']['max_drawdown']:.1%}。", "",
         "| 候選 | 週期 | 類型 | 生死 | 掛在哪 | 近4年最優 Sharpe | WF Sharpe | WF 報酬 | WF 回撤 | WF 筆數 | 費用×2 WF Sharpe | 五項分數 | 憑什麼有邊 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        if "wf" not in r:
            L.append(f"| {r['name']} | {r['timeframe']} |  | {r['status']} | {r.get('death', '')} | | | | | | | | |")
            continue
        w, x = r["wf"], r["wf_cost_x2"]
        sc = "/".join(str(v) for v in r["scores"].values())
        L.append(f"| {r['name']} | {r['timeframe']} | {r['family']} | {'活' if r['status'] == 'alive' else '殺'} | {GATE[r['gate']]} "
                 f"| {r['full_best']['sharpe']:.2f} | {w['sharpe']:.2f} | {w['total_return']:.1%} | {w['max_drawdown']:.1%} | {w['trades']} "
                 f"| {x['sharpe']:.2f} | {sc}＝{r['score_total']} | {r['why']} |")
    L += ["", "五項分數順序：扣完費用的 walk-forward／最大回撤／成交筆數／最長連虧／熊市與震盪。每項 1–5 分，分段寫在 desk/evaluate.py 的 score()。", "",
          "## 死因明細", ""]
    for r in rs:
        if r["status"] != "alive":
            ov = f"（重疊：{r['overlap']}）" if r.get("overlap") else ""
            L.append(f"- **{r['name']}@{r['timeframe']}**：{r.get('death', '')}{ov}")
    (OUT / day / "night_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
