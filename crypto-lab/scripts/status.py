"""顯示研究進度總覽：python -m scripts.status"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABEL = {"dead_stage1": "第 1 關淘汰", "dead_stage2": "第 2 關淘汰", "dead_stage3": "第 3 關淘汰",
         "dead_stage4": "第 4 關淘汰", "alive_stage1": "通過第 1 關", "alive_stage2": "通過第 2 關",
         "alive_stage3": "通過第 3 關", "watchlist": "紙面觀察名單", "review_stage1": "待查前視偏差"}


def main():
    log = pd.read_csv(ROOT / "research_log.csv", dtype=str)
    wl = pd.read_csv(ROOT / "watchlist.csv")
    print(f"累計測試：{len(log)} 次（{log['idea'].nunique()} 個點子）")
    for status, n in log["status"].value_counts().items():
        print(f"  {LABEL.get(status, status)}：{n}")
    alive = log[~log["status"].str.startswith("dead")]
    if len(alive):
        print("\n仍存活的點子：")
        for _, r in alive.iterrows():
            print(f"  {r['idea']} [{r['timeframe']}]  {LABEL.get(r['status'], r['status'])}")
    else:
        print("\n目前沒有存活的點子。")
    print(f"\n紙面觀察名單：{len(wl)} 個策略" + ("" if wl.empty else "\n" + wl[["name", "symbol", "promoted_date"]].to_string(index=False)))
    print("\n最近 5 筆：")
    for _, r in log.tail(5).iterrows():
        print(f"  {r['date']}  {r['idea']} [{r['timeframe']}]  {LABEL.get(r['status'], r['status'])}  {'' if pd.isna(r['death_reason']) else r['death_reason']}")


if __name__ == "__main__":
    main()
