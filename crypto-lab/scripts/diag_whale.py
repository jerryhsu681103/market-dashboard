"""診斷用：whale_retail_divergence 的時間對齊與穩健性檢查（不作為任何關卡的判定依據）。

用法：python -m scripts.diag_whale
檢查：
  1. 特徵延遲 1 小時（4 小時線只用每根前 3 小時的快照）、延遲 1 根 K 線 → 優勢若消失，代表依賴時間對齊細節
  2. 分年度表現
  3. ETHUSDT 交叉驗證（ETH metrics 從 2021-12 起）
"""
import pandas as pd

from lab import engine
from lab.costs import load_costs
from lab.data import DATA, attach_funding, load_insample
from strategies.whale_retail_divergence import signal

P = {"threshold": 1.5, "hold": 3}
fee, slip = load_costs()


def metrics_4h(symbol: str, lag_hours: int) -> pd.DataFrame:
    m = pd.read_parquet(DATA / f"{symbol}_metrics_1h.parquet")
    m = m.where(m > 0)
    m.index = m.index + pd.Timedelta(hours=lag_hours)  # 每筆觀測延後 lag 小時才「可用」
    return m.resample("4h", origin="epoch", label="left", closed="left").last()


def build(symbol: str, lag_hours: int = 0, lag_bars: int = 0):
    df, f = load_insample(symbol, "4h")
    df = attach_funding(df, f)
    m = metrics_4h(symbol, lag_hours).shift(lag_bars).reindex(df.index)
    return df.join(m), f


rows = []
for label, lh, lb in (("基準", 0, 0), ("延遲 1 小時", 1, 0), ("延遲 2 小時", 2, 0), ("延遲 1 根（4 小時）", 0, 1), ("延遲 2 根（8 小時）", 0, 2)):
    df, f = build("BTCUSDT", lh, lb)
    r = engine.run(signal, P, df, fee, slip, f)
    rows.append({"設定": label, "sharpe": r["sharpe"], "pf": r["profit_factor"], "total_return": r["total_return"],
                 "max_dd": r["max_drawdown"], "trades": r["trade_count"]})
print("== BTCUSDT 時間延遲測試（參數 threshold=1.5, hold=3）")
print(pd.DataFrame(rows).round(3).to_string(index=False))

df, f = build("BTCUSDT")
r = engine.run(signal, P, df, fee, slip, f, keep_detail=True)
net = r["returns"]
yearly = net.groupby(net.index.year).agg(
    total_return=lambda x: (1 + x).prod() - 1,
    sharpe=lambda x: x.mean() / x.std() * (2190 ** 0.5) if x.std() > 0 else float("nan"))
yearly["exposure"] = (r["detail"]["pos"] != 0).groupby(net.index.year).mean()
print("\n== BTCUSDT 分年度")
print(yearly.round(3).to_string())

dfe, fe = build("ETHUSDT")
re_ = engine.run(signal, P, dfe, fee, slip, fe)
start = dfe["top_ls_position"].first_valid_index()
print(f"\n== ETHUSDT 交叉驗證（metrics 起始 {start}）")
print({k: round(re_[k], 3) for k in ("sharpe", "profit_factor", "total_return", "max_drawdown")}, "trades", re_["trade_count"])
