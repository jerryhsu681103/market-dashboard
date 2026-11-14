"""診斷用：ETH/BTC 價差引擎的合理性檢查（不作為任何關卡的判定依據）。

用法：python -m scripts.diag_spread
"""
import numpy as np
import pandas as pd

from lab import engine
from lab.data import load_insample
from strategies.eth_btc_ratio_revert import signal

legs = {"ETHUSDT": 1, "BTCUSDT": -1}
parts, funding = [], {}
for sym in legs:
    d, funding[sym] = load_insample(sym, "4h")
    parts.append(d.add_prefix(f"{sym}_"))
df = pd.concat(parts, axis=1)

ratio = df["ETHUSDT_open"] / df["BTCUSDT_open"]
print(f"ETH/BTC 比值（開盤價）：{ratio.iat[0]:.5f} → {ratio.iat[-1]:.5f}，倍數 {ratio.iat[-1] / ratio.iat[0]:.2f}")
print(f"比值最高 {ratio.max():.5f}（{ratio.idxmax().date()}），最低 {ratio.min():.5f}（{ratio.idxmin().date()}）")

for side in (1, -1):
    r = engine.run_spread(lambda d, s=side: pd.Series(s, index=d.index), {}, df, 0.0, 0.0, funding,
                          legs=legs, check_lookahead_bias=False)
    print(f"固定價差 {'+1 做多ETH/空BTC' if side == 1 else '-1 做空ETH/多BTC'}（零成本）："
          f"總報酬 {r['total_return']:.3f}，Sharpe {r['sharpe']:.3f}，最大回撤 {r['max_drawdown']:.3f}，資金費 {r['total_funding']:.3f}")

for p in ({"lookback": 90, "entry_z": 1.5}, {"lookback": 360, "entry_z": 2.5}):
    r0 = engine.run_spread(signal, p, df, 0.0, 0.0, funding, legs=legs)
    print(f"策略 {p}（零成本）：Sharpe {r0['sharpe']:.3f}，總報酬 {r0['total_return']:.3f}，交易 {r0['trade_count']}")

# 比值的自相關：回歸點子需要負自相關
lr = np.log(ratio).diff()
for k in (6, 30, 90):
    rk = np.log(ratio).diff(k)
    print(f"{k} 根報酬與下一個 {k} 根報酬的相關係數：{rk.corr(rk.shift(-k)):.3f}")
