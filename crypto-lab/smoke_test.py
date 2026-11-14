"""02 煙霧測試：BTCUSDT 永續 1 小時線，EMA 20/50 金叉做多、死叉做空。

用法：
    python smoke_test.py                       # 讀 config/costs.json
    python smoke_test.py --fee 0.0005 --slippage 0.0002
"""
import argparse

import pandas as pd

from lab import engine
from lab.costs import load_costs
from lab.data import load_insample
from strategies.ema_cross import signal

ap = argparse.ArgumentParser()
ap.add_argument("--fee", type=float)
ap.add_argument("--slippage", type=float)
ap.add_argument("--symbol", default="BTCUSDT")
a = ap.parse_args()

fee, slip = load_costs()
fee = a.fee if a.fee is not None else fee
slip = a.slippage if a.slippage is not None else slip

df, funding = load_insample(a.symbol)
result = engine.run(signal, {"fast": 20, "slow": 50}, df, fee, slip, funding)

pd.set_option("display.float_format", lambda v: f"{v:,.6f}")
print(f"{a.symbol} 樣本內 EMA 20/50（engine.run，lab/engine.py）")
print(engine.metrics_table(result).to_string())
print("""
數字來源（lab/engine.py）：
  持倉          run():      pos = sig.shift(1)               第 t-1 根信號 → 第 t 根開盤成交
  逐根報酬      run():      bar_ret = open[t+1]/open[t] - 1
  手續費+滑點   simulate(): entry_cost / exit_cost = |部位變動| × (fee + slippage)
  資金費        simulate(): funding = -pos × 結束於該根 K 線的資金費率
  淨報酬        simulate(): net = gross + funding - entry_cost - exit_cost
  total_return  summarize(): equity = (1 + net).cumprod()；equity[-1] - 1
  sharpe        summarize(): mean(net) / std(net) × sqrt(8760)
  max_drawdown  summarize(): min(equity / equity.cummax() - 1)
  profit_factor summarize(): 逐筆交易盈利總和 / 虧損總和（_trade_pnls）
  trade_count   summarize(): 連續同方向持倉段的數量（_trade_pnls）""")
