import numpy as np
import pandas as pd
import pytest

from lab import engine
from lab.permutation import permute_bars, permute_many
from lab.research import plateau_check
from strategies.ema_cross import signal as ema_signal


def make_df(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.concatenate([[100], close[:-1]]) * np.exp(rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, n))
    vol = rng.uniform(1, 10, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def zero_funding(df):
    return pd.Series(0.0, index=df.index[::8])


# ---------- 決定性：同一組參數連跑 3 次，結果完全一致

def test_deterministic_three_runs():
    df = make_df()
    f = pd.Series(0.0001, index=df.index[::8])
    runs = [engine.run(ema_signal, {"fast": 20, "slow": 50}, df, 0.0005, 0.0002, f) for _ in range(3)]
    for r in runs[1:]:
        assert r["returns"].equals(runs[0]["returns"])
        for k in ("total_return", "profit_factor", "sharpe", "max_drawdown", "trade_count"):
            assert r[k] == runs[0][k]


# ---------- 成本必填

@pytest.mark.parametrize("fee,slip", [(None, 0.0), (0.0, None), (None, None)])
def test_costs_required(fee, slip):
    df = make_df(200)
    with pytest.raises(ValueError):
        engine.run(ema_signal, {}, df, fee, slip, zero_funding(df))


def test_funding_required_for_perps():
    df = make_df(200)
    with pytest.raises(ValueError):
        engine.run(ema_signal, {}, df, 0.0, 0.0)


# ---------- 前視偏差

def cheat(df):
    return pd.Series(np.sign(df["close"].shift(-1) - df["close"]).fillna(0).astype(int), index=df.index)


def full_sample_zscore(df):
    z = (df["close"] - df["close"].mean()) / df["close"].std()
    return pd.Series(np.where(z > 0, 1, -1), index=df.index)


@pytest.mark.parametrize("fn", [cheat, full_sample_zscore])
def test_lookahead_detected(fn):
    df = make_df(500)
    with pytest.raises(engine.LookaheadError):
        engine.run(fn, {}, df, 0.0, 0.0, zero_funding(df))


def test_execution_next_open():
    """手算：第 0 根信號做多 → 第 1 根開盤成交，賺 open[2]/open[1]-1。"""
    idx = pd.date_range("2022-01-01", periods=4, freq="h", tz="UTC")
    df = pd.DataFrame({"open": [100, 110, 121, 121], "high": [130] * 4,
                       "low": [90] * 4, "close": [110, 121, 121, 121]}, index=idx)
    sig = lambda d: pd.Series([1, 0, 0, 0], index=d.index)
    r = engine.run(sig, {}, df, 0.001, 0.0, allow_no_funding=True, check_lookahead_bias=False,
                   keep_detail=True)
    net = r["returns"].to_numpy()
    assert net[0] == 0
    assert net[1] == pytest.approx(121 / 110 - 1 - 0.001)   # 進場成本
    assert net[2] == pytest.approx(-0.001)                   # 出場成本
    assert r["trade_count"] == 1


def test_funding_sign():
    """多單在正費率時付錢；空單收錢。"""
    idx = pd.date_range("2022-01-01", periods=10, freq="h", tz="UTC")
    df = pd.DataFrame({c: [100.0] * 10 for c in ("open", "high", "low", "close")}, index=idx)
    f = pd.Series([0.001], index=[idx[5]])
    for side in (1, -1):
        sig = lambda d, s=side: pd.Series(s, index=d.index)
        r = engine.run(sig, {}, df, 0.0, 0.0, f, check_lookahead_bias=False)
        assert r["total_funding"] == pytest.approx(-side * 0.001)


# ---------- 置換

def test_permutation_keeps_endpoints_and_shape():
    df = make_df(1000)
    p = permute_bars(df, seed=1)
    assert p["open"].iat[0] == df["open"].iat[0]
    assert p["close"].iat[-1] == pytest.approx(df["close"].iat[-1])
    assert (p["high"] >= p[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (p["low"] <= p[["open", "close"]].min(axis=1) + 1e-9).all()
    assert p.index.equals(df.index)
    assert not np.allclose(p["close"], df["close"])
    assert permute_bars(df, seed=1).equals(p)  # 種子固定 → 結果固定


def test_permutation_start_index_and_joint():
    a, b = make_df(800, 1), make_df(800, 2)
    out = permute_many({"a": a, "b": b}, seed=3, start_index=300)
    assert out["a"].iloc[:301].equals(a.iloc[:301])
    assert out["a"]["close"].iat[-1] == pytest.approx(a["close"].iat[-1])
    # 同一組打亂順序：兩個幣種的成交量對應關係保持一致
    pa = out["a"]["volume"].to_numpy()[301:]
    pb = out["b"]["volume"].to_numpy()[301:]
    va, vb = a["volume"].to_numpy()[301:], b["volume"].to_numpy()[301:]
    order = [np.where(va == x)[0][0] for x in pa]
    assert np.allclose(vb[order], pb)


# ---------- 參數高原

def test_plateau_spike_detection():
    grid = {"a": [1, 2, 3], "b": [1, 2, 3]}
    rows = [{"a": a, "b": b, "sharpe": 0.1} for a in grid["a"] for b in grid["b"]]
    rows[4]["sharpe"] = 2.0  # 中心尖峰
    assert plateau_check(pd.DataFrame(rows), grid)["is_spike"]
    rows = [{"a": a, "b": b, "sharpe": 1.5} for a in grid["a"] for b in grid["b"]]
    rows[4]["sharpe"] = 2.0
    assert not plateau_check(pd.DataFrame(rows), grid)["is_spike"]
