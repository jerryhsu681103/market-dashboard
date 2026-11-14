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


# ---------- 週期換算

def test_resample_4h_and_bars_helper():
    from lab.data import resample
    from strategies._util import bars
    df = make_df(48)
    df4 = resample(df, "4h")
    assert len(df4) == 12
    assert df4["open"].iat[0] == df["open"].iat[0]
    assert df4["close"].iat[0] == df["close"].iat[3]
    assert df4["high"].iat[0] == df["high"].iloc[:4].max()
    assert df4["volume"].iat[0] == pytest.approx(df["volume"].iloc[:4].sum())
    assert bars(df, 720) == 720 and bars(df4, 720) == 180 and bars(df4, 3) == 1
    # 不完整的週期丟棄
    assert len(resample(df.iloc[1:], "4h")) == 11


# ---------- 價差引擎

def _pair_df(n=300, seed=0, same=False):
    a = make_df(n, seed)
    b = a.copy() if same else make_df(n, seed + 1)
    df = pd.concat([a.add_prefix("AAA_"), b.add_prefix("BBB_")], axis=1)
    return df


def test_spread_identical_legs_only_costs():
    """兩腳價格完全相同時，價差部位毛利為 0，只剩成本。"""
    df = _pair_df(same=True)
    sig = lambda d: pd.Series(np.where(np.arange(len(d)) % 50 < 25, 1, -1), index=d.index)
    f = {"AAA": pd.Series(0.0, index=df.index[::8]), "BBB": pd.Series(0.0, index=df.index[::8])}
    r = engine.run_spread(sig, {}, df, 0.001, 0.0, f, legs={"AAA": 1, "BBB": -1},
                          check_lookahead_bias=False, keep_detail=True)
    assert r["detail"]["gross"].abs().max() < 1e-12
    # 部位變動次數含首次進場（0→1，每腳 1 單位）；其餘為翻倉（每腳 2 單位）
    changes = int((np.diff(r["detail"]["pos"].to_numpy()) != 0).sum())
    expected = 2 * 1 * 0.001 + (changes - 1) * 2 * 2 * 0.001
    assert r["total_cost"] == pytest.approx(expected)


def test_spread_matches_single_leg_sum():
    """價差結果的逐根淨報酬 = 兩腳分別用 run() 計算的淨報酬相加。"""
    df = _pair_df()
    a = df.filter(like="AAA_").rename(columns=lambda c: c[4:])
    b = df.filter(like="BBB_").rename(columns=lambda c: c[4:])
    fa, fb = pd.Series(0.0002, index=df.index[::8]), pd.Series(-0.0001, index=df.index[::8])
    sig = lambda d: pd.Series(np.sign(np.sin(np.arange(len(d)) / 7)).astype(int), index=d.index)
    r = engine.run_spread(sig, {}, df, 0.0005, 0.0002, {"AAA": fa, "BBB": fb},
                          legs={"AAA": 1, "BBB": -1}, check_lookahead_bias=False)
    ra = engine.run(sig, {}, a, 0.0005, 0.0002, fa, check_lookahead_bias=False)
    rb = engine.run(lambda d: -sig(d), {}, b, 0.0005, 0.0002, fb, check_lookahead_bias=False)
    assert np.allclose(r["returns"].to_numpy(), ra["returns"].to_numpy() + rb["returns"].to_numpy())


def test_spread_requires_funding_and_detects_lookahead():
    df = _pair_df()
    f = {"AAA": pd.Series(0.0, index=df.index[::8])}
    with pytest.raises(ValueError):
        engine.run_spread(lambda d: pd.Series(0, index=d.index), {}, df, 0.0, 0.0, f,
                          legs={"AAA": 1, "BBB": -1})
    f["BBB"] = f["AAA"]
    cheat2 = lambda d: pd.Series(np.sign(d["AAA_close"].shift(-1) - d["AAA_close"]).fillna(0).astype(int), index=d.index)
    with pytest.raises(engine.LookaheadError):
        engine.run_spread(cheat2, {}, df, 0.0, 0.0, f, legs={"AAA": 1, "BBB": -1})


def test_permutation_fixed_cols_stay_in_place():
    df = make_df(500)
    df["feat"] = np.arange(500, dtype=float)
    df["funding_rate"] = np.arange(500, dtype=float)
    p = permute_bars(df, seed=7, fixed_cols=["feat"])
    assert p["feat"].equals(df["feat"])
    assert p["funding_rate"].equals(df["funding_rate"])
    assert not p["volume"].equals(df["volume"])  # 一般 K 線屬性仍跟著打亂
