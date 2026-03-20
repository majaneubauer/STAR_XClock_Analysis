# %%

# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
# This script parses and performs basic quality control on xclock TTL event logs.
# The input CSV is an event table recorded by a DAQ. Each row represents one edge
# transition (rising or falling) on one digital channel, and contains:
#   - t_raw : high-resolution monotonic counter (DAQ timebase)
#   - code  : signed channel identifier; sign encodes edge polarity
#   - ts_raw: coarse wall-clock anchor (epoch-like, sparsely updated)
#
# Processing steps:
#   1) Load the CSV robustly (either 3 columns or a single comma-separated column).
#   2) Convert the signed event code into:
#        - channel : integer channel identifier (1..4)
#        - edge    : "rising" or "falling"
#   3) Compute basic counts per channel/edge and an average event rate (Hz) from
#      rising edges. Duration in seconds is derived from ts_raw; high-resolution
#      timing uses t_raw scaled to the same duration.
#   4) For each channel, compute rising-edge inter-event intervals (IEIs) and flag
#      unusually long intervals as candidates for missed pulses. In particular,
#      intervals near integer multiples of the nominal period (2*T0, 3*T0, …) are
#      labeled as "drop_like".
#   5) Create a nearest-neighbor mapping from Channel 3 rising events to the
#      nearest Channel 4 rising event, yielding a correspondence between calcium
#      event indices and measured camera frame indices.
#
# Channel semantics in this setup:
#   - Channel 1: scheduled TTL pulses sent to the behavior camera (nominal 50 Hz).
#   - Channel 4: measured TTL pulses from the behavior camera (actual acquired frames).
#   - Channel 2: demonstration clock (nominal 100 Hz in this example).
#   - Channel 3: TTL events recorded for the Inscopix acquisition stream.
#                When using Inscopix in this mode, this channel can reflect
#                scheduled frames; dropped frames are typically identified from
#                Inscopix metadata (see vendor API/manual).
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd

# Define your x-clock output *.csv path here
CSV_PATH = r"*.csv"

CHANNELS = [1, 2, 3, 4]

TOL_FRAC = 0.08      # tolerance as fraction of nominal period for k*T0 matching
LONG_FACTOR = 1.5    # additional long-IEI flag (IEI > LONG_FACTOR*T0)

CH_TO_MAP = 3        # events that will receive a matched camera-frame index
CH_CAMERA = 4        # measured camera frames (target index space)
EDGE = "rising"      # frame marker edge
TOL_S = None         # optional (seconds), e.g. 0.05; leave None to accept all matches


def load_xclock_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, dtype=str)
    if raw.shape[1] == 1:
        df = raw.iloc[:, 0].astype(str).str.split(",", expand=True).iloc[:, :3]
    else:
        df = raw.iloc[:, :3].copy()
    df.columns = ["t_raw", "code", "ts_raw"]

    df["t_raw"] = pd.to_numeric(df["t_raw"], errors="coerce")
    df["code"] = pd.to_numeric(df["code"], errors="coerce")
    df["ts_raw"] = pd.to_numeric(df["ts_raw"], errors="coerce")

    df = df.dropna(subset=["t_raw", "code"]).sort_values("t_raw").reset_index(drop=True)
    df["channel"] = df["code"].abs().astype(int)
    df["edge"] = np.where(df["code"] > 0, "rising", "falling")
    df = df[df["channel"].isin(CHANNELS)].copy()
    return df


def calibrate_traw_to_seconds(df: pd.DataFrame) -> tuple[float, float]:
    ts = df["ts_raw"].dropna()
    if ts.nunique() < 2:
        raise ValueError("Need at least two distinct ts_raw values to calibrate time in seconds.")

    duration_s = (ts.max() - ts.min()) / 1e9
    t_span = df["t_raw"].iloc[-1] - df["t_raw"].iloc[0]
    if not np.isfinite(t_span) or t_span <= 0:
        raise ValueError("Invalid t_raw span.")

    sec_per_traw = duration_s / t_span
    return duration_s, sec_per_traw


def summarize_counts(df: pd.DataFrame, duration_s: float) -> pd.DataFrame:
    counts = (
        df.groupby(["channel", "edge"])
          .size()
          .unstack(fill_value=0)
          .assign(total=lambda x: x.sum(axis=1))
          .reindex(CHANNELS)
    )
    rise_counts = (
        df[df["edge"].eq("rising")]
          .groupby("channel")
          .size()
          .reindex(CHANNELS)
          .fillna(0)
          .astype(int)
    )
    counts["duration_s"] = duration_s
    counts["avg_hz"] = rise_counts / duration_s
    return counts


def flag_ieis(df: pd.DataFrame, ch: int, sec_per_traw: float) -> pd.DataFrame:
    r = df[(df["channel"] == ch) & (df["edge"] == "rising")].sort_values("t_raw").copy()
    r["pulse_idx"] = np.arange(1, len(r) + 1)

    r["iei_s"] = r["t_raw"].diff() * sec_per_traw
    T0 = np.nanmedian(r["iei_s"].to_numpy())
    if not np.isfinite(T0) or T0 <= 0:
        return pd.DataFrame()

    iei = r["iei_s"].to_numpy()
    ratio = iei / T0
    k = np.rint(ratio)  # float array; avoids pandas NA-to-int issues
    residual = np.abs(iei - k * T0)

    drop_like = (k >= 2) & (residual <= TOL_FRAC * T0)
    is_long = iei > (LONG_FACTOR * T0)

    out = pd.DataFrame({
        "pulse_idx": r["pulse_idx"].to_numpy(),
        "iei_s": iei,
        "k": k,
        "n_missing": np.where(drop_like, k - 1, 0).astype(int),
        "drop_like": drop_like,
        "is_long": is_long,
        "residual_s": residual,
    })

    flagged = out[(out["pulse_idx"] >= 2) & (out["drop_like"] | out["is_long"])].copy()
    print(f"\nCh{ch}: nominal T0={T0:.6f}s (~{1/T0:.3f} Hz)")
    print("Flagged intervals:", len(flagged))
    if len(flagged):
        print(flagged.head(20).to_string(index=False))
    return flagged


def nearest_map(df: pd.DataFrame, ch_to_map: int, ch_camera: int, sec_per_traw: float) -> pd.DataFrame:
    to_map = df[(df["channel"] == ch_to_map) & (df["edge"] == EDGE)].sort_values("t_raw").copy()
    camera = df[(df["channel"] == ch_camera) & (df["edge"] == EDGE)].sort_values("t_raw").copy()

    to_map["to_map_idx"] = np.arange(len(to_map), dtype=int)
    camera["cam_frame"] = np.arange(len(camera), dtype=int)

    tol_traw = (TOL_S / sec_per_traw) if TOL_S is not None else None

    m = pd.merge_asof(
        to_map[["t_raw", "to_map_idx"]],
        camera[["t_raw", "cam_frame"]],
        on="t_raw",
        direction="nearest",
        tolerance=tol_traw,
    )

    m = m.merge(camera[["t_raw", "cam_frame"]].rename(columns={"t_raw": "t_raw_cam"}), on="cam_frame", how="left")
    m["delta_s"] = (m["t_raw"] - m["t_raw_cam"]) * sec_per_traw

    print(f"\nNearest mapping: Ch{ch_to_map} -> Ch{ch_camera} ({EDGE} edges)")
    print("Mapped rows:", m["cam_frame"].notna().sum(), "of", len(m))
    print("\nDelta_s summary (to_map time minus matched camera time):")
    print(m["delta_s"].dropna().describe())
    print("\nFirst 10 mappings:")
    print(m.head(10).to_string(index=False))

    return m


df = load_xclock_csv(CSV_PATH)
duration_s, sec_per_traw = calibrate_traw_to_seconds(df)

counts = summarize_counts(df, duration_s)
print(counts)

for ch in CHANNELS:
    _ = flag_ieis(df, ch, sec_per_traw)

m = nearest_map(df, CH_TO_MAP, CH_CAMERA, sec_per_traw)

# %%