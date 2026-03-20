# %%
# =============================================================================
# CELL 0 — CONFIG
# All paths, constants, and tunable parameters live here.
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from itertools import groupby
import time

# --- Paths ---
BASE      = Path(r"G:\uv-projects\star-protocols-xclock\data\src")
XCLOCK_T1 = BASE / "xclock_timestamps_2025-12-16_15-51-58_MED-1090_SR_T1.csv"
XCLOCK_T2 = BASE / "xclock_timestamps_2025-12-16_16-11-57_MED-1090_SR_T2.csv"
SLEAP_T1  = BASE / "all_videos_labels.v002.000_MED-1090_SR_T1.analysis.csv"
SLEAP_T2  = BASE / "all_videos_labels.v002.001_MED-1090_SR_T2.analysis.csv"
CA_FILE   = BASE / "20260220_MED-1090_accepted_cells_S.csv"

# Affine calibration produced by xclock_pixel-calibration.py — required.
CALIB_FILE = BASE / "px_to_cm_affine_calib_trial1.npz"

# --- XClock channels ---
CH_CAM  = 4          # camera pulse channel
CH_CA30 = 3          # calcium sync channel
EDGE    = "rising"

# --- SLEAP keys & quality thresholds ---
KEY_X, KEY_Y, KEY_SCORE = "center.x", "center.y", "center.score"
KEY_INST           = "instance.score"
MIN_INSTANCE_SCORE = 0.0
MIN_BODYPART_SCORE = 0.0
MAX_GAP_FRAMES     = 10      # max contiguous NaN frames to interpolate across

# --- Out-of-bounds detection ---
OOB_MARGIN_CM = 1            # cm outside arena walls before flagging as out of bounds

# --- Camera frame rate ---
CAMERA_FPS = 50.0            # frames per second -> used for cm/s speed

# --- Speed filtering (applied on SG first derivative) ---
MIN_SPEED_CM_S = 1.5         # set 0.0 to disable
MAX_SPEED_CM_S = 40.0        # set np.inf to disable
SGF_WINDOW     = 17          # frames (~0.34 s at 50 fps); must be odd > SGF_POLYORDER
SGF_POLYORDER  = 1           # linear

# --- Spatial map parameters (cm) ---
# XLIM_CM / YLIM_CM are derived from calibration in Cell 1.
BIN_CM            = 1.75
OCC_MIN_S         = 0.1
SMOOTH_SIGMA_BINS = 1

# --- Door-lift split ---
DOOR_LIFT_FRAME_T1 = 15067   # camera frame separating T1-pre from T1-post

# --- Event binarisation ---
Z_THRESH      = 2.0
REFRACTORY_S  = 0.2
Z_ON_NONZERO  = True
N_MIN_EVENTS  = 10

# --- Permutation test ---
N_PERM   = 100
ALPHA    = 0.05
RNG_SEED = 0

# --- Plots ---
SPD_HIST_XLIM = (0, 60)      # cm/s — fixed x-axis for all speed histograms

# Wong (2011) colorblind-safe palette — used throughout all plots
WONG_BLUE       = "#0072B2"
WONG_ORANGE     = "#E69F00"
WONG_GREEN      = "#009E73"
WONG_YELLOW     = "#F0E442"
WONG_SKY_BLUE   = "#56B4E9"
WONG_VERMILLION = "#D55E00"
WONG_PINK       = "#CC79A7"
WONG_BLACK      = "#000000"

# --- Output ---
SAVE_OUTPUTS = False
OUT_DIR      = Path(r"G:\uv-projects\star-protocols-xclock\data\results")

print("CONFIG loaded.")


# %%
# =============================================================================
# CELL 1 — LOAD & PREPROCESS BEHAVIOR
#
# Pipeline per trial (T1, T2):
#   1. Load affine calibration (.npz)  ->  px_to_cm()  [required, no fallback]
#   2. Load SLEAP CSV  ->  quality filter  ->  one row per frame
#   3. Dense-fill scaffold to every frame index
#   4. Affine px -> cm
#   5. Flag out-of-bounds  (bodypart outside arena +/- OOB_MARGIN_CM)
#      -> NaN out-of-bounds positions so they don't bias smoothing
#   6. Gap-interpolate (<= MAX_GAP_FRAMES)
#   7. Savitzky-Golay smooth on each contiguous finite run
#      deriv=0 -> smoothed position
#      deriv=1, delta=1/CAMERA_FPS -> velocity in cm/s
#   8. Speed from SG first derivative (cm/s)
#   9. Masks:
#        ok_raw   — finite after gap-fill (before speed filter)
#        is_oob   — originally outside arena bounds
#        speed_ok — ok_raw & ~is_oob & speed in [MIN_SPEED, MAX_SPEED]
#
# Outputs: beh_t1, beh_t2  — one row per camera frame
#   columns: frame_idx | x_px | y_px | x_cm | y_cm | x_cm_raw | y_cm_raw
#            | is_oob | x_sg | y_sg | speed_cm_s | ok_raw | speed_ok
#
# Also sets: XLIM_CM, YLIM_CM, W_CM, H_CM  (used by all downstream cells)
# =============================================================================

def _sniff_sep(path: Path, nrows: int = 50) -> str:
    best_sep, best_ncol = None, -1
    for sep in [",", "\t", ";"]:
        try:
            tmp = pd.read_csv(path, sep=sep, nrows=nrows)
            if tmp.shape[1] > best_ncol:
                best_sep, best_ncol = sep, tmp.shape[1]
        except Exception:
            pass
    if best_sep is None:
        raise ValueError(f"Cannot infer separator for {path}")
    return best_sep

def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep=_sniff_sep(path), **kwargs)


def load_affine_calibration(calib_path: Path):
    c    = np.load(calib_path, allow_pickle=True)
    TL   = c["TL"].astype(float)
    Ainv = c["Ainv"].astype(float)
    w_cm = float(c["W_CM"])
    h_cm = float(c["H_CM"])
    def px_to_cm(x_px, y_px):
        P  = np.column_stack([np.asarray(x_px, float), np.asarray(y_px, float)])
        d  = P - TL.reshape(1, 2)
        ab = (Ainv @ d.T).T
        return ab[:, 0] * w_cm, ab[:, 1] * h_cm
    return px_to_cm, w_cm, h_cm


if not CALIB_FILE.exists():
    raise FileNotFoundError(f"Calibration file not found: {CALIB_FILE}")
_px_to_cm, W_CM, H_CM = load_affine_calibration(CALIB_FILE)
print(f"Affine calibration loaded: {CALIB_FILE.name}")

XLIM_CM = (0.0, W_CM)
YLIM_CM = (0.0, H_CM)
print(f"Arena: {W_CM:.1f} x {H_CM:.1f} cm  |  XLIM_CM={XLIM_CM}  YLIM_CM={YLIM_CM}")


def load_sleap_raw(sleap_path: Path) -> pd.DataFrame:
    s = _read_csv(sleap_path)
    needed = {"frame_idx", KEY_INST, KEY_X, KEY_Y, KEY_SCORE}
    missing = sorted(needed - set(s.columns))
    if missing:
        raise ValueError(f"{sleap_path.name}: missing columns {missing}")
    d = s[["frame_idx", KEY_INST, KEY_X, KEY_Y, KEY_SCORE]].copy()
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["frame_idx"]).copy()
    d["frame_idx"] = d["frame_idx"].astype(int)
    d = d[(d[KEY_INST] > MIN_INSTANCE_SCORE) & (d[KEY_SCORE] > MIN_BODYPART_SCORE)]
    d = d.sort_values("frame_idx").groupby("frame_idx", as_index=False).last()
    return d


def preprocess_behavior(raw: pd.DataFrame, n_frames: int) -> pd.DataFrame:
    full = pd.DataFrame({"frame_idx": np.arange(n_frames, dtype=int)})
    full = full.merge(raw[["frame_idx", KEY_X, KEY_Y]], on="frame_idx", how="left")
    full = full.rename(columns={KEY_X: "x_px", KEY_Y: "y_px"})

    x_px = full["x_px"].to_numpy(dtype=float)
    y_px = full["y_px"].to_numpy(dtype=float)
    x_cm, y_cm = _px_to_cm(x_px, y_px)
    full["x_cm"] = x_cm
    full["y_cm"] = y_cm

    m = OOB_MARGIN_CM
    full["is_oob"] = (
        (full["x_cm"] < -m) | (full["x_cm"] > W_CM + m) |
        (full["y_cm"] < -m) | (full["y_cm"] > H_CM + m)
    )
    full["x_cm_raw"] = full["x_cm"].copy()
    full["y_cm_raw"] = full["y_cm"].copy()

    x_c = full["x_cm"].to_numpy(dtype=float).copy()
    y_c = full["y_cm"].to_numpy(dtype=float).copy()
    oob = full["is_oob"].to_numpy()
    x_c[oob] = np.nan
    y_c[oob] = np.nan
    full["x_cm"] = x_c
    full["y_cm"] = y_c

    full["x_cm"] = full["x_cm"].interpolate("linear", limit=MAX_GAP_FRAMES, limit_area="inside")
    full["y_cm"] = full["y_cm"].interpolate("linear", limit=MAX_GAP_FRAMES, limit_area="inside")
    full["ok_raw"] = np.isfinite(full["x_cm"]) & np.isfinite(full["y_cm"])

    x_arr = full["x_cm"].to_numpy(dtype=float)
    y_arr = full["y_cm"].to_numpy(dtype=float)
    x_sg  = x_arr.copy()
    y_sg  = y_arr.copy()
    vx_sg = np.full(n_frames, np.nan, dtype=float)
    vy_sg = np.full(n_frames, np.nan, dtype=float)

    finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    for is_finite, run in groupby(zip(finite_mask, np.arange(n_frames)), key=lambda t: t[0]):
        if not is_finite:
            continue
        idx = np.array([i for _, i in run])
        if len(idx) >= SGF_WINDOW:
            x_sg[idx]  = savgol_filter(x_arr[idx], SGF_WINDOW, SGF_POLYORDER, deriv=0)
            y_sg[idx]  = savgol_filter(y_arr[idx], SGF_WINDOW, SGF_POLYORDER, deriv=0)
            vx_sg[idx] = savgol_filter(x_arr[idx], SGF_WINDOW, SGF_POLYORDER,
                                       deriv=1, delta=1.0 / CAMERA_FPS)
            vy_sg[idx] = savgol_filter(y_arr[idx], SGF_WINDOW, SGF_POLYORDER,
                                       deriv=1, delta=1.0 / CAMERA_FPS)

    full["x_sg"]       = x_sg
    full["y_sg"]       = y_sg
    full["speed_cm_s"] = np.sqrt(vx_sg**2 + vy_sg**2)
    full["speed_ok"]   = (
        full["ok_raw"]
        & ~full["is_oob"]
        & (full["speed_cm_s"] >= MIN_SPEED_CM_S)
        & (full["speed_cm_s"] <= MAX_SPEED_CM_S)
    )
    return full


print("\nLoading SLEAP T1 ...")
raw_t1      = load_sleap_raw(SLEAP_T1)
n_frames_t1 = int(raw_t1["frame_idx"].max()) + 1
beh_t1      = preprocess_behavior(raw_t1, n_frames=n_frames_t1)

print("Loading SLEAP T2 ...")
raw_t2      = load_sleap_raw(SLEAP_T2)
n_frames_t2 = int(raw_t2["frame_idx"].max()) + 1
beh_t2      = preprocess_behavior(raw_t2, n_frames=n_frames_t2)


def _beh_summary(name: str, df: pd.DataFrame):
    n       = len(df)
    n_raw   = int(df["ok_raw"].sum())
    n_oob   = int(df["is_oob"].sum())
    n_spd   = int(df["speed_ok"].sum())
    spd_med = float(df.loc[df["ok_raw"], "speed_cm_s"].median())
    spd_p95 = float(df.loc[df["ok_raw"], "speed_cm_s"].quantile(0.95))
    print(f"  {name}: {n} frames | "
          f"raw ok={n_raw} ({100*n_raw/n:.1f}%) | "
          f"oob={n_oob} ({100*n_oob/n:.1f}%) | "
          f"speed ok={n_spd} ({100*n_spd/n:.1f}%) | "
          f"speed median={spd_med:.1f}  p95={spd_p95:.1f} cm/s")

print("\nBehavior summary:")
_beh_summary("T1", beh_t1)
_beh_summary("T2", beh_t2)


def _traj_ax(ax, df, mask, name):
    ok     = df["ok_raw"].to_numpy() & mask
    oob    = df["is_oob"].to_numpy() & mask
    spd_ok = df["speed_ok"].to_numpy() & mask
    ax.plot(df.loc[ok, "x_sg"], df.loc[ok, "y_sg"],
            lw=0.4, alpha=0.25, color=WONG_SKY_BLUE, zorder=1, label="tracking ok")
    ax.scatter(df.loc[oob, "x_cm_raw"], df.loc[oob, "y_cm_raw"],
               s=4, color=WONG_ORANGE, alpha=0.8, zorder=2, label=f"oob (n={oob.sum()})")
    ax.scatter(df.loc[spd_ok, "x_sg"], df.loc[spd_ok, "y_sg"],
               s=0.8, color=WONG_VERMILLION, alpha=0.35, zorder=3,
               label=f"speed ok (n={spd_ok.sum()})")
    rect = plt.Rectangle((0, 0), W_CM, H_CM,
                          lw=1.5, edgecolor="black", facecolor="none", zorder=4)
    ax.add_patch(rect)
    ax.set_xlim(-5, W_CM + 5); ax.set_ylim(H_CM + 5, -5)
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
    ax.set_title(name)
    ax.legend(markerscale=5, fontsize=7, loc="upper right")

def _spd_ax(ax, df, mask, name):
    ok  = df["ok_raw"].to_numpy() & mask
    spd = df.loc[ok, "speed_cm_s"].to_numpy()
    ax.hist(spd, bins=100, color=WONG_SKY_BLUE, alpha=0.8)
    ax.axvline(MIN_SPEED_CM_S, color=WONG_VERMILLION, lw=1.5, ls="--",
               label=f"min {MIN_SPEED_CM_S} cm/s")
    ax.axvline(MAX_SPEED_CM_S, color=WONG_ORANGE,     lw=1.5, ls="--",
               label=f"max {MAX_SPEED_CM_S} cm/s")
    ax.set_xlim(*SPD_HIST_XLIM)
    ax.set_xlabel("speed (cm/s)"); ax.set_ylabel("frames")
    ax.set_title(f"{name} — speed")
    ax.legend(fontsize=7)

pre_mask  = beh_t1["frame_idx"].to_numpy() <  DOOR_LIFT_FRAME_T1
post_mask = beh_t1["frame_idx"].to_numpy() >= DOOR_LIFT_FRAME_T1
all_t2    = np.ones(len(beh_t2), dtype=bool)

panels = [
    (beh_t1, pre_mask,  "T1 pre  (frame < {})".format(DOOR_LIFT_FRAME_T1)),
    (beh_t1, post_mask, "T1 post (frame >= {})".format(DOOR_LIFT_FRAME_T1)),
    (beh_t2, all_t2,    "T2"),
]

fig, axes = plt.subplots(3, 2, figsize=(13, 13), constrained_layout=True)
for row, (df, mask, name) in enumerate(panels):
    _traj_ax(axes[row, 0], df, mask, name)
    _spd_ax( axes[row, 1], df, mask, name)
fig.suptitle("Behavior QC — Cell 1", fontsize=13, fontweight="bold")

if SAVE_OUTPUTS:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    beh_t1.to_csv(OUT_DIR / "beh_t1.csv", index=False)
    beh_t2.to_csv(OUT_DIR / "beh_t2.csv", index=False)
    fig.savefig(OUT_DIR / "behavior_qc.svg")
    print("Saved -> beh_t1.csv, beh_t2.csv, behavior_qc.svg")

plt.show()
print("\nCell 1 done.")
print("Available: beh_t1, beh_t2, W_CM, H_CM, XLIM_CM, YLIM_CM")


# %%
# =============================================================================
# CELL 2 — XCLOCK TIMEBASES
# =============================================================================

if "beh_t1" not in dir():
    if (OUT_DIR / "beh_t1.csv").exists():
        beh_t1 = pd.read_csv(OUT_DIR / "beh_t1.csv")
        beh_t2 = pd.read_csv(OUT_DIR / "beh_t2.csv")
        print("Loaded beh_t1/t2 from file cache.")
    else:
        raise RuntimeError("beh_t1/t2 not found — run Cell 1 first or enable SAVE_OUTPUTS.")
else:
    print("Using beh_t1/t2 from memory.")


def load_xclock_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, dtype=str)
    if raw.shape[1] == 1:
        df = raw.iloc[:, 0].str.split(",", expand=True).iloc[:, :3]
    else:
        df = raw.iloc[:, :3].copy()
    df.columns = ["t_raw", "code", "ts_raw"]
    df["t_raw"]  = pd.to_numeric(df["t_raw"],  errors="coerce")
    df["code"]   = pd.to_numeric(df["code"],   errors="coerce")
    df["ts_raw"] = pd.to_numeric(df["ts_raw"], errors="coerce")
    df = df.dropna(subset=["t_raw", "code"]).sort_values("t_raw").reset_index(drop=True)
    df["channel"] = df["code"].abs().astype(int)
    df["edge"]    = np.where(df["code"] > 0, "rising", "falling")
    return df


def calibrate_traw_to_seconds(df: pd.DataFrame) -> tuple[float, float]:
    ts = df["ts_raw"].dropna()
    if ts.nunique() < 2:
        raise ValueError("Need >= 2 distinct ts_raw values to calibrate duration.")
    duration_s = (ts.max() - ts.min()) / 1e9
    t_span = df["t_raw"].iloc[-1] - df["t_raw"].iloc[0]
    if not np.isfinite(t_span) or t_span <= 0:
        raise ValueError("Invalid t_raw span.")
    return float(duration_s), float(duration_s / t_span)


def extract_timebase(path: Path, channel: int, edge: str) -> tuple[np.ndarray, dict]:
    df = load_xclock_csv(path)
    duration_s, sec_per_traw = calibrate_traw_to_seconds(df)
    ev = df[(df["channel"] == channel) & (df["edge"] == edge)].sort_values("t_raw")
    if len(ev) < 2:
        raise ValueError(f"Not enough events on ch={channel} edge={edge} in {path.name}")
    t_s = (ev["t_raw"] - ev["t_raw"].iloc[0]) * sec_per_traw
    t_s = t_s.to_numpy(dtype=float)
    dt  = np.diff(t_s)
    meta = {
        "file": path.name, "channel": channel, "edge": edge,
        "duration_s": duration_s, "n_pulses": len(t_s),
        "dt_med_s": float(np.median(dt)),
        "dt_min_s": float(dt.min()), "dt_max_s": float(dt.max()),
        "fps_nominal": round(1.0 / float(np.median(dt)), 2),
    }
    return t_s, meta


def build_camera_timebase(path: Path) -> tuple[pd.DataFrame, dict]:
    t_s, meta = extract_timebase(path, CH_CAM, EDGE)
    cam = pd.DataFrame({"frame_idx": np.arange(len(t_s), dtype=int), "t_cam_s": t_s})
    return cam, meta

def build_ca30_timebase(path: Path) -> tuple[np.ndarray, dict]:
    return extract_timebase(path, CH_CA30, EDGE)


print("Loading XClock T1 ...")
cam_t1,  cam_meta_t1  = build_camera_timebase(XCLOCK_T1)
ca30_t1, ca30_meta_t1 = build_ca30_timebase(XCLOCK_T1)
print("Loading XClock T2 ...")
cam_t2,  cam_meta_t2  = build_camera_timebase(XCLOCK_T2)
ca30_t2, ca30_meta_t2 = build_ca30_timebase(XCLOCK_T2)

dt_join     = float(np.nanmedian([np.median(np.diff(ca30_t1)), np.median(np.diff(ca30_t2))]))
t1_ca_comp  = ca30_t1 - ca30_t1[0]
t2_ca_comp  = (ca30_t2 - ca30_t2[0]) + (t1_ca_comp[-1] + dt_join)
t_ca30_comp = np.concatenate([t1_ca_comp, t2_ca_comp])
t1_cam_comp = cam_t1["t_cam_s"].to_numpy(dtype=float)
t2_cam_comp = cam_t2["t_cam_s"].to_numpy(dtype=float) + (t1_ca_comp[-1] + dt_join)
t_cam_comp  = np.concatenate([t1_cam_comp, t2_cam_comp])
n_cam_t1    = len(cam_t1)
t_boundary  = float(t1_ca_comp[-1] + 0.5 * dt_join)


def _tb_summary(label, meta):
    print(f"  {label}: {meta['n_pulses']} pulses | "
          f"duration={meta['duration_s']:.1f} s | "
          f"dt median={meta['dt_med_s']*1000:.2f} ms | "
          f"nominal fps={meta['fps_nominal']} | "
          f"dt range=[{meta['dt_min_s']*1000:.2f}, {meta['dt_max_s']*1000:.2f}] ms")

print("\nCamera timebases:")
_tb_summary("T1", cam_meta_t1); _tb_summary("T2", cam_meta_t2)
print("\nCa30 timebases:")
_tb_summary("T1", ca30_meta_t1); _tb_summary("T2", ca30_meta_t2)
print(f"\nComposite: dt_join={dt_join*1000:.2f} ms | t_boundary={t_boundary:.2f} s | n_cam_t1={n_cam_t1}")
print(f"  t_cam_comp:  {len(t_cam_comp)} frames  [{t_cam_comp[0]:.2f} -> {t_cam_comp[-1]:.2f} s]")
print(f"  t_ca30_comp: {len(t_ca30_comp)} pulses  [{t_ca30_comp[0]:.2f} -> {t_ca30_comp[-1]:.2f} s]")

if SAVE_OUTPUTS:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cam_t1.to_csv(OUT_DIR / "cam_t1.csv", index=False)
    cam_t2.to_csv(OUT_DIR / "cam_t2.csv", index=False)
    np.savez(OUT_DIR / "xclock_timebases.npz",
             ca30_t1=ca30_t1, ca30_t2=ca30_t2,
             t_cam_comp=t_cam_comp, t_ca30_comp=t_ca30_comp,
             t_boundary=np.array([t_boundary]), n_cam_t1=np.array([n_cam_t1]))
    print("Saved -> cam_t1.csv, cam_t2.csv, xclock_timebases.npz")

fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
for col, (t_cam, t_ca30, name) in enumerate([
    (t1_cam_comp[:n_cam_t1], t1_ca_comp, "T1"),
    (t2_cam_comp - (t1_ca_comp[-1] + dt_join),
     t2_ca_comp  - (t1_ca_comp[-1] + dt_join), "T2"),
]):
    ax = axes[0, col]
    dt_cam = np.diff(t_cam) * 1000
    ax.plot(dt_cam, lw=0.5, color=WONG_BLUE)
    ax.axhline(float(np.median(dt_cam)), color=WONG_VERMILLION, lw=1.2, ls="--",
               label=f"median {np.median(dt_cam):.2f} ms")
    ax.set_xlabel("pulse index"); ax.set_ylabel("interval (ms)")
    ax.set_title(f"{name} — camera pulse intervals"); ax.legend(fontsize=8)

    ax2 = axes[1, col]
    dt_ca = np.diff(t_ca30) * 1000
    ax2.plot(dt_ca, lw=0.5, color=WONG_GREEN)
    ax2.axhline(float(np.median(dt_ca)), color=WONG_VERMILLION, lw=1.2, ls="--",
                label=f"median {np.median(dt_ca):.2f} ms")
    ax2.set_xlabel("pulse index"); ax2.set_ylabel("interval (ms)")
    ax2.set_title(f"{name} — Ca30 pulse intervals"); ax2.legend(fontsize=8)

fig.suptitle("XClock QC — Cell 2", fontsize=13, fontweight="bold")
if SAVE_OUTPUTS:
    fig.savefig(OUT_DIR / "xclock_qc.svg")
    print("Saved -> xclock_qc.svg")
plt.show()
print("\nCell 2 done.")
print("Available: cam_t1, cam_t2, ca30_t1, ca30_t2")
print("           t_cam_comp, t_ca30_comp, t_boundary, n_cam_t1")


# %%
# =============================================================================
# CELL 3 — CALCIUM LOADING + ALIGNMENT
# =============================================================================

if "t_cam_comp" not in dir():
    if (OUT_DIR / "xclock_timebases.npz").exists():
        _tb = np.load(OUT_DIR / "xclock_timebases.npz", allow_pickle=True)
        t_cam_comp  = _tb["t_cam_comp"]
        t_ca30_comp = _tb["t_ca30_comp"]
        t_boundary  = float(_tb["t_boundary"][0])
        n_cam_t1    = int(_tb["n_cam_t1"][0])
        cam_t1 = pd.read_csv(OUT_DIR / "cam_t1.csv")
        cam_t2 = pd.read_csv(OUT_DIR / "cam_t2.csv")
        print("Loaded xclock timebases from file cache.")
    else:
        raise RuntimeError("XClock timebases not found — run Cell 2 first or enable SAVE_OUTPUTS.")
else:
    print("Using xclock timebases from memory.")

if "beh_t1" not in dir():
    if (OUT_DIR / "beh_t1.csv").exists():
        beh_t1 = pd.read_csv(OUT_DIR / "beh_t1.csv")
        beh_t2 = pd.read_csv(OUT_DIR / "beh_t2.csv")
        print("Loaded beh_t1/t2 from file cache.")
    else:
        raise RuntimeError("beh_t1/t2 not found — run Cell 1 first or enable SAVE_OUTPUTS.")
else:
    print("Using beh_t1/t2 from memory.")


def load_ca_matrix(path: Path) -> tuple[np.ndarray, list]:
    df = _read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] == 0:
        raise ValueError(f"No numeric columns in {path.name}")
    mat   = num_df.to_numpy(dtype=float)
    names = list(num_df.columns)
    if mat.shape[0] < mat.shape[1] and mat.shape[1] > 2000 and mat.shape[0] < 500:
        mat   = mat.T
        names = [f"cell_{i}" for i in range(mat.shape[1])]
    if mat.ndim != 2 or mat.shape[0] < 2 or mat.shape[1] < 1:
        raise ValueError(f"Unexpected calcium matrix shape: {mat.shape}")
    return mat, names


def nearest_index(t_query: np.ndarray, t_ref: np.ndarray) -> np.ndarray:
    j         = np.searchsorted(t_ref, t_query, side="left")
    j         = np.clip(j, 1, len(t_ref) - 1)
    left      = j - 1
    use_right = (t_query - t_ref[left]) > (t_ref[j] - t_query)
    return np.where(use_right, j, left)


print("Loading calcium matrix ...")
ca_mat, cell_names = load_ca_matrix(CA_FILE)
n_ca, n_cells = ca_mat.shape
print(f"  shape: {n_ca} samples x {n_cells} cells")

t_ca = np.linspace(float(t_ca30_comp[0]), float(t_ca30_comp[-1]), n_ca)

j_cam          = nearest_index(t_ca, t_cam_comp).astype(int)
delta_s        = t_ca - t_cam_comp[j_cam]
cam_concat_idx = j_cam
beh_frame_idx  = np.where(cam_concat_idx < n_cam_t1,
                           cam_concat_idx,
                           cam_concat_idx - n_cam_t1).astype(int)

print("Alignment QC |delta_s| percentiles (p50/p90/p99) s:",
      np.round(np.nanpercentile(np.abs(delta_s), [50, 90, 99]), 5))

trial_arr = np.where(cam_concat_idx < n_cam_t1, "T1", "T2").astype(object)
print(f"  Trial split — T1: {(trial_arr=='T1').sum()}  T2: {(trial_arr=='T2').sum()}")

beh_x        = np.full(n_ca, np.nan, dtype=float)
beh_y        = np.full(n_ca, np.nan, dtype=float)
beh_speed_ok = np.zeros(n_ca, dtype=bool)
beh_is_oob   = np.zeros(n_ca, dtype=bool)

for trial_label, beh_df in [("T1", beh_t1), ("T2", beh_t2)]:
    idx       = np.where(trial_arr == trial_label)[0]
    frames    = beh_frame_idx[idx]
    reindexed = beh_df.set_index("frame_idx").reindex(frames)
    beh_x[idx]        = reindexed["x_sg"].to_numpy(dtype=float)
    beh_y[idx]        = reindexed["y_sg"].to_numpy(dtype=float)
    beh_speed_ok[idx] = reindexed["speed_ok"].to_numpy(dtype=bool)
    beh_is_oob[idx]   = reindexed["is_oob"].to_numpy(dtype=bool)

n_finite = int((np.isfinite(beh_x) & np.isfinite(beh_y)).sum())
print(f"  Behavior attached — finite xy: {n_finite}/{n_ca} ({100*n_finite/n_ca:.1f}%)")

base_valid   = np.isfinite(beh_x) & np.isfinite(beh_y) & beh_speed_ok & ~beh_is_oob
mask_T1_pre  = (trial_arr == "T1") & base_valid & (beh_frame_idx <  DOOR_LIFT_FRAME_T1)
mask_T1_post = (trial_arr == "T1") & base_valid & (beh_frame_idx >= DOOR_LIFT_FRAME_T1)
mask_T2      = (trial_arr == "T2") & base_valid

print(f"\nCondition mask counts:")
print(f"  T1 pre:  {mask_T1_pre.sum()}")
print(f"  T1 post: {mask_T1_post.sum()}")
print(f"  T2:      {mask_T2.sum()}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
ax = axes[0]
ax.hist(delta_s * 1000, bins=100, color=WONG_BLUE, alpha=0.8)
ax.axvline(0, color=WONG_VERMILLION, lw=1.2, ls="--")
ax.set_xlabel("delta (ms)"); ax.set_ylabel("calcium samples")
ax.set_title("Alignment error distribution  (Ca -> nearest cam frame)")
ax2 = axes[1]
ax2.plot(t_ca, delta_s * 1000, lw=0.4, alpha=0.6, color=WONG_BLUE)
ax2.axhline(0, color=WONG_VERMILLION, lw=1.2, ls="--")
ax2.axvline(t_boundary, color=WONG_ORANGE, lw=1.2, ls="--", label="T1/T2 boundary")
ax2.set_xlabel("calcium time (s)"); ax2.set_ylabel("delta (ms)")
ax2.set_title("Alignment error over time"); ax2.legend(fontsize=8)
fig.suptitle("Calcium Alignment QC — Cell 3", fontsize=13, fontweight="bold")

if SAVE_OUTPUTS:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "calcium_alignment_qc.svg")
    align_df = pd.DataFrame({
        "ca_idx": np.arange(n_ca, dtype=int), "trial": trial_arr,
        "t_ca_s": t_ca, "t_cam_s": t_cam_comp[j_cam], "delta_s": delta_s,
        "cam_concat_idx": cam_concat_idx, "beh_frame_idx": beh_frame_idx,
        "beh_x": beh_x, "beh_y": beh_y,
        "speed_ok": beh_speed_ok, "is_oob": beh_is_oob,
        "mask_T1_pre": mask_T1_pre, "mask_T1_post": mask_T1_post, "mask_T2": mask_T2,
    })
    align_df.to_csv(OUT_DIR / "calcium_alignment.csv", index=False)
    np.savez(OUT_DIR / "calcium_aligned.npz",
             ca_mat=ca_mat, t_ca=t_ca, trial_arr=trial_arr.astype(str),
             beh_frame_idx=beh_frame_idx, beh_x=beh_x, beh_y=beh_y,
             beh_speed_ok=beh_speed_ok, beh_is_oob=beh_is_oob, delta_s=delta_s,
             mask_T1_pre=mask_T1_pre, mask_T1_post=mask_T1_post, mask_T2=mask_T2)
    print("Saved -> calcium_alignment_qc.svg, calcium_alignment.csv, calcium_aligned.npz")

plt.show()
print("\nCell 3 done.")
print("Available: ca_mat, cell_names, t_ca, trial_arr, beh_frame_idx")
print("           beh_x, beh_y, beh_speed_ok, beh_is_oob, delta_s")
print("           mask_T1_pre, mask_T1_post, mask_T2")


# %%
# =============================================================================
# CELL 4 — PLACE CELL ANALYSIS
# =============================================================================

if "ca_mat" not in dir():
    if (OUT_DIR / "calcium_aligned.npz").exists():
        _ca = np.load(OUT_DIR / "calcium_aligned.npz", allow_pickle=True)
        ca_mat        = _ca["ca_mat"]
        t_ca          = _ca["t_ca"]
        trial_arr     = _ca["trial_arr"].astype(object)
        beh_frame_idx = _ca["beh_frame_idx"]
        beh_x         = _ca["beh_x"]
        beh_y         = _ca["beh_y"]
        beh_speed_ok  = _ca["beh_speed_ok"].astype(bool)
        beh_is_oob    = _ca["beh_is_oob"].astype(bool)
        mask_T1_pre   = _ca["mask_T1_pre"].astype(bool)
        mask_T1_post  = _ca["mask_T1_post"].astype(bool)
        mask_T2       = _ca["mask_T2"].astype(bool)
        n_ca, n_cells = ca_mat.shape
        print("Loaded calcium data from file cache.")
    else:
        raise RuntimeError("Calcium data not found — run Cell 3 first or enable SAVE_OUTPUTS.")
else:
    print("Using calcium data from memory.")

if "beh_t1" not in dir():
    if (OUT_DIR / "beh_t1.csv").exists():
        beh_t1 = pd.read_csv(OUT_DIR / "beh_t1.csv")
        beh_t2 = pd.read_csv(OUT_DIR / "beh_t2.csv")
        print("Loaded beh_t1/t2 from file cache.")
    else:
        raise RuntimeError("beh_t1/t2 not found — run Cell 1 first or enable SAVE_OUTPUTS.")
else:
    print("Using beh_t1/t2 from memory.")


def binarize_events_zscore(
    t_s: np.ndarray, s_trace: np.ndarray,
    zthr: float = Z_THRESH, refractory_s: float = REFRACTORY_S,
    z_on_nonzero: bool = Z_ON_NONZERO,
) -> np.ndarray:
    x = np.asarray(s_trace, float)
    x[~np.isfinite(x)] = 0.0
    if z_on_nonzero:
        pos = x[x > 0]
        if pos.size < 10:
            return np.zeros_like(x, dtype=bool)
        mu, sd = float(np.mean(pos)), float(np.std(pos))
    else:
        mu, sd = float(np.mean(x)), float(np.std(x))
    if not np.isfinite(sd) or sd <= 0:
        return np.zeros_like(x, dtype=bool)
    z      = (x - mu) / sd
    above  = z > zthr
    onsets = np.where(above & ~np.r_[False, above[:-1]])[0]
    if onsets.size == 0:
        return np.zeros_like(x, dtype=bool)
    keep, last_t = [], -np.inf
    for i in onsets:
        if t_s[i] - last_t >= refractory_s:
            keep.append(i); last_t = t_s[i]
    ev = np.zeros_like(x, dtype=bool)
    ev[np.array(keep, dtype=int)] = True
    return ev


def spatial_rate_map(
    x: np.ndarray, y: np.ndarray, t_s: np.ndarray, events_bool: np.ndarray,
    xlim: tuple, ylim: tuple,
    bin_cm: float = BIN_CM, occ_min_s: float = OCC_MIN_S,
    smooth_sigma_bins: float = SMOOTH_SIGMA_BINS,
) -> dict:
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(t_s)
    x  = np.asarray(x)[ok];  y  = np.asarray(y)[ok]
    t_s = np.asarray(t_s)[ok]; events_bool = np.asarray(events_bool)[ok]
    if t_s.size < 10:
        raise ValueError("Too few samples to build a rate map.")
    dt  = np.diff(t_s, prepend=t_s[0])
    dt0 = float(np.nanmedian(np.diff(t_s)))
    if not np.isfinite(dt0) or dt0 <= 0:
        raise ValueError("Bad timebase for occupancy.")
    dt[0] = dt0
    dt = np.clip(dt, 0.25 * dt0, 4.0 * dt0)
    xmin, xmax = xlim; ymin, ymax = ylim
    nx = max(1, int(round((xmax - xmin) / bin_cm)))
    ny = max(1, int(round((ymax - ymin) / bin_cm)))
    occ_s, _, _ = np.histogram2d(x, y, bins=[nx, ny],
                                 range=[[xmin, xmax], [ymin, ymax]], weights=dt)
    evc, _, _   = np.histogram2d(x[events_bool], y[events_bool], bins=[nx, ny],
                                 range=[[xmin, xmax], [ymin, ymax]])
    if smooth_sigma_bins > 0:
        occ_sm = gaussian_filter(occ_s, sigma=smooth_sigma_bins)
        evc_sm = gaussian_filter(evc,   sigma=smooth_sigma_bins)
    else:
        occ_sm, evc_sm = occ_s, evc
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = evc_sm / occ_sm
    rate[occ_sm < occ_min_s] = np.nan
    return {"rate": rate.T, "occ_s": occ_s.T, "evc": evc.T,
            "nx": nx, "ny": ny, "dt0": dt0}


def spatial_info_bits_per_event(rate_map: np.ndarray, occ_s_map: np.ndarray) -> float:
    r = np.asarray(rate_map, float); o = np.asarray(occ_s_map, float)
    m = np.isfinite(r) & np.isfinite(o) & (o > 0)
    if m.sum() < 5:
        return np.nan
    p    = o[m] / np.nansum(o[m])
    rbar = np.nansum(p * r[m])
    if not np.isfinite(rbar) or rbar <= 0:
        return np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        info = np.nansum(p * (r[m] / rbar) * np.log2(r[m] / rbar))
    return float(info)


def circular_permutation_pvalue(
    x_cond: np.ndarray, y_cond: np.ndarray,
    t_cond: np.ndarray, events_bool: np.ndarray,
    n_perm: int = N_PERM, rng_seed: int = RNG_SEED,
) -> tuple[float, float]:
    T = len(t_cond)
    events_bool = np.asarray(events_bool, dtype=bool)
    if T == 0 or events_bool.size != T or events_bool.sum() == 0:
        return np.nan, np.nan
    def _info(ev):
        rm = spatial_rate_map(x_cond, y_cond, t_cond, ev, xlim=XLIM_CM, ylim=YLIM_CM)
        return spatial_info_bits_per_event(rm["rate"], rm["occ_s"])
    obs = _info(events_bool)
    rng = np.random.default_rng(rng_seed)
    null_ge = 0
    for _ in range(n_perm):
        if _info(np.roll(events_bool, int(rng.integers(0, T)))) >= obs:
            null_ge += 1
    return float((null_ge + 1.0) / (n_perm + 1.0)), float(obs)


def run_condition(name: str, cond_mask: np.ndarray, topN: int = 12) -> pd.DataFrame:
    cond_mask = np.asarray(cond_mask, dtype=bool)
    n_samples = int(cond_mask.sum())
    print(f"\n{'='*60}\n{name}: {n_samples} samples")
    if n_samples == 0:
        raise ValueError(f"{name}: 0 samples — check upstream masks/alignment.")

    t_cond  = t_ca[cond_mask]
    x_cond  = beh_x[cond_mask]
    y_cond  = beh_y[cond_mask]
    S_cond  = ca_mat[cond_mask, :]
    n_cells = S_cond.shape[1]

    print(f"Binarising events ({n_cells} cells) ...")
    events_list  = []
    event_counts = np.zeros(n_cells, dtype=int)
    for ci in range(n_cells):
        ev = binarize_events_zscore(t_cond, S_cond[:, ci])
        events_list.append(ev)
        event_counts[ci] = int(ev.sum())

    eligible = np.where(event_counts >= N_MIN_EVENTS)[0]
    print(f"Eligible cells (>= {N_MIN_EVENTS} events): {eligible.size}/{n_cells}")
    if eligible.size == 0:
        print("WARNING: no eligible cells — falling back to top by event count.")
        eligible = np.argsort(event_counts)[::-1][:min(topN, n_cells)]

    infos = np.full(n_cells, np.nan)
    peaks = np.full(n_cells, np.nan)
    pvals = np.full(n_cells, np.nan)

    t_start = time.perf_counter()
    for k, ci in enumerate(eligible):
        rm        = spatial_rate_map(x_cond, y_cond, t_cond, events_list[ci],
                                     xlim=XLIM_CM, ylim=YLIM_CM)
        infos[ci] = spatial_info_bits_per_event(rm["rate"], rm["occ_s"])
        peaks[ci] = float(np.nanmax(rm["rate"])) if np.isfinite(rm["rate"]).any() else np.nan
        pvals[ci], _ = circular_permutation_pvalue(x_cond, y_cond, t_cond, events_list[ci])
        done    = k + 1
        elapsed = time.perf_counter() - t_start
        rate_ps = done / elapsed if elapsed > 0 else float("inf")
        eta     = (eligible.size - done) / rate_ps if rate_ps > 0 else float("inf")
        print(f"\r  Permutation test: {done}/{eligible.size} | "
              f"{elapsed:.1f}s elapsed | ETA {eta:.1f}s",
              end="" if done < eligible.size else "\n", flush=True)

    sig       = np.isfinite(pvals) & (pvals <= ALPHA)
    valid     = np.isfinite(infos)
    sig_valid = sig & valid
    if np.any(sig_valid):
        ranked    = np.argsort(infos[sig_valid])[::-1]
        top_cells = np.where(sig_valid)[0][ranked][:topN]
        print(f"Top significant cells (p<={ALPHA}): {len(top_cells)}")
    else:
        ranked    = np.argsort(np.where(valid, infos, -np.inf))[::-1]
        top_cells = ranked[:topN]
        print(f"No significant cells — showing top {len(top_cells)} by spatial info.")

    n_plot      = len(top_cells)
    ncols       = min(3, n_plot)
    nrows       = int(np.ceil(n_plot / ncols)) if ncols > 0 else 1
    arena_ratio = W_CM / H_CM
    cell_w      = 5.0
    cell_h      = cell_w / arena_ratio

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(cell_w * ncols, cell_h * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    rows  = []
    fname = name.replace(" ", "_").lower()
    for k, ci in enumerate(top_cells):
        rm   = spatial_rate_map(x_cond, y_cond, t_cond, events_list[ci],
                                xlim=XLIM_CM, ylim=YLIM_CM)
        info = spatial_info_bits_per_event(rm["rate"], rm["occ_s"])
        peak = float(np.nanmax(rm["rate"])) if np.isfinite(rm["rate"]).any() else np.nan
        sig_marker = "*" if (np.isfinite(pvals[ci]) and pvals[ci] <= ALPHA) else ""

        rows.append({
            "condition":               name,
            "cell_idx":                int(ci),
            "n_events":                int(event_counts[ci]),
            "spatial_info_bits_event": float(info) if np.isfinite(info) else np.nan,
            "peak_rate_ev_s":          float(peak) if np.isfinite(peak) else np.nan,
            "p_value":                 float(pvals[ci]) if np.isfinite(pvals[ci]) else np.nan,
            "significant":             bool(np.isfinite(pvals[ci]) and pvals[ci] <= ALPHA),
        })

        ax = axes[k]
        im = ax.imshow(rm["rate"], origin="upper",
                       extent=[XLIM_CM[0], XLIM_CM[1], YLIM_CM[1], YLIM_CM[0]],
                       interpolation="nearest", aspect="equal", cmap=cmap)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}  Cell {ci}{sig_marker}\n"
                     f"ev={event_counts[ci]}  info={info:.3g}  p={pvals[ci]:.3f}")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("events/s")

    for kk in range(n_plot, len(axes)):
        axes[kk].axis("off")

    fig.suptitle(f"Rate maps — {name}", fontsize=12, fontweight="bold")
    if SAVE_OUTPUTS:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_DIR / f"rate_maps_{fname}.svg")
        print(f"Saved -> rate_maps_{fname}.svg")
    plt.show()

    if n_plot > 0:
        ci0 = int(top_cells[0])
        ev0 = events_list[ci0]
        sig_marker = "*" if (np.isfinite(pvals[ci0]) and pvals[ci0] <= ALPHA) else ""

        if "T1" in name:
            beh_df  = beh_t1
            fr_mask = (beh_df["frame_idx"] < DOOR_LIFT_FRAME_T1) if "pre" in name \
                      else (beh_df["frame_idx"] >= DOOR_LIFT_FRAME_T1)
        else:
            beh_df  = beh_t2
            fr_mask = np.ones(len(beh_t2), dtype=bool)

        traj_ok = beh_df["ok_raw"].to_numpy() & fr_mask
        tx = beh_df.loc[traj_ok, "x_sg"].to_numpy()
        ty = beh_df.loc[traj_ok, "y_sg"].to_numpy()

        fig2, ax2 = plt.subplots(figsize=(cell_w, cell_h), constrained_layout=True)
        ax2.plot(tx, ty, lw=0.6, alpha=0.3, color=WONG_SKY_BLUE)
        ax2.scatter(x_cond[ev0], y_cond[ev0], s=12, color=WONG_VERMILLION, zorder=3)
        rect = plt.Rectangle((0, 0), W_CM, H_CM,
                              lw=1.5, edgecolor="black", facecolor="none", zorder=4)
        ax2.add_patch(rect)
        ax2.set_xlim(-3, W_CM + 3); ax2.set_ylim(H_CM + 3, -3)
        ax2.set_xlabel("x (cm)"); ax2.set_ylabel("y (cm)")
        ax2.set_title(f"{name} — trajectory + events  (cell {ci0}{sig_marker})")
        if SAVE_OUTPUTS:
            fig2.savefig(OUT_DIR / f"trajectory_events_{fname}_cell{ci0}.svg")
            print(f"Saved -> trajectory_events_{fname}_cell{ci0}.svg")
        plt.show()

    return pd.DataFrame(rows)


df_pre  = run_condition("T1 pre",  mask_T1_pre,  topN=12)
df_post = run_condition("T1 post", mask_T1_post, topN=12)
df_t2   = run_condition("T2",      mask_T2,      topN=12)

summary_all = (pd.concat([df_pre, df_post, df_t2], ignore_index=True)
               .sort_values(["condition", "spatial_info_bits_event"],
                            ascending=[True, False])
               .reset_index(drop=True))

print("\nSummary (top cells per condition):")
print(summary_all.to_string(index=False))

if SAVE_OUTPUTS:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_all.to_csv(OUT_DIR / "placecell_summary.csv", index=False)
    print("\nSaved -> placecell_summary.csv")

print("\nCell 4 done.")
print("Available: df_pre, df_post, df_t2, summary_all")


# %%
# =============================================================================
# CELL 4.5 — UPSAMPLED RATE MAP DISPLAY
#
# Replot rate maps from Cell 4 with upsampled + smoothed display.
# Stats are NOT recomputed — reuses Cell 4 results from summary_all.
# Tune UPSAMPLE_FACTOR and DISPLAY_SIGMA at the top of this cell.
# =============================================================================

from scipy.ndimage import zoom

UPSAMPLE_FACTOR = 2     # bilinear upsample before smoothing
DISPLAY_SIGMA   = 2.0   # Gaussian sigma in upsampled bins
DISPLAY_TOPN    = 12


def upsample_rate_map(rate: np.ndarray, factor: int, sigma: float) -> np.ndarray:
    nan_mask    = ~np.isfinite(rate)
    rate_filled = np.where(nan_mask, 0.0, rate)
    rate_up     = zoom(rate_filled, factor, order=1)
    nan_up      = zoom(nan_mask.astype(float), factor, order=0) > 0.5
    rate_up     = gaussian_filter(rate_up, sigma=sigma)
    rate_up[nan_up] = np.nan
    return rate_up


def replot_condition(name: str, cond_mask: np.ndarray,
                     summary_df: pd.DataFrame, topN: int = DISPLAY_TOPN):
    cond_mask = np.asarray(cond_mask, dtype=bool)
    t_cond    = t_ca[cond_mask]
    x_cond    = beh_x[cond_mask]
    y_cond    = beh_y[cond_mask]
    S_cond    = ca_mat[cond_mask, :]

    cond_rows = summary_df[summary_df["condition"] == name].head(topN)
    if len(cond_rows) == 0:
        print(f"{name}: no cells in summary_df — run Cell 4 first.")
        return

    top_cells = cond_rows["cell_idx"].to_numpy(dtype=int)
    n_events  = cond_rows["n_events"].to_numpy(dtype=int)
    infos     = cond_rows["spatial_info_bits_event"].to_numpy(dtype=float)
    pvals     = cond_rows["p_value"].to_numpy(dtype=float)
    sig_flags = cond_rows["significant"].to_numpy(dtype=bool)

    n_plot      = len(top_cells)
    ncols       = min(3, n_plot)
    nrows       = int(np.ceil(n_plot / ncols)) if ncols > 0 else 1
    arena_ratio = W_CM / H_CM
    cell_w      = 5.0
    cell_h      = cell_w / arena_ratio

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(cell_w * ncols, cell_h * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for k, (ci, nev, info, pval, sig) in enumerate(
            zip(top_cells, n_events, infos, pvals, sig_flags)):
        ev           = binarize_events_zscore(t_cond, S_cond[:, ci])
        rm           = spatial_rate_map(x_cond, y_cond, t_cond, ev,
                                        xlim=XLIM_CM, ylim=YLIM_CM)
        rate_display = upsample_rate_map(rm["rate"], UPSAMPLE_FACTOR, DISPLAY_SIGMA)
        sig_marker   = "*" if sig else ""

        ax = axes[k]
        im = ax.imshow(rate_display, origin="upper",
                       extent=[XLIM_CM[0], XLIM_CM[1], YLIM_CM[1], YLIM_CM[0]],
                       interpolation="bilinear", aspect="equal", cmap=cmap)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}  Cell {ci}{sig_marker}\n"
                     f"ev={nev}  info={info:.3g}  p={pval:.3f}")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("events/s")

    for kk in range(n_plot, len(axes)):
        axes[kk].axis("off")

    fig.suptitle(f"Rate maps (x{UPSAMPLE_FACTOR} upsample, sigma={DISPLAY_SIGMA}) — {name}",
                 fontsize=12, fontweight="bold")

    if SAVE_OUTPUTS:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fname = name.replace(" ", "_").lower()
        fig.savefig(OUT_DIR / f"rate_maps_display_{fname}.svg")
        print(f"Saved -> rate_maps_display_{fname}.svg")

    plt.show()


replot_condition("T1 pre",  mask_T1_pre,  summary_all, topN=DISPLAY_TOPN)
replot_condition("T1 post", mask_T1_post, summary_all, topN=DISPLAY_TOPN)
replot_condition("T2",      mask_T2,      summary_all, topN=DISPLAY_TOPN)

print("\nCell 4.5 done.")
print(f"Upsampled x{UPSAMPLE_FACTOR}, sigma={DISPLAY_SIGMA} bins.")
print("Tune UPSAMPLE_FACTOR and DISPLAY_SIGMA at the top of this cell.")


# %%
# =============================================================================
# CELL 5 — EXAMPLE TRACES + POSITION + SPEED
#
# 3-panel overview:
#   Panel 1: z-scored S traces (stacked), cell ID on y-axis
#   Panel 2: x_sg and y_sg position (cm) over frames
#   Panel 3: speed_cm_s with MIN_SPEED threshold
#
# Shaded regions = frames excluded by speed filter.
#
# Tune:
#   EXAMPLE_TRIAL   — "T1 pre" | "T1 post" | "T2"
#   N_EXAMPLE_CELLS — total cells (must be divisible by 3)
#   TRACE_SPACING   — vertical spacing between traces (z-score units)
#   FRAME_WINDOW    — (start, end) frame tuple, or None for full trial
# =============================================================================

EXAMPLE_TRIAL   = "T1 pre"
N_EXAMPLE_CELLS = 9
TRACE_SPACING   = 9.0
FRAME_WINDOW    = (0, 4500)     # tuple or None

# style — all Wong colorblind-safe
TRACE_COLOR       = "0.15"
TRACE_LW          = 0.7
X_COLOR           = WONG_BLUE
Y_COLOR           = WONG_PINK
SPEED_COLOR       = WONG_VERMILLION
EXCL_STRIPE_COLOR = WONG_ORANGE
EXCL_STRIPE_ALPHA = 0.20


def zscore_rows(X: np.ndarray) -> np.ndarray:
    X  = X.astype(float)
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X,  axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def binary_segments(frames: np.ndarray, flag: np.ndarray, value: int = 0):
    segs, in_seg, x0 = [], False, None
    for i, (f, v) in enumerate(zip(frames, flag)):
        if v == value and not in_seg:
            in_seg, x0 = True, f
        elif v != value and in_seg:
            segs.append((x0, frames[i - 1]))
            in_seg = False
    if in_seg:
        segs.append((x0, frames[-1]))
    return segs


def select_example_cells(condition: str, summary_df: pd.DataFrame, n: int = 9) -> list[int]:
    k         = n // 3
    cond_df   = summary_df[summary_df["condition"] == condition].copy()
    cond_df   = cond_df.sort_values("spatial_info_bits_event", ascending=False).reset_index(drop=True)
    sig_df    = cond_df[cond_df["significant"] == True]
    nonsig_df = cond_df[cond_df["significant"] == False]

    top_ids   = sig_df.head(k)["cell_idx"].tolist()
    n_sig     = len(sig_df)
    mid_start = max(0, (n_sig // 2) - 1)
    mid_ids   = sig_df.iloc[mid_start:mid_start + k]["cell_idx"].tolist()
    non_ids   = nonsig_df.tail(k)["cell_idx"].tolist()

    all_ids = top_ids + mid_ids + non_ids
    if len(all_ids) < n:
        for cid in cond_df["cell_idx"].tolist():
            if cid not in all_ids:
                all_ids.append(cid)
            if len(all_ids) >= n:
                break
    return [int(c) for c in all_ids[:n]]


_cond_map = {
    "T1 pre":  (beh_t1, beh_t1["frame_idx"].to_numpy() <  DOOR_LIFT_FRAME_T1),
    "T1 post": (beh_t1, beh_t1["frame_idx"].to_numpy() >= DOOR_LIFT_FRAME_T1),
    "T2":      (beh_t2, np.ones(len(beh_t2), dtype=bool)),
}
if EXAMPLE_TRIAL not in _cond_map:
    raise ValueError(f"EXAMPLE_TRIAL must be one of {list(_cond_map.keys())}")

beh_df, trial_frame_mask = _cond_map[EXAMPLE_TRIAL]
beh_df   = beh_df[trial_frame_mask].reset_index(drop=True)
frames   = beh_df["frame_idx"].to_numpy(dtype=float)
x_pos    = beh_df["x_sg"].to_numpy(dtype=float)
y_pos    = beh_df["y_sg"].to_numpy(dtype=float)
speed    = beh_df["speed_cm_s"].to_numpy(dtype=float)
speed_ok = beh_df["speed_ok"].to_numpy(dtype=int)

if FRAME_WINDOW is not None:
    fw0, fw1 = FRAME_WINDOW
    win      = (frames >= fw0) & (frames <= fw1)
    frames   = frames[win]
    x_pos    = x_pos[win]
    y_pos    = y_pos[win]
    speed    = speed[win]
    speed_ok = speed_ok[win]

T_plot = len(frames)

cell_ids       = select_example_cells(EXAMPLE_TRIAL, summary_all, n=N_EXAMPLE_CELLS)
print(f"Selected cells for {EXAMPLE_TRIAL}: {cell_ids}")

trial_label    = "T1" if "T1" in EXAMPLE_TRIAL else "T2"
ca_trial_idx   = np.where(trial_arr == trial_label)[0]
_beh_frames_ca = beh_frame_idx[ca_trial_idx]
_ca_S          = ca_mat[ca_trial_idx, :]

beh_frames_int = beh_df["frame_idx"].to_numpy(dtype=int)
if FRAME_WINDOW is not None:
    beh_frames_int = beh_frames_int[win]

_sort_idx      = np.argsort(_beh_frames_ca)
_sorted_frames = _beh_frames_ca[_sort_idx]
_mapped_ca     = nearest_index(beh_frames_int.astype(float), _sorted_frames.astype(float))
_mapped_ca     = _sort_idx[_mapped_ca]

traces = np.zeros((N_EXAMPLE_CELLS, T_plot), dtype=float)
for k, ci in enumerate(cell_ids):
    traces[k, :] = _ca_S[_mapped_ca, ci]

traces_z     = zscore_rows(traces)
baselines    = np.array([(N_EXAMPLE_CELLS - 1 - i) * TRACE_SPACING
                          for i in range(N_EXAMPLE_CELLS)], dtype=float)
traces_stack = traces_z + baselines[:, None]
segs_drop    = binary_segments(frames, speed_ok, value=0)

fig = plt.figure(figsize=(11.5, 6.5))
gs  = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[3.2, 1.3, 1.3], hspace=0.06)
ax_tr  = fig.add_subplot(gs[0])
ax_pos = fig.add_subplot(gs[1], sharex=ax_tr)
ax_spd = fig.add_subplot(gs[2], sharex=ax_tr)

for (x0, x1) in segs_drop:
    for ax in (ax_tr, ax_pos, ax_spd):
        ax.axvspan(x0, x1, color=EXCL_STRIPE_COLOR, alpha=EXCL_STRIPE_ALPHA, zorder=0)

for i in range(N_EXAMPLE_CELLS):
    ax_tr.plot(frames, traces_stack[i, :], color=TRACE_COLOR, lw=TRACE_LW, zorder=2)
ax_tr.set_ylabel("Cell ID")
ax_tr.set_yticks(baselines)
ax_tr.set_yticklabels([str(cid) for cid in cell_ids])
ax_tr.tick_params(axis="x", labelbottom=False)
ax_tr.set_title(f"MED-1090 — {EXAMPLE_TRIAL} — example traces + position + speed",
                fontsize=11)

ax_pos.plot(frames, x_pos, color=X_COLOR, lw=1.2, label="x (cm)", zorder=2)
ax_pos.plot(frames, y_pos, color=Y_COLOR, lw=1.2, label="y (cm)", zorder=2)
ax_pos.set_ylabel("Position (cm)")
ax_pos.tick_params(axis="x", labelbottom=False)
ax_pos.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.04, 1.0), fontsize=9)

ax_spd.plot(frames, speed, color=SPEED_COLOR, lw=1.2, label="Speed", zorder=2)
ax_spd.axhline(MIN_SPEED_CM_S, color=WONG_BLACK, lw=1.2, ls="--", zorder=3)
ax_spd.text(1.03, MIN_SPEED_CM_S, f"  {MIN_SPEED_CM_S:.1f} cm/s",
            transform=ax_spd.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=9, color=WONG_BLACK)
ax_spd.set_ylabel("Speed (cm/s)")
ax_spd.set_xlabel("Frame")
ax_spd.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.04, 1.0), fontsize=9)

for ax in (ax_tr, ax_pos, ax_spd):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

if SAVE_OUTPUTS:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = EXAMPLE_TRIAL.replace(" ", "_").lower()
    fig.savefig(OUT_DIR / f"example_traces_{fname}.svg", bbox_inches="tight")
    print(f"Saved -> example_traces_{fname}.svg")

plt.show()
print("\nCell 5 done.")
print(f"Cells shown: {cell_ids}")
print("Tune: EXAMPLE_TRIAL, N_EXAMPLE_CELLS, TRACE_SPACING, FRAME_WINDOW")


# %%
# =============================================================================
# CELL 6 — BEHAVIORAL CORNER INTERACTION ANALYSIS
#
# Defines four interaction zones (circles, r = CORNER_RADIUS_CM) centred at
# the arena corners (NW, NE, SE, SW).  For each condition (T1-post, T2):
#   1. Compute time spent in each corner zone  (seconds)
#   2. Count discrete interaction bouts  (enter/exit events)
#
# Plots:
#   A) Arena overview  — trajectory + 4 circles overlaid  (T1-post and T2)
#   B) Grouped bar plot — interaction time per corner (T1-post vs T2)
#   C) Social preference index — (social − object) / (social + object)
#      plotted for T1-post and T2 side by side
#
# Corner–stimulus mapping (fixed across experiments):
#   T1-post : social = NE,  object = SW
#   T2      : social = SW,  object = NE
#
# Requires (from Cell 1):
#   beh_t1, beh_t2, W_CM, H_CM, DOOR_LIFT_FRAME_T1
#   CAMERA_FPS, WONG_* color constants
# =============================================================================

# --- cache load --------------------------------------------------------------
if "beh_t1" not in dir():
    if (OUT_DIR / "beh_t1.csv").exists():
        beh_t1 = pd.read_csv(OUT_DIR / "beh_t1.csv")
        beh_t2 = pd.read_csv(OUT_DIR / "beh_t2.csv")
        print("Loaded beh_t1/t2 from file cache.")
    else:
        raise RuntimeError("beh_t1/t2 not found — run Cell 1 first or enable SAVE_OUTPUTS.")
else:
    print("Using beh_t1/t2 from memory.")

if "W_CM" not in dir():
    if not CALIB_FILE.exists():
        raise FileNotFoundError(f"Calibration file not found: {CALIB_FILE}")
    _, W_CM, H_CM = load_affine_calibration(CALIB_FILE)
    XLIM_CM = (0.0, W_CM)
    YLIM_CM = (0.0, H_CM)
    print(f"Loaded arena dims from calibration: {W_CM:.1f} x {H_CM:.1f} cm")
else:
    print(f"Using W_CM={W_CM:.1f}, H_CM={H_CM:.1f} from memory.")

# --- parameters --------------------------------------------------------------
CORNER_RADIUS_CM = 15.0   # radius of interaction zone circle

# corner centres  {label: (x_cm, y_cm)}
# NE = top-right (W,0), SW = bottom-left (0,H)
# Note: y increases downward in the arena coordinate system
CORNERS = {
    "NE": (W_CM, 0.0),
    "SW": (0.0,  H_CM),
}

# stimulus assignment per condition
SOCIAL_CORNER = {"T1 post": "NE", "T2": "SW"}
OBJECT_CORNER = {"T1 post": "SW", "T2": "NE"}

# minimum bout duration to count as an interaction (frames)
MIN_BOUT_FRAMES = 3

# bar plot style
BAR_WIDTH  = 0.32
CORNER_COLORS = {
    "NE": WONG_VERMILLION,
    "SW": WONG_ORANGE,
}


# ---------- helpers ----------------------------------------------------------

def in_circle(x: np.ndarray, y: np.ndarray,
              cx: float, cy: float, r: float) -> np.ndarray:
    return (x - cx)**2 + (y - cy)**2 <= r**2


def compute_corner_stats(beh_df: pd.DataFrame,
                          frame_mask: np.ndarray) -> dict:
    """
    For each corner return:
        time_s      — total dwell time while ok_raw & in zone
        n_bouts     — number of discrete entry bouts (>= MIN_BOUT_FRAMES)
        bout_times  — list of per-bout durations in seconds
    """
    sub = beh_df[frame_mask].copy()
    x   = sub["x_sg"].to_numpy(dtype=float)
    y   = sub["y_sg"].to_numpy(dtype=float)
    ok  = sub["ok_raw"].to_numpy(dtype=bool)

    results = {}
    for label, (cx, cy) in CORNERS.items():
        inside = ok & np.isfinite(x) & np.isfinite(y) & \
                 in_circle(x, y, cx, cy, CORNER_RADIUS_CM)

        # dwell time
        dt_per_frame = 1.0 / CAMERA_FPS
        time_s = float(inside.sum() * dt_per_frame)

        # bout counting — find contiguous True runs >= MIN_BOUT_FRAMES
        padded = np.r_[False, inside, False]
        starts = np.where(~padded[:-1] &  padded[1:])[0]
        ends   = np.where( padded[:-1] & ~padded[1:])[0]
        lens   = ends - starts
        valid  = lens >= MIN_BOUT_FRAMES
        bout_times = (lens[valid] * dt_per_frame).tolist()

        results[label] = {
            "time_s":     time_s,
            "n_bouts":    int(valid.sum()),
            "bout_times": bout_times,
        }
    return results


# ---------- compute ----------------------------------------------------------

# frame masks  (use ok_raw only — speed filter not applied for proximity measure)
mask_t1post_frames = beh_t1["frame_idx"].to_numpy() >= DOOR_LIFT_FRAME_T1
mask_t2_frames     = np.ones(len(beh_t2), dtype=bool)

stats = {
    "T1 post": compute_corner_stats(beh_t1, mask_t1post_frames),
    "T2":      compute_corner_stats(beh_t2, mask_t2_frames),
}

print("Corner interaction summary")
print(f"{'':8s} {'NE':>8s} {'SW':>8s}")
for cond, cs in stats.items():
    times  = [f"{cs[c]['time_s']:.1f}s" for c in ["NE","SW"]]
    bouts  = [f"({cs[c]['n_bouts']})" for c in ["NE","SW"]]
    print(f"  {cond:8s} " +
          "  ".join(f"{t:>6s}{b:>5s}" for t, b in zip(times, bouts)))

# social preference index
spi = {}
for cond in ["T1 post", "T2"]:
    sc = SOCIAL_CORNER[cond]
    oc = OBJECT_CORNER[cond]
    social = stats[cond][sc]["time_s"]
    obj    = stats[cond][oc]["time_s"]
    denom  = social + obj
    spi[cond] = float((social - obj) / denom) if denom > 0 else np.nan

print(f"\nSocial Preference Index  (social−object)/(social+object):")
for cond, idx in spi.items():
    print(f"  {cond}: {idx:+.3f}")


# =============================================================================
# PLOT A — Arena overview with corner circles
# =============================================================================

def _draw_arena_with_circles(ax, beh_df, frame_mask, title):
    sub    = beh_df[frame_mask]
    ok     = sub["ok_raw"].to_numpy()
    x_traj = sub.loc[sub["ok_raw"], "x_sg"].to_numpy()
    y_traj = sub.loc[sub["ok_raw"], "y_sg"].to_numpy()

    # trajectory
    ax.plot(x_traj, y_traj, lw=0.5, alpha=0.3, color=WONG_SKY_BLUE, zorder=1)

    # corner circles
    theta = np.linspace(0, 2 * np.pi, 200)
    for label, (cx, cy) in CORNERS.items():
        patch = plt.Circle((cx, cy), CORNER_RADIUS_CM,
                            color=CORNER_COLORS[label], alpha=0.20,
                            zorder=2, linewidth=0)
        ax.add_patch(patch)
        edge = plt.Circle((cx, cy), CORNER_RADIUS_CM,
                           color=CORNER_COLORS[label], alpha=0.7,
                           fill=False, linewidth=1.5, zorder=3)
        ax.add_patch(edge)
        ax.text(cx + (3 if cx > 0 else -3),
                cy + (3 if cy > 0 else -3),
                label, fontsize=9, fontweight="bold",
                color=CORNER_COLORS[label], zorder=4,
                ha="left" if cx == 0 else "right",
                va="top"  if cy == 0 else "bottom")

    # arena box
    rect = plt.Rectangle((0, 0), W_CM, H_CM,
                          lw=1.5, edgecolor="black", facecolor="none", zorder=5)
    ax.add_patch(rect)
    ax.set_xlim(-5, W_CM + 5)
    ax.set_ylim(H_CM + 5, -5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
    ax.set_title(title)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


arena_ratio = W_CM / H_CM
cell_w = 5.5
cell_h = cell_w / arena_ratio

fig_A, axes_A = plt.subplots(1, 2,
                              figsize=(cell_w * 2 + 1.5, cell_h),
                              constrained_layout=True)
_draw_arena_with_circles(axes_A[0], beh_t1, mask_t1post_frames,
                          f"T1 post (frame ≥ {DOOR_LIFT_FRAME_T1})")
_draw_arena_with_circles(axes_A[1], beh_t2, mask_t2_frames, "T2")
fig_A.suptitle("Corner interaction zones", fontsize=12, fontweight="bold")

if SAVE_OUTPUTS:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_A.savefig(OUT_DIR / "corner_arena_overview.svg")
    print("Saved -> corner_arena_overview.svg")

plt.show()


# =============================================================================
# PLOT B — Grouped bar: interaction time per corner (T1-post vs T2)
# =============================================================================

corner_order = ["NE", "SW"]
cond_list    = ["T1 post", "T2"]
cond_colors  = {
    "T1 post": WONG_BLUE,
    "T2":      WONG_VERMILLION,
}

x_pos  = np.arange(len(corner_order))
fig_B, axes_B = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

# left panel — interaction time
ax = axes_B[0]
for i, cond in enumerate(cond_list):
    vals   = [stats[cond][c]["time_s"] for c in corner_order]
    offset = (i - 0.5) * BAR_WIDTH
    bars   = ax.bar(x_pos + offset, vals, width=BAR_WIDTH,
                    color=cond_colors[cond], label=cond,
                    edgecolor="white", linewidth=0.8, zorder=3)

ax.set_xticks(x_pos)
ax.set_xticklabels(corner_order, fontsize=11)
ax.set_ylabel("Interaction time (s)")
ax.set_title("Time in corner zones")
ax.legend(frameon=False, fontsize=9)
ax.grid(axis="y", lw=0.5, alpha=0.4, zorder=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# right panel — bout count
ax2 = axes_B[1]
for i, cond in enumerate(cond_list):
    vals   = [stats[cond][c]["n_bouts"] for c in corner_order]
    offset = (i - 0.5) * BAR_WIDTH
    ax2.bar(x_pos + offset, vals, width=BAR_WIDTH,
            color=cond_colors[cond], label=cond,
            edgecolor="white", linewidth=0.8, zorder=3)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(corner_order, fontsize=11)
ax2.set_ylabel("Interaction bouts (n)")
ax2.set_title(f"Interaction bouts (≥{MIN_BOUT_FRAMES} frames)")
ax2.legend(frameon=False, fontsize=9)
ax2.grid(axis="y", lw=0.5, alpha=0.4, zorder=0)
for sp in ("top", "right"):
    ax2.spines[sp].set_visible(False)

fig_B.suptitle("Corner interactions — T1 post vs T2", fontsize=12, fontweight="bold")

if SAVE_OUTPUTS:
    fig_B.savefig(OUT_DIR / "corner_interaction_bars.svg")
    print("Saved -> corner_interaction_bars.svg")

plt.show()


# =============================================================================
# PLOT C — Social preference index
# =============================================================================

fig_C, ax_C = plt.subplots(figsize=(4.5, 4.5), constrained_layout=True)

cond_labels = list(spi.keys())
spi_vals    = [spi[c] for c in cond_labels]
bar_colors  = [cond_colors[c] for c in cond_labels]
x_spi       = np.arange(len(cond_labels))

bars = ax_C.bar(x_spi, spi_vals, width=0.45,
                color=bar_colors, edgecolor="white", linewidth=0.8, zorder=3)

ax_C.axhline(0, color=WONG_BLACK, lw=1.2, ls="--", zorder=4)
ax_C.set_xticks(x_spi)
ax_C.set_xticklabels(cond_labels, fontsize=11)
ax_C.set_ylabel("Social preference index\n(social − object) / (social + object)")
ax_C.set_ylim(-1.05, 1.05)
ax_C.set_title("Social preference index")
ax_C.grid(axis="y", lw=0.5, alpha=0.4, zorder=0)

# annotate corner assignment
for i, cond in enumerate(cond_labels):
    sc = SOCIAL_CORNER[cond]
    oc = OBJECT_CORNER[cond]
    ax_C.text(i, -0.98,
              f"social={sc}\nobject={oc}",
              ha="center", va="bottom", fontsize=8,
              color="0.45")

for sp in ("top", "right"):
    ax_C.spines[sp].set_visible(False)

if SAVE_OUTPUTS:
    fig_C.savefig(OUT_DIR / "social_preference_index.svg")
    print("Saved -> social_preference_index.svg")

plt.show()

print("\nCell 6 done.")
print("Available: stats, spi")
print("  stats[cond][corner] -> time_s, n_bouts, bout_times")
print("  spi[cond]           -> social preference index")
print(f"\nCorner assignment:")
for cond in ["T1 post", "T2"]:
    print(f"  {cond}: social={SOCIAL_CORNER[cond]}  object={OBJECT_CORNER[cond]}")


# %%
# =============================================================================
# CELL 6.5 — VIDEO FRAME + SLEAP SKELETON OVERLAY
#
# Picks a random frame from the T1 post-door-lift epoch, reads it from the
# video with cv2, then overlays ALL SLEAP bodypart keypoints as large dots.
# Only keypoints with a valid score (>= MIN_KP_SCORE) are shown.
#
# Requires:
#   SLEAP_T1, VIDEO_T1, DOOR_LIFT_FRAME_T1  — from Cell 0 / draft script
#   sniff_read_csv or pandas                — standard
# =============================================================================

import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# --- parameters --------------------------------------------------------------
FRAME_SEED    = 42        # set None for a different random frame each run
MIN_KP_SCORE  = 0.1       # minimum keypoint score to display
DOT_RADIUS    = 5        # scatter dot size (points^2 in matplotlib scatter)
DOT_COLOR     = "#FFD700" # gold — highly visible on dark video backgrounds
DOT_EDGE      = "white"
DOT_EDGE_LW   = .5
LABEL_KP      = False      # annotate each dot with its bodypart name

# --- paths (re-use from Cell 0 if already defined) ---------------------------
if "VIDEO_T1" not in dir():
    VIDEO_T1 = BASE / "MED-1090_SR_T1.mp4"
if "SLEAP_T1" not in dir():
    raise RuntimeError("SLEAP_T1 not defined — run Cell 0 first.")
if "DOOR_LIFT_FRAME_T1" not in dir():
    DOOR_LIFT_FRAME_T1 = 15067


# ---------- helpers ----------------------------------------------------------

def _read_frame_cv2(video_path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _load_sleap_frame(sleap_path, frame_idx: int) -> pd.DataFrame:
    """Return all rows for a specific frame_idx from the SLEAP CSV."""
    sep = ","
    for s in [",", "\t", ";"]:
        try:
            tmp = pd.read_csv(sleap_path, sep=s, nrows=5)
            if tmp.shape[1] > 1:
                sep = s
                break
        except Exception:
            pass
    df = pd.read_csv(sleap_path, sep=sep)
    df.columns = df.columns.str.strip()
    return df[df["frame_idx"] == frame_idx].copy()


def _parse_keypoints(frame_rows: pd.DataFrame) -> list[dict]:
    """
    Extract all (name, x, y, score) keypoints from a SLEAP frame slice.
    Handles the column naming convention: <part>.x / <part>.y / <part>.score
    """
    kps = []
    # find all .x columns (excluding instance.score)
    x_cols = [c for c in frame_rows.columns
               if c.endswith(".x") and not c.startswith("instance")]
    for xc in x_cols:
        part  = xc[:-2]           # strip trailing ".x"
        yc    = part + ".y"
        sc    = part + ".score"
        if yc not in frame_rows.columns:
            continue
        score_val = float(frame_rows[sc].iloc[0]) \
                    if sc in frame_rows.columns else 1.0
        x_val = float(frame_rows[xc].iloc[0])
        y_val = float(frame_rows[yc].iloc[0])
        if np.isfinite(x_val) and np.isfinite(y_val):
            kps.append({"name": part, "x": x_val, "y": y_val, "score": score_val})
    return kps


# ---------- pick a random T1-post frame that has SLEAP data ------------------

sleap_df = pd.read_csv(SLEAP_T1)
sleap_df.columns = sleap_df.columns.str.strip()

post_frames = sleap_df[sleap_df["frame_idx"] >= DOOR_LIFT_FRAME_T1]["frame_idx"].unique()
if len(post_frames) == 0:
    raise RuntimeError("No SLEAP frames found after DOOR_LIFT_FRAME_T1.")

rng = random.Random(FRAME_SEED)
chosen_frame = int(rng.choice(post_frames))
print(f"Chosen frame: {chosen_frame}  (T1 post, seed={FRAME_SEED})")

# ---------- load video frame + keypoints -------------------------------------

frame_rgb  = _read_frame_cv2(VIDEO_T1, chosen_frame)
frame_rows = _load_sleap_frame(SLEAP_T1, chosen_frame)
keypoints  = _parse_keypoints(frame_rows)

valid_kps  = [k for k in keypoints if k["score"] >= MIN_KP_SCORE]
print(f"Keypoints found: {len(keypoints)}  |  valid (score >= {MIN_KP_SCORE}): {len(valid_kps)}")
for k in valid_kps:
    print(f"  {k['name']:20s}  x={k['x']:.1f}  y={k['y']:.1f}  score={k['score']:.3f}")

# ---------- plot -------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
ax.imshow(frame_rgb)

if valid_kps:
    xs     = [k["x"]    for k in valid_kps]
    ys     = [k["y"]    for k in valid_kps]
    names  = [k["name"] for k in valid_kps]

    ax.scatter(xs, ys,
               s=DOT_RADIUS**2,
               c=DOT_COLOR,
               edgecolors=DOT_EDGE,
               linewidths=DOT_EDGE_LW,
               zorder=5)

    if LABEL_KP:
        for x, y, name in zip(xs, ys, names):
            ax.text(x + 8, y - 8, name,
                    fontsize=7, color=DOT_COLOR,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1",
                              fc="black", ec="none", alpha=0.45),
                    zorder=6)

ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_title(f"T1 post — frame {chosen_frame} — SLEAP keypoints",
             fontsize=11)

legend_patch = mpatches.Patch(color=DOT_COLOR,
                               label=f"keypoints (score ≥ {MIN_KP_SCORE})")
ax.legend(handles=[legend_patch], loc="upper left",
          framealpha=0.6, fontsize=8)

if SAVE_OUTPUTS:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"sleap_overlay_T1post_frame{chosen_frame}.svg",
                bbox_inches="tight")
    print(f"Saved -> sleap_overlay_T1post_frame{chosen_frame}.svg")

plt.show()
print("\nCell 6.5 done.")
print("Tune: FRAME_SEED, MIN_KP_SCORE, DOT_RADIUS, DOT_COLOR, LABEL_KP")


# %%
# =============================================================================
# CELL 7 — TOP PLACE CELL: TRAJECTORY + RATE MAP  (T1-pre, T1-post, T2)
#
# For each condition plots side by side:
#   Left  — full camera-rate trajectory (grey) + binarized events (orange dots)
#   Right — upsampled spatial firing rate map (viridis, white = unvisited)
#
# Cell to plot is taken from summary_all (top-ranked by spatial info per
# condition), or can be overridden with OVERRIDE_CELLS below.
#
# Requires (from earlier cells):
#   summary_all, t_ca, beh_x, beh_y, ca_mat
#   mask_T1_pre, mask_T1_post, mask_T2
#   beh_t1, beh_t2
#   W_CM, H_CM, XLIM_CM, YLIM_CM
#   DOOR_LIFT_FRAME_T1, CAMERA_FPS
#   binarize_events_zscore, spatial_rate_map, upsample_rate_map  (defined above)
#   Wong color constants
# =============================================================================

# --- cell to track across all three conditions --------------------------------
TRACK_CELL = 283   # set to the cell index you want to follow

# --- display -----------------------------------------------------------------
UPSAMPLE_FACTOR_7 = 2
DISPLAY_SIGMA_7   = 2.0
EVENT_DOT_SIZE    = 18     # scatter marker size (pt^2)
TRAJ_LW           = 0.5
TRAJ_ALPHA        = 0.35

# condition definitions
_conditions = [
    ("T1 pre",  mask_T1_pre,  beh_t1,
     beh_t1["frame_idx"].to_numpy() < DOOR_LIFT_FRAME_T1),
    ("T1 post", mask_T1_post, beh_t1,
     beh_t1["frame_idx"].to_numpy() >= DOOR_LIFT_FRAME_T1),
    ("T2",      mask_T2,      beh_t2,
     np.ones(len(beh_t2), dtype=bool)),
]

arena_ratio = W_CM / H_CM
map_w       = 4.0                   # width of each sub-panel (inches)
map_h       = map_w / arena_ratio
fig_w       = map_w * 2 * 3 + 0.6  # 3 conditions x 2 panels + spacing
fig_h       = map_h + 0.8

cmap = plt.cm.viridis.copy()
cmap.set_bad(color="white")

fig, axes = plt.subplots(
    1, 6,
    figsize=(fig_w, fig_h),
    constrained_layout=True,
    gridspec_kw={"wspace": 0.05}
)

for col, (cond_name, cond_mask, beh_df, fr_mask) in enumerate(_conditions):

    ax_traj = axes[col * 2]
    ax_map  = axes[col * 2 + 1]

    # --- always the same cell, look up its stats for this condition if available --
    ci   = int(TRACK_CELL)
    rows = summary_all[(summary_all["condition"] == cond_name) &
                       (summary_all["cell_idx"]  == ci)]
    if len(rows) > 0:
        pval = float(rows.iloc[0]["p_value"])
        info = float(rows.iloc[0]["spatial_info_bits_event"])
        sig  = bool(rows.iloc[0]["significant"])
    else:
        # cell was not in top-N for this condition — compute stats on the fly
        t_c = t_ca[cond_mask]; x_c = beh_x[cond_mask]; y_c = beh_y[cond_mask]
        ev_tmp = binarize_events_zscore(t_c, ca_mat[cond_mask, ci])
        rm_tmp = spatial_rate_map(x_c, y_c, t_c, ev_tmp, xlim=XLIM_CM, ylim=YLIM_CM)
        info   = spatial_info_bits_per_event(rm_tmp["rate"], rm_tmp["occ_s"])
        pval, _ = circular_permutation_pvalue(x_c, y_c, t_c, ev_tmp)
        sig    = np.isfinite(pval) and pval <= ALPHA
        print(f"  {cond_name}: cell {ci} not in summary — recomputed: "
              f"info={info:.3f}  p={pval:.3f}")

    # --- condition arrays ---
    t_cond = t_ca[cond_mask]
    x_cond = beh_x[cond_mask]
    y_cond = beh_y[cond_mask]
    S_cond = ca_mat[cond_mask, :]

    ev = binarize_events_zscore(t_cond, S_cond[:, ci])
    rm = spatial_rate_map(x_cond, y_cond, t_cond, ev,
                          xlim=XLIM_CM, ylim=YLIM_CM)
    rate_display = upsample_rate_map(rm["rate"], UPSAMPLE_FACTOR_7, DISPLAY_SIGMA_7)

    # --- full camera-rate trajectory for this condition epoch ---
    traj_ok = beh_df["ok_raw"].to_numpy() & fr_mask
    tx = beh_df.loc[traj_ok, "x_sg"].to_numpy()
    ty = beh_df.loc[traj_ok, "y_sg"].to_numpy()

    # ---- trajectory panel ----
    ax_traj.plot(tx, ty, lw=TRAJ_LW, alpha=TRAJ_ALPHA,
                 color=WONG_SKY_BLUE, zorder=1)
    ax_traj.scatter(x_cond[ev], y_cond[ev],
                    s=EVENT_DOT_SIZE, color=WONG_ORANGE,
                    zorder=3, linewidths=0)
    rect = plt.Rectangle((0, 0), W_CM, H_CM,
                          lw=1.2, edgecolor="black", facecolor="none", zorder=4)
    ax_traj.add_patch(rect)
    ax_traj.set_xlim(-2, W_CM + 2)
    ax_traj.set_ylim(H_CM + 2, -2)
    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("x (cm)", fontsize=8)
    ax_traj.set_ylabel("y (cm)", fontsize=8)
    ax_traj.tick_params(labelsize=7)
    ax_traj.set_title(f"{cond_name}  |  cell {ci}",
                      fontsize=9, fontweight="bold")
    for sp in ("top", "right"):
        ax_traj.spines[sp].set_visible(False)

    # ---- rate map panel ----
    im = ax_map.imshow(
        rate_display, origin="upper",
        extent=[XLIM_CM[0], XLIM_CM[1], YLIM_CM[1], YLIM_CM[0]],
        interpolation="bilinear", aspect="equal", cmap=cmap
    )
    ax_map.set_xticks([]); ax_map.set_yticks([])
    for sp in ax_map.spines.values():
        sp.set_visible(False)

    sig_marker = "*" if sig else ""
    ax_map.set_title(
        f"info={info:.2f}  p={pval:.3f}{sig_marker}",
        fontsize=8
    )
    cb = plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.02)
    cb.set_label("events/s", fontsize=7)
    cb.ax.tick_params(labelsize=7)

fig.suptitle(
    f"Cell {TRACK_CELL} — trajectory + rate map across conditions  "
    f"(upsample ×{UPSAMPLE_FACTOR_7}, σ={DISPLAY_SIGMA_7})",
    fontsize=10, fontweight="bold"
)

if SAVE_OUTPUTS:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "top_placecell_traj_ratemap_all_conditions.svg",
                bbox_inches="tight")
    print("Saved -> top_placecell_traj_ratemap_all_conditions.svg")

plt.show()
print("\nCell 7 done.")
print(f"Change TRACK_CELL at the top of this cell to follow a different cell.")
