# %%
import cv2
import numpy as np
from pathlib import Path

# ---------------- USER CONFIG ----------------
BASE = Path(r"G:\uv-projects\star-protocols-xclock\data\src")  # <-- change
VIDEO_T1 = str(BASE / "MED-1090_SR_T1.mp4")  # <-- change if filename differs

# Your chamber dimensions (cm)
W_CM = 61.2
H_CM = 40.0

# Choose ONE of these frame selection modes:
FRAME_MODE = "fixed"      # "fixed" or "random_mid"
FIXED_FRAME_IDX = 32202   # used if FRAME_MODE == "fixed"
RANDOM_SEED = 42          # used if FRAME_MODE == "random_mid"

# Output file (so you can reuse calibration without re-clicking)
CALIB_OUT = BASE / "px_to_cm_affine_calib_trial1.npz"
# --------------------------------------------

def pick_frame_idx(video_path: str, mode: str, fixed_idx: int, seed: int) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if n <= 0:
        raise RuntimeError("Video reports 0 frames (check codec/path).")

    if mode == "fixed":
        return int(np.clip(fixed_idx, 0, n - 1))

    # random_mid
    rng = np.random.default_rng(seed)
    lo = max(0, n // 4)
    hi = max(lo, 3 * n // 4)
    return int(rng.integers(lo, hi + 1))

def read_frame_bgr(video_path: str, frame_idx: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame  # BGR

def cv2_collect_clicks(frame_bgr, window_name, n_clicks=4):
    clicks = []
    disp = frame_bgr.copy()

    help_lines = [
        "Click 4 chamber corners in this ORDER:",
        "  #1 TOP-LEFT (TL)",
        "  #2 TOP-RIGHT (TR)",
        "  #3 BOTTOM-RIGHT (BR)",
        "  #4 BOTTOM-LEFT (BL)",
        "Keys: [r]=reset  [q]=abort",
    ]

    def redraw():
        nonlocal disp
        disp = frame_bgr.copy()

        y0 = 25
        for i, line in enumerate(help_lines):
            cv2.putText(
                disp, line, (15, y0 + 22 * i),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
            )

        for k, (x, y) in enumerate(clicks, start=1):
            cv2.drawMarker(
                disp, (int(x), int(y)), (0, 255, 255),
                markerType=cv2.MARKER_TILTED_CROSS, markerSize=18, thickness=2
            )
            cv2.putText(
                disp, f"#{k}", (int(x) + 10, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
            )

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((float(x), float(y)))
            redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()

    while True:
        cv2.imshow(window_name, disp)
        key = cv2.waitKey(20) & 0xFF

        if len(clicks) >= n_clicks:
            break
        if key == ord('r'):
            clicks = []
            redraw()
        if key == ord('q'):
            cv2.destroyWindow(window_name)
            raise RuntimeError("Calibration aborted (q pressed).")

    cv2.destroyWindow(window_name)
    return np.array(clicks, dtype=float)

def build_affine_px_to_cm(TL, TR, BL, w_cm, h_cm):
    """
    Define skew-aware affine mapping using a parallelogram basis.
    u = TR - TL  (x-axis in px)
    v = BL - TL  (y-axis in px)

    Solve: p - TL = a*u + b*v
    Then: x_cm = a * w_cm, y_cm = b * h_cm
    """
    TL = TL.astype(float)
    TR = TR.astype(float)
    BL = BL.astype(float)

    u = TR - TL
    v = BL - TL

    A = np.column_stack([u, v])  # 2x2
    det = float(np.linalg.det(A))
    if abs(det) < 1e-9:
        raise RuntimeError("Degenerate corner configuration (u and v nearly colinear). Re-click corners.")
    Ainv = np.linalg.inv(A)

    def px_to_cm(x_px, y_px):
        P = np.column_stack([x_px, y_px]).astype(float)
        d = P - TL.reshape(1, 2)
        ab = (Ainv @ d.T).T  # Nx2
        a = ab[:, 0]
        b = ab[:, 1]
        return a * w_cm, b * h_cm

    return px_to_cm, u, v, A, Ainv, det

def main():
    frame_idx = pick_frame_idx(VIDEO_T1, FRAME_MODE, FIXED_FRAME_IDX, RANDOM_SEED)
    frame = read_frame_bgr(VIDEO_T1, frame_idx)

    print(f"Calibration video: {VIDEO_T1}")
    print(f"Calibration frame idx: {frame_idx}")
    print(f"Chamber size (cm): W_CM={W_CM}, H_CM={H_CM}")
    print("OpenCV window will appear; click TL, TR, BR, BL in that order.")

    pts = cv2_collect_clicks(frame, "Calibration: TL,TR,BR,BL", n_clicks=4)
    TL, TR, BR, BL = pts

    px_to_cm, u, v, A, Ainv, det = build_affine_px_to_cm(TL, TR, BL, W_CM, H_CM)

    print(f"Affine basis norms: |u|={np.linalg.norm(u):.2f}px, |v|={np.linalg.norm(v):.2f}px, det={det:.2f}")

    # Quick sanity check: corners should map near (0,0), (W,0), (W,H), (0,H)
    def chk(pt, name):
        x_cm, y_cm = px_to_cm(np.array([pt[0]]), np.array([pt[1]]))
        print(f"{name} -> ({x_cm[0]:.2f} cm, {y_cm[0]:.2f} cm)")

    chk(TL, "TL")
    chk(TR, "TR")
    chk(BR, "BR (not used for basis)")
    chk(BL, "BL")

    # Save calibration so main pipeline can reuse without clicking again
    np.savez(
        CALIB_OUT,
        video_path=VIDEO_T1,
        frame_idx=frame_idx,
        W_CM=W_CM, H_CM=H_CM,
        TL=TL, TR=TR, BR=BR, BL=BL,
        A=A, Ainv=Ainv
    )
    print(f"Saved calibration to: {CALIB_OUT}")

if __name__ == "__main__":
    main()