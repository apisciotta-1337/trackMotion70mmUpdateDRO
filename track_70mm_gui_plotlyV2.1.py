import cv2, csv, os, tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── CONSTANTS ────────────────────────────────────────────
SCALE_MM_PER_PX = 0.04904
TEMPLATE_HALF   = 12
MATCH_METHOD    = cv2.TM_CCOEFF
SMOOTH_WIN      = 7
FONT            = cv2.FONT_HERSHEY_SIMPLEX
MARGIN_HEIGHT   = 40
ALERT_WARN      = 0.75
ALERT_CRIT      = 1.00
REF_X           = 251
CROP_X, CROP_Y, CROP_W, CROP_H = 400, 1, 75, 600
LIVE_WINDOW     = 300
UPDATE_EVERY    = 5

# ── GUI FILE PICKERS ────────────────────────────────────
root = tk.Tk(); root.withdraw()

video_path = filedialog.askopenfilename(
    title="Select input video",
    filetypes=[("Video files", "*.mp4 *.mov *.avi")])
if not video_path:
    raise SystemExit("No video selected.")

base = simpledialog.askstring("Output", "Enter base output file name (no extension):")
if not base:
    raise SystemExit("No output name provided.")

out_dir      = os.path.dirname(video_path)
out_video    = os.path.join(out_dir, base + ".mp4")
csv_path     = os.path.join(out_dir, base + ".csv")
html_path    = os.path.join(out_dir, base + ".html")

# ── CAPTURE FIRST FRAME & TEMPLATE ──────────────────────
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Cannot open video.")
fps = cap.get(cv2.CAP_PROP_FPS)

cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
ok, frame = cap.read()
if not ok:
    raise RuntimeError("Cannot read frame @1 s")

roi      = frame[CROP_Y:CROP_Y+CROP_H, CROP_X:CROP_X+CROP_W]
ref_rot  = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
rot_h, rot_w = ref_rot.shape[:2]

x1, x2 = max(0, REF_X - TEMPLATE_HALF), min(rot_w - 1, REF_X + TEMPLATE_HALF)
template = cv2.cvtColor(ref_rot[:, x1:x2], cv2.COLOR_BGR2GRAY)

# ── VIDEO WRITER ────────────────────────────────────────
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_video, fourcc, fps, (rot_w, rot_h + MARGIN_HEIGHT))

# ── MATPLOTLIB LIVE PLOT ────────────────────────────────
preview = tk.Toplevel()
preview.title("Live Deviation Preview")

fig, ax = plt.subplots(figsize=(6, 3))
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, LIVE_WINDOW)
ax.set_ylim(-3, 3)
ax.set_xlabel("Frame")
ax.set_ylabel("Deviation (mm)")
fig.tight_layout()

canvas = FigureCanvasTkAgg(fig, master=preview)
canvas.get_tk_widget().pack(fill="both", expand=True)

# ── TRACK LOOP ──────────────────────────────────────────
frames, raw_mm = [], []
prev_dev = None
idx = 0

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    idx += 1
    timestamp = idx / fps

    roi = frame[CROP_Y:CROP_Y+CROP_H, CROP_X:CROP_X+CROP_W]
    roi_rot = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(roi_rot, cv2.COLOR_BGR2GRAY)
    _, _, _, loc = cv2.minMaxLoc(cv2.matchTemplate(gray, template, MATCH_METHOD))
    best_x = loc[0] + TEMPLATE_HALF
    dev_mm = (best_x - REF_X) * SCALE_MM_PER_PX

    if prev_dev is None:
        vel_mm_s = 0.0
    else:
        vel_mm_s = (dev_mm - prev_dev) * fps
    prev_dev = dev_mm

    frames.append(idx)
    raw_mm.append(dev_mm)

    cv2.line(roi_rot, (REF_X, 0), (REF_X, rot_h), (0, 0, 255), 2)
    cv2.line(roi_rot, (best_x, 0), (best_x, rot_h), (0, 255, 0), 1)

    if abs(dev_mm) >= ALERT_CRIT:
        bg = np.full((MARGIN_HEIGHT, rot_w, 3), (60, 60, 255), np.uint8)
    elif abs(dev_mm) >= ALERT_WARN:
        bg = np.full((MARGIN_HEIGHT, rot_w, 3), (0, 255, 255), np.uint8)
    else:
        bg = np.full((MARGIN_HEIGHT, rot_w, 3), 245, np.uint8)

    color = (255, 255, 255)
    arrow = "↑" if vel_mm_s > 0.01 else ("↓" if vel_mm_s < -0.01 else "→")
    text = f"Frame: {idx}  Time: {timestamp:6.2f}s  Δ: {dev_mm:+.2f} mm  {vel_mm_s:+.2f} mm/s {arrow}"
    cv2.putText(bg, text, (10, 28), FONT, 0.65, color, 2, cv2.LINE_AA)

    framed = np.vstack((bg, roi_rot))
    writer.write(framed)

    if idx % UPDATE_EVERY == 0:
        window_x = frames[-LIVE_WINDOW:]
        window_y = raw_mm[-LIVE_WINDOW:]

        line.set_data(window_x, window_y)
        ax.set_xlim(max(0, window_x[0]), window_x[-1] + 1)
        ypad = 0.25
        ymin, ymax = np.min(window_y), np.max(window_y)
        yspan = max(ymax - ymin, 1.0)
        ycenter = (ymax + ymin) / 2
        ax.set_ylim(ycenter - yspan/2 - ypad, ycenter + yspan/2 + ypad)
        canvas.draw()
        preview.update()

cap.release()
writer.release()

# ── SMOOTH & SAVE CSV ───────────────────────────────────
smoothed = pd.Series(raw_mm).rolling(SMOOTH_WIN, center=True).mean()
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Frame", "Dev_mm_raw", "Dev_mm_smooth"])
    for fr, r, s in zip(frames, raw_mm, smoothed):
        w.writerow([fr, f"{r:.3f}", f"{s:.3f}" if not pd.isna(s) else ""])

# ── STATIC PLOT ─────────────────────────────────────────
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=frames, y=raw_mm, mode="lines", name="Raw",
    line=dict(color="rgba(0,100,255,.3)", dash="dot")
))
fig2.add_trace(go.Scatter(
    x=frames, y=smoothed, mode="lines+markers", name="Smoothed",
    line=dict(color="red")
))
fig2.update_layout(
    title="Deviation of 70 mm Tick (Raw vs. Smoothed)",
    xaxis_title="Frame",
    yaxis_title="Deviation (mm)",
    plot_bgcolor="rgb(245,245,245)",
    hovermode="x unified",
    height=500
)
fig2.write_html(html_path)

# ── DONE ────────────────────────────────────────────────
preview.destroy()
messagebox.showinfo(
    "Complete",
    f"Outputs saved in:\n\n• {out_video}\n• {csv_path}\n• {html_path}")
