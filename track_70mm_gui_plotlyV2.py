"""
track_70mm_gui_plotly_live.py
-----------------------------
Tk-GUI + OpenCV + Matplotlib live preview + Plotly post report
for tracking the 70 mm tick mark.
"""

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
SCALE_MM_PER_PX = 0.04904         # 77 mm / 1570 px
TEMPLATE_HALF   = 12              # template half-width (px)
MATCH_METHOD    = cv2.TM_CCOEFF
SMOOTH_WIN      = 7               # smoothing window (frames)
FONT            = cv2.FONT_HERSHEY_SIMPLEX
MARGIN_HEIGHT   = 40              # DRO strip height (px)
ALERT_WARN      = 0.75            # yellow threshold (mm)
ALERT_CRIT      = 1.00            # red   threshold (mm)
REF_X           = 251             # fixed 70 mm tick position (px)
CROP_X, CROP_Y, CROP_W, CROP_H = 400, 1, 75, 600   # ruler strip

# ── GUI FILE PICKERS ────────────────────────────────────
root = tk.Tk(); root.withdraw()

video_path = filedialog.askopenfilename(
    title="Select input video", filetypes=[("Video","*.mp4 *.mov *.avi")])
if not video_path: raise SystemExit("No video selected.")

base = simpledialog.askstring("Output",
        "Enter base output file name (no extension):")
if not base: raise SystemExit("No output name provided.")

out_dir      = os.path.dirname(video_path)
out_video    = os.path.join(out_dir, base + ".mp4")
csv_path     = os.path.join(out_dir, base + ".csv")
html_path    = os.path.join(out_dir, base + ".html")

# ── CAPTURE FIRST FRAME & TEMPLATE ──────────────────────
cap = cv2.VideoCapture(video_path)
if not cap.isOpened(): raise IOError("Cannot open video.")
fps = cap.get(cv2.CAP_PROP_FPS)

cap.set(cv2.CAP_PROP_POS_MSEC, 1000)       # 1 s mark
ok, frame = cap.read()
if not ok: raise RuntimeError("Cannot read frame @1 s")

roi      = frame[CROP_Y:CROP_Y+CROP_H, CROP_X:CROP_X+CROP_W]
ref_rot  = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
rot_h, rot_w = ref_rot.shape[:2]

x1, x2 = max(0, REF_X-TEMPLATE_HALF), min(rot_w-1, REF_X+TEMPLATE_HALF)
template = cv2.cvtColor(ref_rot[:, x1:x2], cv2.COLOR_BGR2GRAY)

# ── VIDEO WRITER ────────────────────────────────────────
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_video, fourcc, fps, (rot_w, rot_h + MARGIN_HEIGHT))

# ── LIVE PREVIEW WINDOW (Matplotlib inside Tk) ──────────
preview = tk.Toplevel(); preview.title("Live Deviation Preview")
fig, ax = plt.subplots(figsize=(6,3))
line_raw, = ax.plot([], [], lw=1.5, label="raw dev (mm)")
ax.set_xlim(0, 100); ax.set_ylim(-3, 3)
ax.set_xlabel("Frame"); ax.set_ylabel("Deviation (mm)")
ax.legend(); fig.tight_layout()
canvas = FigureCanvasTkAgg(fig, master=preview)
canvas.get_tk_widget().pack(fill="both", expand=True)

# ── TRACK LOOP ──────────────────────────────────────────
frames, raw_mm = [], []
prev_dev = None
idx = 0

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:
    ok, frame = cap.read()
    if not ok: break
    idx += 1

    # ― Locate tick ―
    roi = frame[CROP_Y:CROP_Y+CROP_H, CROP_X:CROP_X+CROP_W]
    roi_rot = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
    gray    = cv2.cvtColor(roi_rot, cv2.COLOR_BGR2GRAY)
    _,_,_,loc = cv2.minMaxLoc(cv2.matchTemplate(gray, template, MATCH_METHOD))
    best_x  = loc[0] + TEMPLATE_HALF
    dev_mm  = (best_x - REF_X) * SCALE_MM_PER_PX

    # ― Velocity (mm/s) ―
    if prev_dev is None:
        vel_mm_s = 0.0
    else:
        vel_mm_s = (dev_mm - prev_dev) * fps
    prev_dev = dev_mm

    frames.append(idx); raw_mm.append(dev_mm)

    # ― Draw reference & measured lines ―
    cv2.line(roi_rot, (REF_X ,0), (REF_X ,rot_h), (0,0,255), 2)
    cv2.line(roi_rot, (best_x,0), (best_x,rot_h), (0,255,0), 1)

    # ― Build DRO margin ―
    # Choose background color by alert level
    if abs(dev_mm) >= ALERT_CRIT:
        color =  (0,0,255)   # red
        bg    = np.full((MARGIN_HEIGHT, rot_w, 3), (60,60,255), np.uint8)
    elif abs(dev_mm) >= ALERT_WARN:
        color =  (0,0,0)     # black on yellow
        bg    = np.full((MARGIN_HEIGHT, rot_w, 3), (0,255,255), np.uint8)
    else:
        color =  (0,0,0)     # black on light-gray
        bg    = np.full((MARGIN_HEIGHT, rot_w, 3), 245, np.uint8)

    # Deviation text, velocity, direction arrow
    arrow = "↑" if vel_mm_s > 0.01 else ("↓" if vel_mm_s < -0.01 else "→")
    text  = f"{dev_mm:+.2f} mm  {vel_mm_s:+.2f} mm/s {arrow}"
    cv2.putText(bg, text, (10, 28), FONT, 0.8, color, 2, cv2.LINE_AA)

    framed = np.vstack((bg, roi_rot))
    writer.write(framed)

    # ― Update live plot every 5 frames for speed ―
    if idx % 5 == 0:
        line_raw.set_data(frames, raw_mm)
        ax.set_xlim(max(0, idx-200), idx+10)
        canvas.draw_idle()
        preview.update()

cap.release(); writer.release()

# ── SMOOTH & SAVE CSV ───────────────────────────────────
smoothed = pd.Series(raw_mm).rolling(SMOOTH_WIN, center=True).mean()
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["Frame","Dev_mm_raw","Dev_mm_smooth"])
    for fr, r, s in zip(frames, raw_mm, smoothed):
        w.writerow([fr, f"{r:.3f}", f"{s:.3f}" if not pd.isna(s) else ""])

# ── STATIC PLOT WITH PLOTLY ─────────────────────────────
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=frames, y=raw_mm, mode="lines", name="Raw",
    line=dict(color="rgba(0,100,255,.3)", dash="dot")))
fig2.add_trace(go.Scatter(
    x=frames, y=smoothed, mode="lines+markers", name="Smoothed",
    line=dict(color="red")))
fig2.update_layout(
    title="Deviation of 70 mm Tick (Raw vs. Smoothed)",
    xaxis_title="Frame", yaxis_title="Deviation (mm)",
    plot_bgcolor="rgb(245,245,245)", hovermode="x unified", height=500)
fig2.write_html(html_path)

# ── FINISH ──────────────────────────────────────────────
preview.destroy()
messagebox.showinfo(
    "Complete",
    f"Outputs saved in:\n\n• {out_video}\n• {csv_path}\n• {html_path}")
