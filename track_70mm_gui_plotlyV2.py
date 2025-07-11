"""
track_70mm_gui_plotly_live_webgl.py
-----------------------------------
Tk-GUI + OpenCV tracker with:

• Real-time Plotly WebGL preview (opens in browser)
• Color-coded out-of-tolerance alerts
• Velocity / direction readout
• Post-run Plotly HTML report, CSV, processed video
"""

import cv2, csv, os, tempfile, webbrowser, tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ───────────────────────────────── CONSTANTS ────────────────────────────────
SCALE_MM_PER_PX = 0.04904          # 77 mm / 1570 px
TEMPLATE_HALF   = 12               # template half-width (px)
MATCH_METHOD    = cv2.TM_CCOEFF
SMOOTH_WIN      = 7                # smoothing window (frames)
FONT            = cv2.FONT_HERSHEY_SIMPLEX
MARGIN_HEIGHT   = 40               # DRO strip height (px)
ALERT_WARN      = 0.75             # yellow threshold (mm)
ALERT_CRIT      = 1.00             # red    threshold (mm)
REF_X           = 251              # fixed 70 mm tick (px)
CROP_X, CROP_Y, CROP_W, CROP_H = 400, 1, 75, 600   # ruler strip
LIVE_WINDOW     = 300              # frames displayed in live chart
UPDATE_EVERY    = 5                # update live chart every N frames

# ─────────────────────────────── GUI FILE PICKERS ───────────────────────────
root = tk.Tk(); root.withdraw()

video_path = filedialog.askopenfilename(
    title="Select input video",
    filetypes=[("Video files", "*.mp4 *.mov *.avi")])
if not video_path:
    raise SystemExit("No video selected.")

base = simpledialog.askstring("Output",
        "Enter base output file name (no extension):")
if not base:
    raise SystemExit("No output name provided.")

out_dir      = os.path.dirname(video_path)
out_video    = os.path.join(out_dir, base + ".mp4")
csv_path     = os.path.join(out_dir, base + ".csv")
html_path    = os.path.join(out_dir, base + ".html")

# ─────────────────────────── CAPTURE FIRST FRAME & TEMPLATE ────────────────
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Cannot open video.")
fps = cap.get(cv2.CAP_PROP_FPS)

cap.set(cv2.CAP_PROP_POS_MSEC, 1000)       # 1-second mark
ok, frame = cap.read()
if not ok:
    raise RuntimeError("Cannot read frame @1 s")

roi      = frame[CROP_Y:CROP_Y+CROP_H, CROP_X:CROP_X+CROP_W]
ref_rot  = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
rot_h, rot_w = ref_rot.shape[:2]

x1, x2 = max(0, REF_X-TEMPLATE_HALF), min(rot_w-1, REF_X+TEMPLATE_HALF)
template = cv2.cvtColor(ref_rot[:, x1:x2], cv2.COLOR_BGR2GRAY)

# ───────────────────────────── VIDEO WRITER (processed) ─────────────────────
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    out_video, fourcc, fps, (rot_w, rot_h + MARGIN_HEIGHT))

# ────────────────────────────── LIVE PLOTLY PREVIEW ─────────────────────────
fig_live = make_subplots(rows=1, cols=1)
trace = go.Scattergl(
    x=[], y=[], mode="lines", line=dict(color="royalblue"), name="dev (mm)")
fig_live.add_trace(trace)
fig_live.update_layout(
    title="Live deviation (last 300 frames)",
    xaxis_title="Frame", yaxis_title="Deviation (mm)",
    xaxis_range=[0, LIVE_WINDOW], yaxis_range=[-3, 3],
    margin=dict(l=40,r=10,t=40,b=40), height=300)

tmp_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
tmp_html.write(fig_live.to_html(full_html=True, include_plotlyjs="cdn").encode())
tmp_html.close()
webbrowser.open("file://" + tmp_html.name)   # launches browser window

# ──────────────────────────────── TRACK LOOP ───────────────────────────────
frames, raw_mm = [], []
prev_dev = None
idx = 0

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    idx += 1

    # ― detect tick position ―
    roi = frame[CROP_Y:CROP_Y+CROP_H, CROP_X:CROP_X+CROP_W]
    roi_rot = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
    gray    = cv2.cvtColor(roi_rot, cv2.COLOR_BGR2GRAY)
    _,_,_,loc = cv2.minMaxLoc(cv2.matchTemplate(gray, template, MATCH_METHOD))
    best_x  = loc[0] + TEMPLATE_HALF
    dev_mm  = (best_x - REF_X) * SCALE_MM_PER_PX

    # ― velocity (mm/s) ―
    if prev_dev is None:
        vel_mm_s = 0.0
    else:
        vel_mm_s = (dev_mm - prev_dev) * fps
    prev_dev = dev_mm

    frames.append(idx); raw_mm.append(dev_mm)

    # ― draw reference/measurement lines ―
    cv2.line(roi_rot, (REF_X ,0), (REF_X ,rot_h), (0,0,255), 2)
    cv2.line(roi_rot, (best_x,0), (best_x,rot_h), (0,255,0), 1)

    # ― build DRO margin with alert coloring ―
    if abs(dev_mm) >= ALERT_CRIT:
        bg = np.full((MARGIN_HEIGHT, rot_w, 3), (60,60,255), np.uint8)  # red bg
        color = (0,0,255)   # red text
    elif abs(dev_mm) >= ALERT_WARN:
        bg = np.full((MARGIN_HEIGHT, rot_w, 3), (0,255,255), np.uint8)   # yellow
        color = (0,0,0)     # black text
    else:
        bg = np.full((MARGIN_HEIGHT, rot_w, 3), 245, np.uint8)           # gray
        color = (0,0,0)     # black text

    arrow = "↑" if vel_mm_s > 0.01 else ("↓" if vel_mm_s < -0.01 else "→")
    text  = f"{dev_mm:+.2f} mm  {vel_mm_s:+.2f} mm/s {arrow}"
    cv2.putText(bg, text, (10, 28), FONT, 0.8, color, 2, cv2.LINE_AA)

    framed = np.vstack((bg, roi_rot))
    writer.write(framed)

    # ― update live plot every N frames ―
    if idx % UPDATE_EVERY == 0:
        # limit to last LIVE_WINDOW points
        x_window = frames[-LIVE_WINDOW:]
        y_window = raw_mm[-LIVE_WINDOW:]
        with fig_live.batch_update():
    trace.x = x_window
    trace.y = y_window
    fig_live.layout.xaxis.range = [x_window[0], x_window[-1] + 1]

    ymin = np.min(y_window)
    ymax = np.max(y_window)
    padding = 0.25
    fig_live.layout.yaxis.range = [ymin - padding, ymax + padding]

cap.release(); writer.release()

# ───────────────────────────── SMOOTH & SAVE CSV ────────────────────────────
smoothed = pd.Series(raw_mm).rolling(SMOOTH_WIN, center=True).mean()
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["Frame","Dev_mm_raw","Dev_mm_smooth"])
    for fr, r, s in zip(frames, raw_mm, smoothed):
        w.writerow([fr, f"{r:.3f}", f"{s:.3f}" if not pd.isna(s) else ""])

# ──────────────────────────── STATIC PLOT REPORT ───────────────────────────
fig_report = go.Figure()
fig_report.add_trace(go.Scatter(
    x=frames, y=raw_mm, mode="lines", name="Raw",
    line=dict(color="rgba(0,100,255,.3)", dash="dot")))
fig_report.add_trace(go.Scatter(
    x=frames, y=smoothed, mode="lines+markers", name="Smoothed",
    line=dict(color="red")))
fig_report.update_layout(
    title="Deviation of 70 mm Tick (Raw vs. Smoothed)",
    xaxis_title="Frame", yaxis_title="Deviation (mm)",
    plot_bgcolor="rgb(245,245,245)", hovermode="x unified", height=500)
fig_report.write_html(html_path)

# ───────────────────────────────── DONE ────────────────────────────────────
messagebox.showinfo(
    "Complete",
    f"Outputs saved in:\n\n• {out_video}\n• {csv_path}\n• {html_path}")
