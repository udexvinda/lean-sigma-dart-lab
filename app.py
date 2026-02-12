import math
import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Lean Sigma Dart Lab", layout="wide")

# -----------------------------
# Dartboard scoring (simplified but realistic)
# -----------------------------
SECTOR_ORDER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

def angle_to_sector(theta_deg: float) -> int:
    """theta_deg: 0 at +x axis, CCW. Dartboard sectors start at 20 centered at 90 degrees (top)."""
    # Rotate so 90° (top) is sector 20 center
    # Each sector is 18 degrees
    # We'll map theta to [0,360)
    theta = theta_deg % 360.0
    # shift so 90 degrees corresponds to index 0 center
    # sector boundaries at +/-9 deg around centers
    shifted = (90.0 - theta) % 360.0
    idx = int((shifted + 9.0) // 18.0) % 20
    return SECTOR_ORDER[idx]

def score_dart(x: float, y: float) -> int:
    """
    x,y in normalized board coordinates where radius 1.0 is board edge.
    Approx dartboard rings (very simplified):
      - Bull: r <= 0.05 => 50
      - Outer bull: r <= 0.10 => 25
      - Triple ring: 0.55 <= r <= 0.60
      - Double ring: 0.95 <= r <= 1.00
      - Single otherwise within r<=1
      - Miss if r>1
    """
    r = math.sqrt(x*x + y*y)
    if r > 1.0:
        return 0
    if r <= 0.05:
        return 50
    if r <= 0.10:
        return 25

    theta = math.degrees(math.atan2(y, x))
    base = angle_to_sector(theta)

    if 0.55 <= r <= 0.60:
        return 3 * base
    if 0.95 <= r <= 1.00:
        return 2 * base
    return base

def draw_board(hit=None, title="Dartboard"):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")

    # board outline
    board = plt.Circle((0,0), 1.0, fill=False, linewidth=2)
    ax.add_patch(board)

    # rings (visual guides)
    for rr in [0.10, 0.60, 0.55, 1.00, 0.95]:
        ax.add_patch(plt.Circle((0,0), rr, fill=False, linestyle="--", linewidth=1))

    # bull / inner bull
    ax.add_patch(plt.Circle((0,0), 0.10, fill=False, linewidth=2))
    ax.add_patch(plt.Circle((0,0), 0.05, fill=False, linewidth=2))

    # sector lines
    for i in range(20):
        ang = math.radians(90 - i*18)
        ax.plot([0, math.cos(ang)], [0, math.sin(ang)], linewidth=0.7)

    if hit is not None:
        x, y = hit
        ax.scatter([x], [y], s=140, marker="X")
        ax.text(x, y, " HIT", fontsize=10, va="bottom")

    ax.set_title(title)
    return fig

# -----------------------------
# Game state
# -----------------------------
if "mode" not in st.session_state:
    st.session_state.mode = "calibrating"  # calibrating | flying | landed
    st.session_state.t = 0
    st.session_state.freeze = None  # (hx, vy, strength)
    st.session_state.hit = None
    st.session_state.last_score = None
    st.session_state.data = []  # collected throws

# Auto-refresh for animation (only when calibrating or flying)
if st.session_state.mode in ("calibrating", "flying"):
    st_autorefresh(interval=100, key="tick")  # 100ms ~ 10 FPS
    st.session_state.t += 1

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1,1])

with left:
    st.subheader("Calibration Deck")

    # Oscillators: values between -1..1, strength 0..1
    t = st.session_state.t

    # Horizontal dot (left-right)
    hx = math.sin(t / 7.0)  # -1..1
    # Vertical dot (up-down)
    vy = math.sin(t / 9.0 + 1.7)  # -1..1
    # Strength dot (gentle-strong)
    strength = (math.sin(t / 11.0 + 0.8) + 1) / 2  # 0..1

    if st.session_state.mode == "calibrating":
        live_hx, live_vy, live_s = hx, vy, strength
    else:
        live_hx, live_vy, live_s = st.session_state.freeze

    st.write("**Horizontal (Left ↔ Right)**")
    st.progress(int((live_hx + 1) / 2 * 100))
    st.caption(f"x = {live_hx:+.3f}")

    st.write("**Vertical (Up ↕ Down)**")
    st.progress(int((live_vy + 1) / 2 * 100))
    st.caption(f"y = {live_vy:+.3f}")

    st.write("**Strength (Gentle → Strong)**")
    st.progress(int(live_s * 100))
    st.caption(f"strength = {live_s:.3f}")

    throw = st.button("🎯 THROW", type="primary", use_container_width=True)

    if throw and st.session_state.mode == "calibrating":
        st.session_state.freeze = (hx, vy, strength)
        st.session_state.mode = "flying"
        st.session_state.flight_frame = 0

with right:
    st.subheader("Dartboard")

    # Dart flight animation: show a "moving dart" before landing
    if st.session_state.mode == "flying":
        # advance flight
        st.session_state.flight_frame += 1
        frames = 12
        progress = min(st.session_state.flight_frame / frames, 1.0)

        # Show board without hit + a simple "dart approaching" text/progress
        fig = draw_board(hit=None, title=f"Dart approaching… {int(progress*100)}%")
        st.pyplot(fig, clear_figure=True)

        if progress >= 1.0:
            # Compute landing from frozen values:
            hx, vy, s = st.session_state.freeze

            # Map calibration to target aim (normalized)
            # Aim point:
            aim_x = 0.70 * hx
            aim_y = 0.70 * vy

            # Strength influences noise/spread (gentle=more stable, strong=more overshoot)
            # You can tune these to create teaching scenarios.
            sigma = 0.03 + 0.10 * s
            overshoot = 0.00 + 0.12 * s

            x = aim_x + np.random.normal(0, sigma) + overshoot * np.sign(aim_x) * 0.2
            y = aim_y + np.random.normal(0, sigma) + overshoot * np.sign(aim_y) * 0.2

            st.session_state.hit = (x, y)
            sc = score_dart(x, y)
            st.session_state.last_score = sc

            # Save record (this is your dataset for SPC etc.)
            r = math.sqrt(x*x + y*y)
            st.session_state.data.append({
                "throw_id": len(st.session_state.data) + 1,
                "hx": hx, "vy": vy, "strength": s,
                "x": x, "y": y, "radius": r,
                "score": sc
            })

            st.session_state.mode = "landed"

    if st.session_state.mode == "landed":
        x, y = st.session_state.hit
        sc = st.session_state.last_score
        fig = draw_board(hit=(x, y), title=f"LANDED — Score: {sc}")
        st.pyplot(fig, clear_figure=True)

        st.metric("Score", sc)
        st.caption("Use this score + landing data for Descriptive Stats / SPC / Capability, etc.")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Next Throw ↻", use_container_width=True):
                st.session_state.mode = "calibrating"
                st.session_state.freeze = None
                st.session_state.hit = None
                st.session_state.last_score = None
        with c2:
            if st.button("Reset Session 🧹", use_container_width=True):
                st.session_state.mode = "calibrating"
                st.session_state.t = 0
                st.session_state.freeze = None
                st.session_state.hit = None
                st.session_state.last_score = None
                st.session_state.data = []

# Dataset view
st.divider()
st.subheader("Collected Data (for SPC / Analytics)")
if st.session_state.data:
    st.dataframe(st.session_state.data, use_container_width=True)
    # Export
    import pandas as pd
    df = pd.DataFrame(st.session_state.data)
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "dart_data.csv", "text/csv")
else:
    st.info("No throws yet. Hit THROW to generate your first data point.")
