import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Lean Sigma Dart Lab", layout="wide")

SECTOR_ORDER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]


def angle_to_sector(theta_deg: float) -> int:
    theta = theta_deg % 360.0
    shifted = (90.0 - theta) % 360.0
    idx = int((shifted + 9.0) // 18.0) % 20
    return SECTOR_ORDER[idx]


def score_dart(x: float, y: float) -> int:
    r = math.sqrt(x * x + y * y)
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
    fig, ax = plt.subplots(figsize=(5.8, 5.8))  # a bit bigger like your screenshot
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")

    # outline + simple rings
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, linewidth=2))
    for rr in [0.10, 0.55, 0.60, 0.95, 1.00]:
        ax.add_patch(plt.Circle((0, 0), rr, fill=False, linestyle="--", linewidth=1))

    # sector lines
    for i in range(20):
        ang = math.radians(90 - i * 18)
        ax.plot([0, math.cos(ang)], [0, math.sin(ang)], linewidth=0.7)

    # bull
    ax.add_patch(plt.Circle((0, 0), 0.10, fill=False, linewidth=2))
    ax.add_patch(plt.Circle((0, 0), 0.05, fill=False, linewidth=2))

    # hit dot (orange-like, with glow)
    if hit is not None:
        x, y = hit
        ax.scatter([x], [y], s=110, marker="o")              # dot
        ax.scatter([x], [y], s=260, marker="o", alpha=0.25)  # glow

    ax.set_title(title)
    return fig


# -----------------------------
# Session state
# -----------------------------
if "t" not in st.session_state:
    st.session_state.t = 0
if "hit" not in st.session_state:
    st.session_state.hit = None
if "last_score" not in st.session_state:
    st.session_state.last_score = None
if "data" not in st.session_state:
    st.session_state.data = []
if "freeze" not in st.session_state:
    st.session_state.freeze = None  # (hx, vy, strength)

# Auto-refresh ONLY for calibration bars (lightweight)
st_autorefresh(interval=140, key="tick")
st.session_state.t += 1

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Calibration Deck")

    t = st.session_state.t

    # moving signals
    hx = math.sin(t / 7.0)              # -1..1
    vy = math.sin(t / 9.0 + 1.7)        # -1..1
    strength = (math.sin(t / 11.0 + 0.8) + 1) / 2  # 0..1

    # If last throw exists, show frozen values (optional: comment out if you want always moving)
    if st.session_state.freeze is not None:
        live_hx, live_vy, live_s = st.session_state.freeze
    else:
        live_hx, live_vy, live_s = hx, vy, strength

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

    if throw:
        # freeze the current values
        st.session_state.freeze = (hx, vy, strength)

        # compute hit immediately (NO animation)
        s = strength
        aim_x = 0.70 * hx
        aim_y = 0.70 * vy

        sigma = 0.03 + 0.10 * s
        overshoot = 0.00 + 0.12 * s

        x = aim_x + np.random.normal(0, sigma) + overshoot * np.sign(aim_x) * 0.2
        y = aim_y + np.random.normal(0, sigma) + overshoot * np.sign(aim_y) * 0.2

        st.session_state.hit = (x, y)
        sc = score_dart(x, y)
        st.session_state.last_score = sc

        r = math.sqrt(x * x + y * y)
        st.session_state.data.append(
            {
                "throw_id": len(st.session_state.data) + 1,
                "hx": hx,
                "vy": vy,
                "strength": s,
                "x": x,
                "y": y,
                "radius": r,
                "score": sc,
            }
        )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear last hit", use_container_width=True):
            st.session_state.hit = None
            st.session_state.last_score = None
            st.session_state.freeze = None
    with c2:
        if st.button("Reset session 🧹", use_container_width=True):
            st.session_state.hit = None
            st.session_state.last_score = None
            st.session_state.freeze = None
            st.session_state.data = []
            st.session_state.t = 0

with right:
    st.subheader("Dartboard")

    # ✅ IMPORTANT: only draw the board ONCE per rerun, but reruns are cheap now because board is not drawn repeatedly in animation.
    # Still, we'll keep it minimal.
    if st.session_state.hit is None:
        fig = draw_board(hit=None, title="Ready")
    else:
        fig = draw_board(hit=st.session_state.hit, title=f"Score: {st.session_state.last_score}")

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    if st.session_state.last_score is not None:
        st.metric("Score", st.session_state.last_score)

# Dataset
st.divider()
st.subheader("Collected Data (for SPC / Analytics)")
if st.session_state.data:
    st.dataframe(st.session_state.data, use_container_width=True)

    import pandas as pd
    df = pd.DataFrame(st.session_state.data)
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "dart_data.csv",
        "text/csv",
    )
else:
    st.info("No throws yet. Hit THROW to generate your first data point.")
