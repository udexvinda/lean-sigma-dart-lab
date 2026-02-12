import math
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import altair as alt

st.set_page_config(page_title="Lean Sigma Dart Lab", layout="wide")

# -----------------------------
# Dartboard scoring (simplified)
# -----------------------------
SECTOR_ORDER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]


def angle_to_sector(theta_deg: float) -> int:
    """theta_deg: 0 at +x axis, CCW. Dartboard sectors start at 20 centered at 90 degrees (top)."""
    theta = theta_deg % 360.0
    shifted = (90.0 - theta) % 360.0
    idx = int((shifted + 9.0) // 18.0) % 20
    return SECTOR_ORDER[idx]


def score_dart(x: float, y: float) -> int:
    """
    x,y in normalized board coordinates where radius 1.0 is board edge.
    Simplified rings:
      - Bull: r <= 0.05 => 50
      - Outer bull: r <= 0.10 => 25
      - Triple ring: 0.55 <= r <= 0.60
      - Double ring: 0.95 <= r <= 1.00
      - Single otherwise within r<=1
      - Miss if r>1
    """
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


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# -----------------------------
# Session state init
# -----------------------------
if "t" not in st.session_state:
    st.session_state.t = 0

if "freeze" not in st.session_state:
    st.session_state.freeze = None  # (hx, vy, strength)

if "hit" not in st.session_state:
    st.session_state.hit = None  # (x, y)

if "last_score" not in st.session_state:
    st.session_state.last_score = None

if "data" not in st.session_state:
    st.session_state.data = []  # list of dict rows


# -----------------------------
# Auto-refresh ONLY to animate the calibration bars
# -----------------------------
st_autorefresh(interval=140, key="tick")
st.session_state.t += 1


# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Calibration Deck")

    t = st.session_state.t

    # Moving signals
    hx = math.sin(t / 7.0)  # -1..1
    vy = math.sin(t / 9.0 + 1.7)  # -1..1
    strength = (math.sin(t / 11.0 + 0.8) + 1) / 2  # 0..1

    # Display frozen values if a throw happened; otherwise live
    if st.session_state.freeze is None:
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

    if throw:
        # Freeze values at the throw moment
        st.session_state.freeze = (hx, vy, strength)

        # Map to aim point (normalized board coords)
        aim_x = 0.70 * hx
        aim_y = 0.70 * vy

        # Strength affects spread
        sigma = 0.03 + 0.10 * strength
        overshoot = 0.00 + 0.12 * strength

        x = aim_x + np.random.normal(0, sigma) + overshoot * np.sign(aim_x) * 0.2
        y = aim_y + np.random.normal(0, sigma) + overshoot * np.sign(aim_y) * 0.2

        # Allow slight overshoot outside the board for "miss" (kept in visible frame)
        x = clamp(x, -1.2, 1.2)
        y = clamp(y, -1.2, 1.2)

        st.session_state.hit = (x, y)
        sc = score_dart(x, y)
        st.session_state.last_score = sc

        r = math.sqrt(x * x + y * y)
        st.session_state.data.append(
            {
                "throw_id": len(st.session_state.data) + 1,
                "hx": float(hx),
                "vy": float(vy),
                "strength": float(strength),
                "x": float(x),
                "y": float(y),
                "radius": float(r),
                "score": int(sc),
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

    # Only plot the real hit (no fake bound points)
    if st.session_state.hit is None:
        dfp = pd.DataFrame({"x": [], "y": []})
    else:
        x, y = st.session_state.hit
        dfp = pd.DataFrame({"x": [x], "y": [y]})

    chart = (
        alt.Chart(dfp)
        .mark_circle(size=140)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-1.2, 1.2])),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-1.2, 1.2])),
        )
        .properties(height=520)
    )

    st.altair_chart(chart, use_container_width=True)

    if st.session_state.last_score is not None:
        st.metric("Score", st.session_state.last_score)

st.divider()
st.subheader("Collected Data (for SPC / Analytics)")

if st.session_state.data:
    dfd = pd.DataFrame(st.session_state.data)
    st.dataframe(dfd, use_container_width=True)

    st.download_button(
        "Download CSV",
        dfd.to_csv(index=False).encode("utf-8"),
        "dart_data.csv",
        "text/csv",
    )
else:
    st.info("No throws yet. Hit THROW to generate your first data point.")
