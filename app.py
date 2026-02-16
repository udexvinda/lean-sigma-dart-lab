import math
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import altair as alt

st.set_page_config(page_title="Lean Sigma Dart Lab", layout="wide")

# -----------------------------
# Scoring: Highest at center (0,0), decreases with distance (radius)
# -----------------------------
def score_by_distance(x: float, y: float, max_score: int = 100, r_zero: float = 1.0) -> int:
    r = math.sqrt(x * x + y * y)
    if r >= r_zero:
        return 0
    s = max_score * (1.0 - (r / r_zero) ** 2)  # quadratic falloff
    return int(round(max(0.0, s)))


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

if "show_throws" not in st.session_state:
    st.session_state.show_throws = False


# -----------------------------
# Auto-refresh ONLY to animate the calibration bars
# Freeze calibration movement when viewing throws.
# -----------------------------
if not st.session_state.show_throws:
    st_autorefresh(interval=140, key="tick")
    st.session_state.t += 1


# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Calibration Deck")

    t = st.session_state.t

    # Moving signals (based on t)
    hx = math.sin(t / 7.0)  # -1..1
    vy = math.sin(t / 9.0 + 1.7)  # -1..1
    strength = (math.sin(t / 11.0 + 0.8) + 1) / 2  # 0..1

    # If viewing throws: freeze deck at current live values (if not already frozen)
    if st.session_state.show_throws and st.session_state.freeze is None:
        st.session_state.freeze = (hx, vy, strength)

    # Display frozen values if present; otherwise live
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

    throw = st.button("🎯 THROW", type="primary", use_container_width=True, disabled=st.session_state.show_throws)

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

        # Keep visible (even if miss)
        x = clamp(x, -1.2, 1.2)
        y = clamp(y, -1.2, 1.2)

        st.session_state.hit = (x, y)

        # Scoring: based on distance from center
        sc = score_by_distance(x, y, max_score=100, r_zero=1.0)
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
            # If not viewing throws, allow movement again by unfreezing
            if not st.session_state.show_throws:
                st.session_state.freeze = None
    with c2:
        if st.button("Reset session 🧹", use_container_width=True):
            st.session_state.hit = None
            st.session_state.last_score = None
            st.session_state.freeze = None
            st.session_state.data = []
            st.session_state.t = 0
            st.session_state.show_throws = False


with right:
    st.subheader("Virtual Board")

    # Show only the latest hit on the right panel
    if st.session_state.hit is None:
        dfp = pd.DataFrame({"x": [], "y": []})
    else:
        x, y = st.session_state.hit
        dfp = pd.DataFrame({"x": [x], "y": [y]})

    chart = (
        alt.Chart(dfp)
        .mark_circle(size=160)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-1.2, 1.2])),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-1.2, 1.2])),
            tooltip=["x:Q", "y:Q"],
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

    # Buttons row: Download CSV + View my THROWs
    b1, b2 = st.columns([1, 1])

    with b1:
        st.download_button(
            "Download CSV",
            dfd.to_csv(index=False).encode("utf-8"),
            "dart_data.csv",
            "text/csv",
            use_container_width=True,
        )

    with b2:
        if st.button("View my THROWs", use_container_width=True):
            st.session_state.show_throws = not st.session_state.show_throws
            # When turning ON: freeze the deck at current live values
            if st.session_state.show_throws and st.session_state.freeze is None:
                st.session_state.freeze = (hx, vy, strength)
            # When turning OFF: unfreeze to resume movement
            if not st.session_state.show_throws:
                st.session_state.freeze = None

    # Show all throws chart (labeled by throw_id) when toggled on
    if st.session_state.show_throws:
        st.markdown("#### My THROWs (all points labeled by throw_id)")

        base = (
            alt.Chart(dfd)
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=[-1.2, 1.2])),
                y=alt.Y("y:Q", scale=alt.Scale(domain=[-1.2, 1.2])),
                tooltip=["throw_id:Q", "x:Q", "y:Q", "radius:Q", "score:Q"],
            )
            .properties(height=520)
        )

        points = base.mark_circle(size=120)
        labels = base.mark_text(dx=10, dy=-10, fontSize=12).encode(text=alt.Text("throw_id:Q"))

        st.altair_chart(points + labels, use_container_width=True)

else:
    st.info("No throws yet. Hit THROW to generate your first data point.")
