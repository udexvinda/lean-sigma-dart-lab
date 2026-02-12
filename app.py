import math
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Lean Sigma Dart Lab", layout="wide")

# -----------------------------
# Dartboard scoring (simplified)
# -----------------------------
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


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# -----------------------------
# Load dartboard image as base64 for HTML background
# -----------------------------
ASSET_PATH = Path(__file__).parent / "assets" / "dartboard.png"


@st.cache_data
def dartboard_base64() -> str:
    img_bytes = ASSET_PATH.read_bytes()
    return base64.b64encode(img_bytes).decode("utf-8")


def xy_to_percent(x: float, y: float, domain: float = 1.2) -> tuple[float, float]:
    """
    Convert normalized x,y into percent coordinates in the image container.
    x,y expected in [-domain, +domain], but the real board is within radius 1.0.
    - left=0%, right=100%
    - top=0%, bottom=100% (note y axis inverted for screen coords)
    """
    x = clamp(x, -domain, domain)
    y = clamp(y, -domain, domain)

    px = (x + domain) / (2 * domain) * 100.0
    py = (domain - y) / (2 * domain) * 100.0  # invert y for screen
    return px, py


def render_board_with_dot(hit_xy=None, title="Dartboard", size_px=520):
    """
    Renders a square dartboard using HTML background-image and overlays an orange dot (if hit_xy provided).
    """
    b64 = dartboard_base64()

    dot_html = ""
    if hit_xy is not None:
        x, y = hit_xy
        px, py = xy_to_percent(x, y, domain=1.2)

        # dot size and glow
        dot_html = f"""
        <div style="
            position:absolute;
            left:{px}%;
            top:{py}%;
            transform: translate(-50%, -50%);
            width:14px;
            height:14px;
            border-radius:50%;
            background:#ff8c1a;
            box-shadow: 0 0 0 6px rgba(255,140,26,0.25);
            border: 2px solid rgba(255,255,255,0.85);
        "></div>
        """

    html = f"""
    <div style="width:{size_px}px; max-width:100%;">
      <div style="font-weight:600; margin-bottom:8px;">{title}</div>
      <div style="
          position:relative;
          width:{size_px}px;
          height:{size_px}px;
          max-width:100%;
          background-image:url('data:image/png;base64,{b64}');
          background-size:contain;
          background-repeat:no-repeat;
          background-position:center;
          border-radius: 10px;
      ">
        {dot_html}
      </div>
    </div>
    """
    st.components.v1.html(html, height=size_px + 40)


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
    st.session_state.data = []

# Animate calibration deck only (lightweight)
st_autorefresh(interval=140, key="tick")
st.session_state.t += 1

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Calibration Deck")

    t = st.session_state.t

    hx = math.sin(t / 7.0)  # -1..1
    vy = math.sin(t / 9.0 + 1.7)  # -1..1
    strength = (math.sin(t / 11.0 + 0.8) + 1) / 2  # 0..1

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
        st.session_state.freeze = (hx, vy, strength)

        # Compute hit immediately (no animation)
        aim_x = 0.70 * hx
        aim_y = 0.70 * vy

        sigma = 0.03 + 0.10 * strength
        overshoot = 0.00 + 0.12 * strength

        x = aim_x + np.random.normal(0, sigma) + overshoot * np.sign(aim_x) * 0.2
        y = aim_y + np.random.normal(0, sigma) + overshoot * np.sign(aim_y) * 0.2

        # keep visible even if miss
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

    if st.session_state.hit is None:
        render_board_with_dot(hit_xy=None, title="Ready", size_px=520)
    else:
        render_board_with_dot(
            hit_xy=st.session_state.hit,
            title=f"Score: {st.session_state.last_score}",
            size_px=520,
        )

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
