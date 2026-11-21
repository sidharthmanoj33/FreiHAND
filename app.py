import json
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

st.set_page_config(page_title="Hand Pose Detection", page_icon="🖐️", layout="wide")

COLORS = [(225, 29, 72), (245, 158, 11), (16, 185, 129), (59, 130, 246), (168, 85, 247)]  # RGB per finger
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
JOINT_NAMES = ["wrist"] + [f"{f}_{j}" for f in FINGERS for j in ("mcp", "pip", "dip", "tip")]

st.title("🖐️ Hand Pose Detection: Upload a Photo")
st.caption(
    "Project context: FreiHAND dataset and toolbox (github.com/lmb-freiburg/freihand). "
    "Detector: MediaPipe Hands, which is not trained on FreiHAND. "
    "Joint order matches FreiHAND: wrist, then thumb, index, middle, ring, pinky with four joints each."
)

src = st.radio("Input", ["Upload a photo", "Use camera"], horizontal=True)
if src == "Upload a photo":
    f = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
else:
    f = st.camera_input("Take a photo")
min_conf = st.slider("Minimum detection confidence", 0.1, 0.9, 0.4, 0.05)

def detect(rgb):
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                  min_detection_confidence=min_conf) as hands:
        return hands.process(rgb)

def draw(rgb, res):
    out = rgb.copy()
    h, w = out.shape[:2]
    t = max(2, w // 250)
    for hand in res.multi_hand_landmarks:
        pts = [(int(p.x * w), int(p.y * h)) for p in hand.landmark]
        for fi in range(5):
            ids = [0] + [1 + fi * 4 + k for k in range(4)]
            for a, b in zip(ids[:-1], ids[1:]):
                cv2.line(out, pts[a], pts[b], COLORS[fi], t)
        for i, p in enumerate(pts):
            col = (255, 255, 255) if i == 0 else COLORS[(i - 1) // 4]
            cv2.circle(out, p, t * 2, col, -1)
            cv2.circle(out, p, t * 2, (0, 0, 0), 1)
    return out

if f is not None:
    img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    rgb = np.array(img)
    res = detect(rgb)
    if not res.multi_hand_landmarks:
        st.warning("No hand found. Try a clearer, closer photo with the whole hand visible.")
        st.image(img, use_container_width=True)
    else:
        n = len(res.multi_hand_landmarks)
        st.success(f"Detected {n} hand{'s' if n > 1 else ''}.")
        c1, c2 = st.columns([1.3, 1])
        c1.image(draw(rgb, res), caption="2D keypoints", use_container_width=True)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        for hand in res.multi_hand_world_landmarks:
            P = np.array([[p.x, p.z, -p.y] for p in hand.landmark])
            for fi in range(5):
                ids = [0] + [1 + fi * 4 + k for k in range(4)]
                c = np.array(COLORS[fi]) / 255
                ax.plot(P[ids, 0], P[ids, 1], P[ids, 2], color=c, lw=2)
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], color="k", s=10)
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.set_zlabel("y (m)")
        c2.pyplot(fig)
        c2.caption("3D keypoints (estimated, metres, centered on the hand). Use the plot to compare finger shapes.")

        W, H = img.size
        data = {"image_size": [W, H], "joint_names": JOINT_NAMES, "hands": []}
        for i, (lm, wl) in enumerate(zip(res.multi_hand_landmarks, res.multi_hand_world_landmarks)):
            lab = res.multi_handedness[i].classification[0]
            data["hands"].append({
                "handedness": lab.label, "score": lab.score,
                "keypoints_2d_px": [[p.x * W, p.y * H] for p in lm.landmark],
                "keypoints_3d_m": [[p.x, p.y, p.z] for p in wl.landmark],
            })
            with st.expander(f"Hand {i+1}: {lab.label} ({lab.score:.0%}) joint table"):
                st.dataframe({"joint": JOINT_NAMES,
                              "x_px": [round(p.x * W, 1) for p in lm.landmark],
                              "y_px": [round(p.y * H, 1) for p in lm.landmark],
                              "x_m": [round(p.x, 4) for p in wl.landmark],
                              "y_m": [round(p.y, 4) for p in wl.landmark],
                              "z_m": [round(p.z, 4) for p in wl.landmark]})
        st.download_button("Download keypoints (JSON)", json.dumps(data, indent=1),
                           "keypoints.json", "application/json")
else:
    st.info("Upload a hand photo to begin.")

st.markdown("---")
st.caption("The 3D values are model estimates and less reliable than the 2D positions. "
           "This app predicts keypoints only, not the MANO hand mesh described in the FreiHAND paper.")
