# backend/main.py  (local overlay only + better season mapping + robust Pillow usage)
import os
import uuid
import json
import math
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import requests

import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# ---------- Pillow resampling compatibility ----------
try:
    # Pillow >= 9.1.0
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except Exception:
    # older Pillow
    try:
        RESAMPLE_LANCZOS = Image.LANCZOS
    except Exception:
        RESAMPLE_LANCZOS = Image.BICUBIC
    try:
        RESAMPLE_BICUBIC = Image.BICUBIC
    except Exception:
        RESAMPLE_BICUBIC = Image.NEAREST
    try:
        RESAMPLE_NEAREST = Image.NEAREST
    except Exception:
        RESAMPLE_NEAREST = Image.NEAREST

# ---------- Paths & config ----------
BASE_DIR = os.path.dirname(__file__)
OUTFITS_DIR = os.path.join(BASE_DIR, "outfits")
THUMBS_DIR = os.path.join(BASE_DIR, "thumbs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")
CATALOG_PATH = os.path.join(DATA_DIR, "catalog.json")

WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "")  # set in your environment

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- FastAPI app ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection


def load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Utilities ----------
def ensure_image(bytestr):
    nparr = np.frombuffer(bytestr, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def palette_for_label(label):
    mapping = {
        "light": ["#001f3f", "#808000", "#ffd1dc", "#800020"],  # navy, olive, pastel pink, burgundy
        "medium": ["#008080", "#8b0000", "#ffdb58", "#8b5e3c"],  # teal, maroon, mustard, earth
        "deep": ["#4169e1", "#ffd700", "#ff0000", "#fffff0"],    # royal blue, gold, red, ivory
    }
    return mapping.get(label, mapping["medium"])


# ---------- Skin tone detection ----------
def extract_skin_rgb(bgr_image):
    # Use face detection to get cheek-region; fall back to center crop
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        h, w = bgr_image.shape[:2]

        if results.detections:
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box
            x = int(max(0, bbox.xmin * w))
            y = int(max(0, bbox.ymin * h))
            bw = int(min(w, bbox.width * w))
            bh = int(min(h, bbox.height * h))
            # choose cheek region: slightly below top of bbox and to sides
            cx = x + bw // 2
            cy = y + int(bh * 0.45)
            r = max(10, int(min(bw, bh) * 0.15))
            x1 = max(0, cx - r)
            y1 = max(0, cy - r)
            x2 = min(w, cx + r)
            y2 = min(h, cy + r)
            crop = bgr_image[y1:y2, x1:x2]
        else:
            # fallback: central upper-body crop
            cy, cx = h // 3, w // 2
            r = min(w, h) // 8
            crop = bgr_image[max(0, cy - r):min(h, cy + r),
                             max(0, cx - r):min(w, cx + r)]

        if crop is None or crop.size == 0:
            return None

        pixels = crop.reshape(-1, 3)
        # filter out extreme brightness/very dark pixels
        mask = (pixels.sum(axis=1) > 60) & (pixels.sum(axis=1) < 700)
        if mask.sum() == 0:
            use = pixels
        else:
            use = pixels[mask]

        avg = np.median(use, axis=0).astype(int)  # BGR median
        rgb = (int(avg[2]), int(avg[1]), int(avg[0]))
        return rgb


def detect_skin_label(rgb):
    r, g, b = rgb
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if lum > 180:
        return "light"
    if lum > 120:
        return "medium"
    return "deep"


# ---------- Pose + overlay ----------
def is_full_body(pose_landmarks, img_shape):
    # require shoulders and hips to be detected & reasonable vertical separation
    h, w = img_shape[:2]
    lm = pose_landmarks.landmark
    try:
        L_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        R_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        L_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        R_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
    except Exception:
        return False

    def valid(point):
        return 0 <= point.x <= 1 and 0 <= point.y <= 1

    if not (valid(L_sh) and valid(R_sh) and valid(L_hip) and valid(R_hip)):
        return False

    shoulder_y = (L_sh.y + R_sh.y) / 2
    hip_y = (L_hip.y + R_hip.y) / 2
    # require hips significantly below shoulders
    return (hip_y - shoulder_y) * h > (h * 0.08)  # at least 8% of image height


def overlay_outfit(user_bgr, outfit_path, scale_adj=1.0, y_offset=0):
    """
    Local overlay:
      - uses shoulder width to compute outfit width
      - rotates outfit to match shoulder angle
      - uses torso length (shoulder->hip) to compute vertical placement.
    """
    with mp_pose.Pose(static_image_mode=True) as pose:
        rgb = cv2.cvtColor(user_bgr, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if not result.pose_landmarks:
            return None, "no_person"
        if not is_full_body(result.pose_landmarks, user_bgr.shape):
            return None, "not_full_body"

        h, w = user_bgr.shape[:2]
        lm = result.pose_landmarks.landmark

        LS = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        RS = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        LH = lm[mp_pose.PoseLandmark.LEFT_HIP]
        RH = lm[mp_pose.PoseLandmark.RIGHT_HIP]

        left_pt = (int(LS.x * w), int(LS.y * h))
        right_pt = (int(RS.x * w), int(RS.y * h))
        shoulder_w_px = max(10, abs(right_pt[0] - left_pt[0]))

        # torso height = pixel distance between avg shoulders and avg hips
        shoulder_y_px = int(((LS.y + RS.y) / 2) * h)
        hip_y_px = int(((LH.y + RH.y) / 2) * h)
        torso_h = max(10, abs(hip_y_px - shoulder_y_px))

        # compute rotation angle of shoulders (degrees)
        dx = right_pt[0] - left_pt[0]
        dy = right_pt[1] - left_pt[1]
        angle_deg = math.degrees(math.atan2(dy, dx))

        try:
            outfit = Image.open(outfit_path).convert("RGBA")
        except Exception:
            return None, "outfit_not_found"

        # desired width based on shoulder width + scale
        try:
            desired_w = int(shoulder_w_px * 1.6 * float(scale_adj))
        except Exception:
            desired_w = outfit.width

        if desired_w <= 0:
            desired_w = outfit.width

        ratio = desired_w / float(outfit.width)
        new_h = max(1, int(outfit.height * ratio))

        # resize
        try:
            outfit_resized = outfit.resize((desired_w, new_h), RESAMPLE_LANCZOS)
        except Exception:
            outfit_resized = outfit.resize((desired_w, new_h), RESAMPLE_BICUBIC)

        # rotate using bicubic
        try:
            outfit_rotated = outfit_resized.rotate(
                angle_deg, expand=True, resample=RESAMPLE_BICUBIC
            )
        except Exception:
            outfit_rotated = outfit_resized.rotate(angle_deg, expand=True)

        # placement
        anchor_x = int((left_pt[0] + right_pt[0]) / 2)
        top_target_y = shoulder_y_px + int(torso_h * 0.03) + int(y_offset)
        paste_x = anchor_x - outfit_rotated.width // 2
        paste_y = top_target_y - int(outfit_rotated.height * 0.12)

        # paste on person
        person = Image.fromarray(rgb).convert("RGBA")
        paste_x = max(-outfit_rotated.width, min(person.width, paste_x))
        paste_y = max(-outfit_rotated.height, min(person.height, paste_y))

        try:
            person.paste(outfit_rotated, (paste_x, paste_y), outfit_rotated)
        except Exception:
            try:
                person.paste(outfit_rotated, (paste_x, paste_y))
            except Exception:
                return None, "paste_failed"

        out_bgr = cv2.cvtColor(np.array(person.convert("RGB")), cv2.COLOR_RGB2BGR)
        return out_bgr, None


# ---------- Endpoints ----------
@app.get("/catalog")
def catalog():
    return {"items": load_catalog()}


@app.post("/try_on")
async def try_on(
    image: UploadFile = File(...),
    item_id: str = Form(...),
    scale: Optional[float] = Form(1.0),
    y_offset: Optional[int] = Form(0),
):
    """
    Local-only try-on:
      1) Load user image
      2) Find catalog item and its outfit_png
      3) Run MediaPipe overlay
      4) Return composed image + simple recommendations
    """
    contents = await image.read()
    user_img = ensure_image(contents)
    if user_img is None:
        return JSONResponse({"error": "could_not_read_image"}, status_code=400)

    catalog_data = load_catalog()
    item = next((it for it in catalog_data if it["id"] == item_id), None)
    if not item:
        return JSONResponse({"error": "item_not_found"}, status_code=404)

    outfit_rel = item.get("outfit_png", "")
    if not outfit_rel:
        return JSONResponse({"error": "outfit_not_available"}, status_code=400)

    outfit_file = os.path.join(BASE_DIR, outfit_rel.lstrip("/"))
    if not os.path.exists(outfit_file):
        return JSONResponse({"error": "outfit_file_missing"}, status_code=500)

    try:
        out_img, err = overlay_outfit(
            user_img, outfit_file, scale_adj=scale, y_offset=y_offset
        )
        if err:
            return JSONResponse({"error": err}, status_code=400)

        out_name = f"tryon_{item_id}_{uuid.uuid4().hex}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, out_img)

        recommends = []
        for rid in item.get("recommend", []):
            rec = next((it for it in catalog_data if it["id"] == rid), None)
            if rec:
                recommends.append(
                    {"id": rec["id"], "name": rec["name"], "thumb": rec.get("thumb")}
                )

        return {"result_image": f"/outputs/{out_name}", "recommendations": recommends}
    except Exception as e:
        return JSONResponse(
            {"error": f"fallback_overlay_failed: {e}"}, status_code=500
        )


@app.post("/skin_tone")
async def skin_tone(image: UploadFile = File(...)):
    contents = await image.read()
    img = ensure_image(contents)
    if img is None:
        return JSONResponse({"error": "could_not_read_image"}, status_code=400)

    rgb = extract_skin_rgb(img)
    if rgb is None:
        return JSONResponse({"error": "could_not_detect_face"}, status_code=400)

    label = detect_skin_label(rgb)
    palette = palette_for_label(label)
    return {"skin_rgb": rgb, "label": label, "palette": palette}


@app.get("/outputs/{filename}")
def serve_output(filename: str):
    fp = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(fp):
        return JSONResponse({"error": "file_not_found"}, status_code=404)
    return FileResponse(fp)


@app.get("/outfits/{filename}")
def serve_outfit(filename: str):
    fp = os.path.join(OUTFITS_DIR, filename)
    if not os.path.exists(fp):
        return JSONResponse({"error": "file_not_found"}, status_code=404)
    return FileResponse(fp)


@app.get("/thumbs/{filename}")
def serve_thumb(filename: str):
    fp = os.path.join(THUMBS_DIR, filename)
    if not os.path.exists(fp):
        return JSONResponse({"error": "file_not_found"}, status_code=404)
    return FileResponse(fp)


# ---------- Season-based recommendations ----------
@app.get("/season_recommend")
def season_recommend(city: Optional[str] = None):
    season = None

    # 1) Try real weather if API key + city are available
    if city and WEATHER_API_KEY:
        try:
            r = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": city, "appid": WEATHER_API_KEY, "units": "metric"},
                timeout=6,
            )
            if r.status_code == 200:
                data = r.json()
                temp = data.get("main", {}).get("temp")
                if temp is not None:
                    # tuned thresholds (Celsius)
                    if temp >= 30:
                        season = "summer"
                    elif temp >= 20:
                        season = "spring"
                    elif temp >= 12:
                        season = "autumn"
                    else:
                        season = "winter"
        except Exception as e:
            print("Weather API failed:", e)
            season = None

    # 2) Fallback: infer season from month
    if season is None:
        import datetime

        m = datetime.datetime.now().month
        if m in (12, 1, 2):
            season = "winter"
        elif m in (3, 4, 5):
            season = "spring"
        elif m in (6, 7, 8):
            season = "summer"
        else:
            season = "autumn"

    items = load_catalog()
    filtered = [
        it
        for it in items
        if season in [s.lower() for s in it.get("season", [])]
        or "all" in [s.lower() for s in it.get("season", [])]
    ]
    return {"season": season, "items": filtered}
