# backend/main.py
import os
import uuid
import json
import base64
import math
import time
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import requests

import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# ---------- Pillow resampling compatibility ----------
try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except Exception:
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

def palette_for_label(label):
    mapping = {
        "light": ["#ffd1dc", "#fff2e6", "#fbe8c7", "#ffd3a5"],
        "medium": ["#ffdb58", "#f4a261", "#c084fc", "#8b5e3c"],
        "deep": ["#4169e1", "#ff0000", "#ffd700", "#2f855a"],
    }
    return mapping.get(label, mapping["medium"])

# Skin tone detection
def extract_skin_rgb(bgr_image):
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
            cx = x + bw // 2
            cy = y + int(bh * 0.45)
            r = max(10, int(min(bw, bh) * 0.15))
            x1 = max(0, cx - r)
            y1 = max(0, cy - r)
            x2 = min(w, cx + r)
            y2 = min(h, cy + r)
            crop = bgr_image[y1:y2, x1:x2]
        else:
            cy, cx = h // 3, w // 2
            r = min(w, h) // 8
            crop = bgr_image[max(0, cy - r):min(h, cy + r),
                             max(0, cx - r):min(w, cx + r)]

        if crop is None or crop.size == 0:
            return None

        pixels = crop.reshape(-1, 3)
        mask = (pixels.sum(axis=1) > 60) & (pixels.sum(axis=1) < 700)
        use = pixels[mask] if mask.sum() > 0 else pixels
        avg = np.median(use, axis=0).astype(int)
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

# Pose + overlay (local fallback)
def is_full_body(pose_landmarks, img_shape):
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
    return (hip_y - shoulder_y) * h > (h * 0.08)

def overlay_outfit(user_bgr, outfit_path):
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

        shoulder_y_px = int(((LS.y + RS.y) / 2) * h)
        hip_y_px = int(((LH.y + RH.y) / 2) * h)
        torso_h = max(10, abs(hip_y_px - shoulder_y_px))

        dx = right_pt[0] - left_pt[0]
        dy = right_pt[1] - left_pt[1]
        angle_deg = math.degrees(math.atan2(dy, dx))

        try:
            outfit = Image.open(outfit_path).convert("RGBA")
        except Exception:
            return None, "outfit_not_found"

        desired_w = int(shoulder_w_px * 1.6)
        if desired_w <= 0:
            desired_w = outfit.width
        ratio = desired_w / float(outfit.width)
        new_h = max(1, int(outfit.height * ratio))

        try:
            outfit_resized = outfit.resize((desired_w, new_h), RESAMPLE_LANCZOS)
        except Exception:
            outfit_resized = outfit.resize((desired_w, new_h), RESAMPLE_BICUBIC)

        try:
            outfit_rotated = outfit_resized.rotate(angle_deg, expand=True, resample=RESAMPLE_BICUBIC)
        except Exception:
            outfit_rotated = outfit_resized.rotate(angle_deg, expand=True)

        anchor_x = int((left_pt[0] + right_pt[0]) / 2)
        top_target_y = shoulder_y_px + int(torso_h * 0.03)
        paste_x = anchor_x - outfit_rotated.width // 2
        paste_y = top_target_y - int(outfit_rotated.height * 0.12)

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

# ---------- OOT-diffusion Space helper (robust & debuggable) ----------
def call_oot_space(person_bytes: bytes, outfit_bytes: bytes, timeout: int = 90):
    """
    Robust call to HuggingFace Space endpoint for OOTDiffusion.
    Uses Authorization header if HF_SPACE_TOKEN or HF_TOKEN provided.
    Tries JSON first, then multipart fallback. Raises RuntimeError with detailed info on failure.
    """
    space_endpoint = "https://levihsu-ootdiffusion.hf.space/run/predict"

    


    # ensure no trailing slash issues
    

    token = os.environ.get("HF_SPACE_TOKEN") or os.environ.get("HF_TOKEN") or os.environ.get("HF_API_TOKEN")
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # prepare JSON payload (data URLs)
    p_b64 = base64.b64encode(person_bytes).decode("utf-8")
    g_b64 = base64.b64encode(outfit_bytes).decode("utf-8")
    payload = {"data": [f"data:image/png;base64,{p_b64}", f"data:image/png;base64,{g_b64}"]}

    # Helper to summarize response for error messages (keep short)
    def short(s, n=800):
        try:
            return s if len(s) <= n else s[:n] + "..."
        except Exception:
            return "<non-text-response>"

    # 1) Try JSON POST
    try:
        r = requests.post(space_endpoint, json=payload, headers=headers, timeout=timeout)
    except Exception as e:
        # network error
        raise RuntimeError(f"OOT space request failure (network exception): {e}")

    # if not OK, attempt multipart fallback (and capture responses)
    if r.status_code != 200:
        resp_text = ""
        try:
            resp_text = r.text
        except Exception:
            resp_text = "<unreadable response body>"
        # Attempt multipart fallback
        try:
            files = {
                "person": ("person.png", person_bytes, "image/png"),
                "garment": ("garment.png", outfit_bytes, "image/png"),
            }
            r2 = requests.post(space_endpoint, files=files, headers=headers, timeout=timeout)
            if r2.status_code == 200:
                r = r2
            else:
                # neither method worked: raise reason with status codes and snippets
                snippet1 = short(resp_text)
                try:
                    snippet2 = short(r2.text)
                except Exception:
                    snippet2 = "<unreadable fallback response>"
                raise RuntimeError(
                    f"OOT endpoint returned {r.status_code} (json-post). resp_snippet={snippet1} ; "
                    f"multipart returned {r2.status_code}. fallback_snippet={snippet2} ; "
                    f"Ensure HF token is correct and Space is accessible."
                )
        except Exception as e2:
            raise RuntimeError(f"OOT multipart fallback failed: {e2} ; initial_status={r.status_code} initial_snippet={short(resp_text)}")

    # parse JSON
    try:
        resp_json = r.json()
    except Exception as e:
        # include text to aid debugging
        body = ""
        try:
            body = r.text
        except Exception:
            body = "<non-text>"
        raise RuntimeError(f"OOT returned non-json: {e} - response_text_snippet={short(body)}")

    # find base64 image in the response object (search recursively)
    def find_base64(obj):
        if isinstance(obj, str) and obj.startswith("data:image"):
            return obj.split(",", 1)[1]
        if isinstance(obj, dict):
            for v in obj.values():
                res = find_base64(v)
                if res:
                    return res
        if isinstance(obj, list):
            for v in obj:
                res = find_base64(v)
                if res:
                    return res
        return None

    image_b64 = find_base64(resp_json) or (isinstance(resp_json.get("data"), list) and find_base64(resp_json.get("data")))
    recommendations = resp_json.get("recommendations") if isinstance(resp_json, dict) else []

    if not image_b64:
        # include helpful debugging pieces from response keys/text
        try:
            keys = ", ".join(map(str, resp_json.keys())) if isinstance(resp_json, dict) else str(type(resp_json))
            sample = json.dumps(resp_json)[:1000]
        except Exception:
            keys = "<could-not-list-keys>"
            sample = "<could-not-serialize-response>"
        raise RuntimeError(f"OOT space did not return an image. response_keys={keys} response_sample={sample}. If this is a gated Space, ensure HF_SPACE_TOKEN has 'read' permission and can access the Space.")

    return {"result_image_b64": image_b64, "recommendations": recommendations}

# Optional quick startup probe to show accessible status (non-fatal)
def check_space_endpoint():
    endpoint = "https://hf.space/embed/levihsu/OOTDiffusion/+/api/predict"
    token = os.environ.get("HF_SPACE_TOKEN") or os.environ.get("HF_TOKEN") or os.environ.get("HF_API_TOKEN")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        # Use HEAD or OPTIONS to check reachability quickly
        r = requests.options(endpoint, headers=headers, timeout=8)
        status = r.status_code
        txt = (r.text[:400] + "...") if r.text else ""
        print(f"[startup] OOTDiffusion space probe: status={status}, snippet={txt}")
        if status in (401, 403):
            print("[startup] WARNING: Space returned 401/403. Check HF_SPACE_TOKEN permissions (must have read access to gated models).")
        if status == 404:
            print("[startup] WARNING: Space returned 404. Ensure the space id 'levihsu/OOTDiffusion' exists and is public or accessible to your account.")
    except Exception as e:
        print(f"[startup] OOTDiffusion probe failed: {e} (this is non-fatal)")

# Run probe at import/start (non-blocking-ish)
try:
    # small sleep to avoid noisy logs if requests not ready — but do it quickly
    check_space_endpoint()
except Exception:
    pass

# ---------- Endpoints ----------
@app.get("/catalog")
def catalog():
    return {"items": load_catalog()}

@app.post("/try_on")
async def try_on(
    image: UploadFile = File(...),
    item_id: str = Form(...),
):
    """
    Try-on flow:
      1) load user image
      2) find item and its outfit_png (local)
      3) try OOT-diffusion Space (person + garment)
      4) if fails -> local overlay fallback (kept as last resort)
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

    # read outfit bytes
    with open(outfit_file, "rb") as f:
        outfit_bytes = f.read()

    remote_error = None

    # 1) Try OOT-diffusion Space
    try:
        result = call_oot_space(contents, outfit_bytes, timeout=120)
        output_bytes = base64.b64decode(result["result_image_b64"])
        out_name = f"tryon_{item_id}_{uuid.uuid4().hex}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        with open(out_path, "wb") as f:
            f.write(output_bytes)
        return {"result_image": f"/outputs/{out_name}", "recommendations": result.get("recommendations", [])}
    except Exception as e:
        # save the remote error message for diagnostics (returned to frontend in fallback case)
        remote_error = {"error": "oot_remote_failed", "message": str(e)}
        print("OOT remote failed:", remote_error)

    # 2) Local overlay fallback
    try:
        out_img, err = overlay_outfit(user_img, outfit_file)
        if err:
            return JSONResponse({"error": err, "remote_error": remote_error}, status_code=400)
        out_name = f"tryon_{item_id}_{uuid.uuid4().hex}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, out_img)

        recommends = []
        for rid in item.get("recommend", []):
            rec = next((it for it in catalog_data if it["id"] == rid), None)
            if rec:
                recommends.append({"id": rec["id"], "name": rec["name"], "thumb": rec.get("thumb")})

        # include remote_error so frontend can display why remote failed
        resp = {"result_image": f"/outputs/{out_name}", "recommendations": recommends}
        if remote_error:
            resp["remote_error"] = remote_error
        return resp
    except Exception as e:
        return JSONResponse({"error": f"fallback_overlay_failed: {e}", "remote_error": remote_error}, status_code=500)

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

# Season recommendations (India month-based mapping)
@app.get("/season_recommend")
def season_recommend(city: Optional[str] = None, month: Optional[int] = Query(None, ge=1, le=12)):
    month_to_key = {
        12: "winter", 1: "winter", 2: "winter",
        3: "summer", 4: "summer", 5: "summer",
        6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
        10: "post-monsoon", 11: "post-monsoon"
    }
    key_to_display = {
        "winter": "Winter",
        "summer": "Summer",
        "monsoon": "Monsoon (Rainy Season)",
        "post-monsoon": "Post-Monsoon / Autumn"
    }
    import datetime
    if month is not None:
        m = month
    else:
        m = datetime.datetime.now().month
    season_key = month_to_key.get(m, "summer")
    season_display = key_to_display.get(season_key, season_key.title())
    items = load_catalog()
    def normalize_season_tokens(it):
        seasons = it.get("season", []) or []
        out = []
        for s in seasons:
            if not isinstance(s, str): continue
            t = s.strip().lower()
            if t in ("autumn", "post-monsoon", "post monsoon"): out.append("post-monsoon")
            elif t in ("rainy", "rainy season", "monsoon", "monsun"): out.append("monsoon")
            elif t in ("winter",): out.append("winter")
            elif t in ("summer",): out.append("summer")
            elif t == "all": out.append("all")
            else: out.append(t)
        return set(out)
    def item_matches(it, key):
        tokens = normalize_season_tokens(it)
        return ("all" in tokens) or (key in tokens)
    filtered = [it for it in items if item_matches(it, season_key)]
    return {"season": season_display, "season_key": season_key, "season_display": season_display, "items": filtered}
