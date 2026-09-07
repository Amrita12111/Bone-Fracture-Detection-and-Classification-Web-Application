import cv2, numpy as np, base64, os, requests
from ultralytics import YOLO
 
ROBOFLOW_API_KEY = "dOFUmbk8GS2FaOeDXqbJ"
 
FRACTURE_TYPE_INFO = {
    "Transverse":   {"severity":"Medium","color":"#f97316","description":"A straight horizontal break across the bone.","advice":"Immobilization with cast or splint required. Consult orthopedist within 24 hours."},
    "Oblique":      {"severity":"Medium","color":"#f97316","description":"A diagonal break running across the bone.","advice":"May require surgical fixation if displaced. Immediate orthopedic consultation recommended."},
    "Spiral":       {"severity":"High",  "color":"#ef4444","description":"A twisting break around the bone caused by rotational force.","advice":"Often requires surgical fixation. Immediate orthopedic evaluation necessary."},
    "Comminuted":   {"severity":"High",  "color":"#ef4444","description":"The bone is shattered into three or more fragments.","advice":"Surgical fixation almost always required. Emergency orthopedic consultation necessary."},
    "Greenstick":   {"severity":"Low",   "color":"#f59e0b","description":"Incomplete fracture — bone bends and partially breaks, common in children.","advice":"Casting usually sufficient. Follow up with pediatric orthopedist within 48 hours."},
    "Avulsion":     {"severity":"Medium","color":"#f97316","description":"A tendon or ligament pulls a fragment of bone away.","advice":"Immobilization required. Surgical intervention may be needed depending on displacement."},
    "Hairline":     {"severity":"Low",   "color":"#f59e0b","description":"A small thin crack often caused by repetitive stress.","advice":"Rest and immobilization usually sufficient. Follow up with orthopedist within 48 hours."},
    "Impacted":     {"severity":"Medium","color":"#f97316","description":"One bone fragment is driven into another.","advice":"Immobilization and orthopedic consultation required. Surgery may be needed."},
    "Compression":  {"severity":"High",  "color":"#ef4444","description":"Bone is crushed or collapses — often in vertebrae.","advice":"Immediate specialist referral required. Spinal compression requires urgent evaluation."},
    "Segmental":    {"severity":"High",  "color":"#ef4444","description":"The same bone is broken in two places creating a floating segment.","advice":"Surgical fixation required. Emergency orthopedic care necessary."},
    "Dislocation":  {"severity":"High",  "color":"#ef4444","description":"Fracture combined with joint dislocation.","advice":"Emergency orthopedic care required. Do not attempt to relocate joint."},
    "Pathological": {"severity":"High",  "color":"#ef4444","description":"Fracture through diseased or weakened bone.","advice":"Immediate specialist referral required. Further imaging and tests recommended."},
    "Longitudinal": {"severity":"Medium","color":"#f97316","description":"Fracture line runs along the length of the bone.","advice":"Casting or surgical fixation depending on displacement. Follow up immediately."},
}
 
# All valid fracture type names — used for API response matching
VALID_TYPES = set(FRACTURE_TYPE_INFO.keys())
 
# Roboflow API class name → canonical type name
TYPE_ALIAS = {
    # exact matches
    "transverse":          "Transverse",
    "oblique":             "Oblique",
    "spiral":              "Spiral",
    "comminuted":          "Comminuted",
    "greenstick":          "Greenstick",
    "green stick":         "Greenstick",
    "avulsion":            "Avulsion",
    "hairline":            "Hairline",
    "hairline/stress":     "Hairline",
    "stress fracture":     "Hairline",
    "impacted":            "Impacted",
    "compression":         "Compression",
    "segmental":           "Segmental",
    "pathological":        "Pathological",
    "longitudinal":        "Longitudinal",
    "dislocation":         "Dislocation",
    "fracture dislocation":"Dislocation",
    # Dataset 2 class names (Yakin dataset)
    "avulsion fracture":   "Avulsion",
    "bone fracture detection - v1 2023-03-05 5-51pm": None,  # ignore this class
    "comminuted fracture": "Comminuted",
    "dislocation fracture":"Dislocation",
    "greenstick fracture": "Greenstick",
    "hairline fracture":   "Hairline",
    "impacted fracture":   "Impacted",
    "longitudinal fracture":"Longitudinal",
    "oblique fracture":    "Oblique",
    "pathological fracture":"Pathological",
    "spiral fracture":     "Spiral",
    "transverse fracture": "Transverse",
}
 
UPPER_LIMB_MAP = {
    "elbow_positive":    "Elbow",
    "fingers_positive":  "Hand / Fingers",
    "forearm_fracture":  "Forearm",
    "humerus":           "Humerus",
    "humerus_fracture":  "Humerus",
    "shoulder_fracture": "Shoulder",
    "wrist_positive":    "Wrist",
}
 
 
# ── API helpers ────────────────────────────────────────────────────────────
def _roboflow_post(model_id, b64, timeout=12):
    try:
        r = requests.post(
            f"https://detect.roboflow.com/{model_id}",
            params={"api_key": ROBOFLOW_API_KEY},
            data=b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json().get("predictions", [])
    except Exception as e:
        print(f"[API {model_id}] {e}")
        return []
 
 
def _normalise_type(raw_class: str):
    """Convert raw API class string → canonical fracture type or None."""
    key = raw_class.strip().lower()
    # Direct alias lookup
    if key in TYPE_ALIAS:
        return TYPE_ALIAS[key]
    # Substring match against known aliases
    for alias, canonical in TYPE_ALIAS.items():
        if alias and alias in key:
            return canonical
    # Capitalise and check directly
    cap = raw_class.strip().capitalize()
    if cap in VALID_TYPES:
        return cap
    # Generic labels to ignore
    if any(g in key for g in ["fracture", "bone", "positive", "negative", "detection"]):
        return None
    return None
 
 
def get_fracture_type(image_bytes: bytes):
    """
    3-stage fracture type classification cascade.
 
    Stage 1: Roboflow primary type model  (bone-fracture-tn84w v3)
    Stage 2: Roboflow alternate type model (bone-fracture-vqdiz v2 — Dataset 1)
    Stage 3: Visual morphology analysis (OpenCV — always produces a result)
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
 
    # ── Stage 1: Primary type API ──────────────────────────────────────────
    preds = _roboflow_post("bone-fracture-tn84w/3", b64)
    if preds:
        # Sort by confidence descending
        for p in sorted(preds, key=lambda x: x["confidence"], reverse=True):
            canonical = _normalise_type(p["class"])
            if canonical:
                conf = round(p["confidence"] * 100, 1)
                print(f"[Stage1-API] {canonical} {conf}%")
                return canonical, conf, "roboflow-primary"
 
    # ── Stage 2: Alternate classification endpoint ─────────────────────────
    # Try classify endpoint (returns top class directly)
    try:
        r = requests.post(
            "https://classify.roboflow.com/bone-fracture-tn84w/3",
            params={"api_key": ROBOFLOW_API_KEY},
            data=b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            top  = data.get("top", "")
            conf = data.get("confidence", 0)
            canonical = _normalise_type(top)
            if canonical:
                print(f"[Stage2-classify] {canonical} {conf*100:.1f}%")
                return canonical, round(conf * 100, 1), "roboflow-classify"
    except Exception as e:
        print(f"[Stage2-classify] {e}")
 
    # ── Stage 3: Visual morphology (always returns a result) ───────────────
    print("[Stage3] Using visual morphology")
    return None, None, "visual-morphology"  # signal to caller to use ROI analysis
 
 
def get_body_part(image_bytes: bytes, bbox=None, img_shape=None):
    """Identify body part from X-ray using Roboflow API + positional fallback."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
 
    preds = _roboflow_post("bone-fracture-detection-cwkot/1", b64)
    if preds:
        best = max(preds, key=lambda x: x["confidence"])
        part = UPPER_LIMB_MAP.get(best["class"].lower().replace(" ", "_"))
        if part and best["confidence"] >= 0.40:
            return part, round(best["confidence"] * 100, 1)
 
    # Positional fallback from bbox centre
    if bbox and img_shape:
        ih, iw = img_shape[:2]
        cx = (bbox[0] + bbox[2]) / 2 / iw
        cy = (bbox[1] + bbox[3]) / 2 / ih
        for threshold, label in [
            (0.20, "Head / Skull"),
            (0.35, "Shoulder"),
            (0.50, "Elbow / Forearm"),
            (0.65, "Wrist / Hand"),
            (0.75, "Hip / Pelvis"),
            (0.88, "Knee"),
        ]:
            if cy < threshold:
                return label, None
        return "Ankle / Foot", None
 
    return "Not identified", None
 
 
def preprocess_xray(img):
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.addWeighted(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), 0.7, img, 0.3, 0)
 
 
def visual_morphology(roi, body_region=None):
    """
    Classify fracture type from crack geometry when API returns no result.
    Uses Hough line analysis + fragment counting + edge density scoring.
    """
    if roi is None or roi.size == 0:
        return "Transverse", 65.0
 
    gray     = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    edges    = cv2.Canny(enhanced, 30, 100)
    lines    = cv2.HoughLinesP(edges, 1, np.pi/180, 15, minLineLength=10, maxLineGap=8)
    h, w     = roi.shape[:2]
 
    edge_density    = np.sum(edges > 0) / (h * w + 1)
    mean_brightness = float(np.mean(gray))
    std_brightness  = float(np.std(gray))
    aspect_ratio    = w / (h + 1e-9)
    _, binary       = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fragment_count  = cv2.connectedComponentsWithStats(binary)[0] - 1
 
    angles, lengths = [], []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            angles.append(abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
            lengths.append(float(np.hypot(x2 - x1, y2 - y1)))
 
    n   = len(angles)
    ma  = float(np.mean(angles))  if angles else 45.0
    std = float(np.std(angles))   if angles else 0.0
    tl  = sum(lengths)            if lengths else 0.0
    ml  = max(lengths)            if lengths else 0.0
 
    s = {k: 0.0 for k in FRACTURE_TYPE_INFO}
 
    # Hairline — very low edge density, few lines
    if edge_density < 0.05 and n < 5:    s["Hairline"]   += 0.7
    if 0 < ml < 20 and edge_density<0.08: s["Hairline"]   += 0.3
 
    if n > 0:
        # Transverse — mostly horizontal lines
        hz = sum(1 for a in angles if a < 25 or a > 155)
        if hz / n > 0.5 or ma < 30 or ma > 150:
            s["Transverse"] += 0.7 if hz/n > 0.5 else 0.3
 
        # Oblique — diagonal lines
        ob = sum(1 for a in angles if 30 < a < 60 or 120 < a < 150)
        if ob / n > 0.4 or (35 < ma < 65) or (115 < ma < 145):
            s["Oblique"] += 0.6 if ob/n > 0.4 else 0.4
 
    # Spiral — high angle variance
    if std > 25 and n > 4:   s["Spiral"] += 0.5
    if tl > 100 and std > 20: s["Spiral"] += 0.4
 
    # Comminuted — many fragments, high edge density
    if fragment_count > 8 and edge_density > 0.12: s["Comminuted"] += 0.7
    if n > 8 and edge_density > 0.10:              s["Comminuted"] += 0.4
 
    # Greenstick — incomplete, low density
    if fragment_count < 4 and edge_density < 0.08 and aspect_ratio > 1.5:
        s["Greenstick"] += 0.5
 
    # Avulsion — small detached fragment
    if 2 < fragment_count < 6 and edge_density < 0.10: s["Avulsion"] += 0.4
 
    # Impacted — high brightness, low std
    if mean_brightness > 160 and std_brightness < 40: s["Impacted"] += 0.5
 
    # Compression — wide, flat
    if aspect_ratio > 2.0 and mean_brightness > 140: s["Compression"] += 0.5
 
    # Segmental — two break zones
    if n > 5 and fragment_count > 5: s["Segmental"] += 0.4
 
    # Longitudinal — near-vertical lines
    if n > 0 and ma > 60 and ma < 90: s["Longitudinal"] += 0.4
 
    # Penalise Comminuted for hand/wrist (many small bones look like fragments)
    is_hand = body_region and any(
        k in (body_region or "").lower()
        for k in ["hand", "finger", "wrist"]
    )
    if is_hand:
        s["Comminuted"] *= 0.3
        s["Segmental"]  *= 0.4
        s["Hairline"]   += 0.3
        s["Avulsion"]   += 0.2
 
    best  = max(s, key=s.get)
    conf  = round(
        min(max(0.55 + (s[best] / max(sum(s.values()), 1)) * 0.37, 0.55), 0.85) * 100, 1
    )
    print(f"[Visual] {best} {conf}%")
    return best, conf
 
 
# ── Main detector ──────────────────────────────────────────────────────────
class FractureDetector:
    def __init__(self, model_path: str, type_model_path: str = None,
                 body_part_model_path: str = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = YOLO(model_path)
        print(f"✅ Detection model loaded: {model_path}")
 
        # Model 2 — fracture type classifier (optional)
        self.type_model = None
        if type_model_path:
            if os.path.exists(type_model_path):
                self.type_model = YOLO(type_model_path)
                print(f"✅ Type classifier loaded: {type_model_path}")
            else:
                print(f"⚠ Type classifier not found at {type_model_path} — falling back to API + visual morphology")
 
        # Model 3 — body part classifier (optional)
        # Classification model (yolov8s-cls) — reads via .probs, not .boxes,
        # since it labels the whole image rather than localising anything.
        self.body_part_model = None
        if body_part_model_path:
            if os.path.exists(body_part_model_path):
                self.body_part_model = YOLO(body_part_model_path)
                print(f"✅ Body-part classifier loaded: {body_part_model_path}")
            else:
                print(f"⚠ Body-part classifier not found at {body_part_model_path} — falling back to API + positional guess")
 
    def detect(self, image_bytes: bytes, conf_threshold: float = 0.25):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
 
        img    = preprocess_xray(img)
        result = self.model.predict(
            img, conf=conf_threshold, iou=0.55,
            agnostic_nms=True, augment=True, verbose=False
        )[0]
        annotated  = result.plot()
        detections = []
        ih, iw     = img.shape[:2]
 
        if len(result.boxes) == 0:
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
            return {
                "success": True, "no_fracture": True,
                "total_detected": 0, "detections": [],
                "highest_severity": "None",
                "annotated_image": f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}",
            }
 
        # ── Get fracture TYPE once per image ──────────────────────────────
        # Priority: Model 2 (local) → Roboflow API → visual morphology
        ftype, type_conf, cls_source = None, None, None
 
        if self.type_model:
            try:
                type_result = self.type_model.predict(
                    img, conf=0.10, iou=0.50, agnostic_nms=True,
                    augment=True, verbose=False
                )[0]
                if len(type_result.boxes) > 0:
                    # Pick the highest-confidence box from the type model
                    best_idx  = int(type_result.boxes.conf.argmax())
                    raw_name  = type_result.names[int(type_result.boxes.cls[best_idx])]
                    canonical = _normalise_type(raw_name)
                    if canonical:
                        ftype      = canonical
                        type_conf  = round(float(type_result.boxes.conf[best_idx]) * 100, 1)
                        cls_source = "local-type-model"
                        print(f"[Model2] {ftype} {type_conf}%")
            except Exception as e:
                print(f"[Model2 error] {e}")
 
        # Fallback to Roboflow API cascade if Model 2 gave no result
        if not ftype:
            ftype, type_conf, cls_source = get_fracture_type(image_bytes)
 
        # ── Get BODY PART once per image ────────────────────────────────────
        # Priority: Model 3 (local classifier) → Roboflow API → positional guess
        region, part_conf = None, None
 
        if self.body_part_model:
            try:
                bp_result = self.body_part_model.predict(img, verbose=False)[0]
                top1_idx  = int(bp_result.probs.top1)
                region    = bp_result.names[top1_idx]
                part_conf = round(float(bp_result.probs.top1conf) * 100, 1)
                print(f"[Model3] {region} {part_conf}%")
            except Exception as e:
                print(f"[Model3 error] {e}")
 
        # Fallback to Roboflow API + positional guess if Model 3 gave no result
        if not region:
            areas        = [(int(b.xyxy[0][2])-int(b.xyxy[0][0])) *
                            (int(b.xyxy[0][3])-int(b.xyxy[0][1]))
                            for b in result.boxes]
            best_box     = result.boxes[int(np.argmax(areas))]
            primary_bbox = list(map(int, best_box.xyxy[0].tolist()))
            region, part_conf = get_body_part(
                image_bytes, bbox=primary_bbox, img_shape=img.shape
            )
 
        # ── Build detection entries ────────────────────────────────────────
        for box in result.boxes:
            conf         = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
 
            # If API returned a type use it; else run visual morphology on ROI
            if ftype:
                det_type   = ftype
                det_conf   = type_conf
                det_source = cls_source
            else:
                x1p = max(0, x1-30);  y1p = max(0, y1-30)
                x2p = min(iw, x2+30); y2p = min(ih, y2+30)
                roi = img[y1p:y2p, x1p:x2p]
                det_type, det_conf = visual_morphology(roi, body_region=region)
                det_source = "visual-morphology"
 
            info = FRACTURE_TYPE_INFO.get(det_type, {
                "severity": "Medium", "color": "#f97316",
                "description": "Fracture detected.",
                "advice": "Consult an orthopedic specialist immediately.",
            })
 
            detections.append({
                "class":                 det_type,
                "confidence":            round(conf * 100, 2),
                "type_confidence":       det_conf,
                "classification_source": det_source,
                "body_part":             region,
                "body_confidence":       part_conf,
                "bbox":                  [x1, y1, x2, y2],
                "severity":              info["severity"],
                "color":                 info["color"],
                "description":           info["description"],
                "advice":                info["advice"],
            })
 
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return {
            "success":          True,
            "no_fracture":      False,
            "total_detected":   len(detections),
            "detections":       detections,
            "highest_severity": next(
                (l for l in ("High", "Medium", "Low")
                 if any(d["severity"] == l for d in detections)), "None"
            ),
            "annotated_image": f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}",
        }
 