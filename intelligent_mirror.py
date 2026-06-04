import cv2
import mediapipe as mp
import math
import threading
import time
import textwrap
import os

# ─── MediaPipe setup ───────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_pose      = mp.solutions.pose
mp_drawing   = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

PHI = 1.618

# ─── Gender-aware ideal ratio targets ─────────────────────────────────────────
IDEAL_RATIOS = {
    "female": {
        "Face H/W":   1.618,
        "Width/Eye":  1.618,
        "H/NoseLip":  2.90,
        "Nose/Mouth": 0.618,
        "Eye/Face W": 0.382,
    },
    "male": {
        "Face H/W":   1.45,
        "Width/Eye":  1.50,
        "H/NoseLip":  2.60,
        "Nose/Mouth": 0.680,
        "Eye/Face W": 0.360,
    }
}

# ─── Face landmark indices ─────────────────────────────────────────────────────
LM = {
    "forehead":      10,
    "chin":          152,
    "left_cheek":    234,
    "right_cheek":   454,
    "left_eye_out":  133,
    "right_eye_out": 362,
    "left_eye_in":   173,
    "right_eye_in":  398,
    "nose_tip":      2,
    "upper_lip":     13,
    "left_nose":     129,
    "right_nose":    358,
    "left_mouth":    61,
    "right_mouth":   291,
}

SYMMETRY_PAIRS = [
    (133, 362), (173, 398), (129, 358),
    (61,  291), (234, 454), (70,  300), (105, 334),
]

POSE_LM = {
    "nose":           0,
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_ear":       7,
    "right_ear":      8,
    "left_hip":       23,
    "right_hip":      24,
}


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 4 — OFFLINE LLM (llama-cpp-python)
#  Install: pip3 install llama-cpp-python
#  Download a model .gguf file and set MODEL_PATH below.
#
#  Recommended free models (download once from HuggingFace):
#    • gemma-2-2b-it-Q4_K_M.gguf   (~1.5 GB, fast)
#    • Llama-3.2-3B-Instruct-Q4_K_M.gguf (~2 GB, better quality)
#
#  Example download (run once in terminal):
#    pip3 install huggingface_hub
#    python3 -c "from huggingface_hub import hf_hub_download; \
#      hf_hub_download('bartowski/gemma-2-2b-it-GGUF', \
#      'gemma-2-2b-it-Q4_K_M.gguf', local_dir='.')"
# ══════════════════════════════════════════════════════════════════════════════

# ── Set your model path here ──────────────────────────────────────────────────
MODEL_PATH = "./gemma-2-2b-it-Q4_K_M.gguf"   # change to your downloaded model

LLM_AVAILABLE = False
llm = None

def load_llm():
    """Load the local LLM model. Called once at startup in a background thread."""
    global llm, LLM_AVAILABLE
    try:
        from llama_cpp import Llama
        if not os.path.exists(MODEL_PATH):
            print(f"[LLM] Model not found at: {MODEL_PATH}")
            print("[LLM] Running without AI suggestions.")
            print("[LLM] See comments in the file to download a model.")
            return
        print(f"[LLM] Loading model: {MODEL_PATH} ...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=512,          # context window — keep small for speed
            n_threads=4,        # CPU threads
            verbose=False,
        )
        LLM_AVAILABLE = True
        print("[LLM] Model loaded. AI suggestions enabled.")
    except ImportError:
        print("[LLM] llama-cpp-python not installed.")
        print("[LLM] Run: pip3 install llama-cpp-python")
        print("[LLM] Continuing without AI suggestions.")


def build_prompt(bq, posture_score, gender, worst_ratio, posture_issue):
    """Build a short, focused prompt for the local LLM."""
    return f"""You are a helpful personal appearance and wellness coach.
A {gender} user just scanned themselves in an intelligent mirror.

Results:
- Beauty Quotient (BQ): {bq:.0f}/100
- Weakest facial proportion: {worst_ratio}
- Posture Score: {posture_score:.0f}/100
- Main posture issue: {posture_issue}

Give exactly 3 short, practical, actionable tips.
Tip 1 and Tip 2 must be about improving facial appearance or grooming related to {worst_ratio}.
Tip 3 must be about fixing the posture issue: {posture_issue}.
Each tip is one sentence. Number them 1, 2, 3.
Be encouraging and specific. Do not repeat the scores back."""


class LLMWorker:
    """
    Runs LLM inference in a background thread so the camera never freezes.
    Call request() to queue a new suggestion.
    Read .suggestion for the latest result.
    """
    def __init__(self):
        self.suggestion   = ["Press S to get AI suggestions"]
        self.is_thinking  = False
        self._thread      = None
        self._last_prompt = None

    def request(self, bq, posture_score, gender, worst_ratio, posture_issue):
        """Start a background LLM call (ignored if one is already running)."""
        if self.is_thinking or not LLM_AVAILABLE:
            return
        prompt = build_prompt(bq, posture_score, gender, worst_ratio, posture_issue)
        if prompt == self._last_prompt:
            return   # same inputs — no need to regenerate
        self._last_prompt = prompt
        self._thread = threading.Thread(
            target=self._run, args=(prompt,), daemon=True
        )
        self._thread.start()

    def _run(self, prompt):
        self.is_thinking = True
        self.suggestion  = ["Generating AI tips..."]
        try:
            output = llm(
                prompt,
                max_tokens=180,
                temperature=0.7,
                stop=["4.", "\n\n\n"],   # stop after 3 tips
            )
            raw = output["choices"][0]["text"].strip()
            # Split into individual lines, filter blanks
            lines = [l.strip() for l in raw.split("\n") if l.strip()]
            # Keep only lines that look like numbered tips
            tips = [l for l in lines if l and (l[0].isdigit() or l.startswith("-"))]
            self.suggestion = tips if tips else [raw]
        except Exception as e:
            self.suggestion = [f"LLM error: {e}"]
        finally:
            self.is_thinking = False


# ══════════════════════════════════════════════════════════════════════════════
#  MATH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def dist(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def angle_deg(a, b, c):
    ax, ay = a.x - b.x, a.y - b.y
    cx, cy = c.x - b.x, c.y - b.y
    dot    = ax * cx + ay * cy
    mag_a  = math.sqrt(ax**2 + ay**2)
    mag_c  = math.sqrt(cx**2 + cy**2)
    if mag_a * mag_c == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag_a * mag_c)))))


def horizontal_angle_deg(p1, p2):
    dx, dy = p2.x - p1.x, p2.y - p1.y
    return math.degrees(math.atan2(-dy, dx))


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — BQ SCORING
# ══════════════════════════════════════════════════════════════════════════════

def compute_symmetry(landmarks):
    lm = landmarks.landmark
    midline_x = (lm[LM["forehead"]].x + lm[LM["chin"]].x) / 2
    deviations = []
    for (li, ri) in SYMMETRY_PAIRS:
        ld = abs(lm[li].x - midline_x)
        rd = abs(lm[ri].x - midline_x)
        larger = max(ld, rd, 1e-6)
        deviations.append(abs(ld - rd) / larger)
    return round(max(0, 100 * (1 - sum(deviations) / len(deviations) * 4)), 1)


def compute_ratios(landmarks, gender="female"):
    lm = landmarks.landmark
    targets = IDEAL_RATIOS[gender]
    face_h   = dist(lm[LM["forehead"]],     lm[LM["chin"]])
    face_w   = dist(lm[LM["left_cheek"]],   lm[LM["right_cheek"]])
    eye_span = dist(lm[LM["left_eye_out"]], lm[LM["right_eye_out"]])
    nose_w   = dist(lm[LM["left_nose"]],    lm[LM["right_nose"]])
    mouth_w  = dist(lm[LM["left_mouth"]],   lm[LM["right_mouth"]])
    nose_lip = dist(lm[LM["nose_tip"]],     lm[LM["upper_lip"]])
    actuals = {
        "Face H/W":   face_h   / (face_w   + 1e-6),
        "Width/Eye":  face_w   / (eye_span + 1e-6),
        "H/NoseLip":  face_h   / (nose_lip + 1e-6),
        "Nose/Mouth": nose_w   / (mouth_w  + 1e-6),
        "Eye/Face W": eye_span / (face_w   + 1e-6),
    }
    results, scores = {}, []
    for name, actual in actuals.items():
        ideal = targets[name]
        score = round(max(0, 100 * (1 - abs(actual - ideal) / ideal * 2)), 1)
        results[name] = (actual, ideal, score)
        scores.append(score)
    return round(sum(scores) / len(scores), 1), results


def compute_bq(landmarks, gender="female"):
    ratio_score, ratio_details = compute_ratios(landmarks, gender)
    sym_score = compute_symmetry(landmarks)
    bq = round(0.6 * ratio_score + 0.4 * sym_score, 1)
    return bq, ratio_score, sym_score, ratio_details


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — POSTURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def make_point(x, y):
    return type('P', (), {'x': x, 'y': y})()


def compute_posture(pose_landmarks):
    lm = pose_landmarks.landmark
    ls   = lm[POSE_LM["left_shoulder"]]
    rs   = lm[POSE_LM["right_shoulder"]]
    le   = lm[POSE_LM["left_ear"]]
    re   = lm[POSE_LM["right_ear"]]
    lh   = lm[POSE_LM["left_hip"]]
    rh   = lm[POSE_LM["right_hip"]]

    mid_shoulder = make_point((ls.x+rs.x)/2, (ls.y+rs.y)/2)
    mid_ear      = make_point((le.x+re.x)/2, (le.y+re.y)/2)
    mid_hip      = make_point((lh.x+rh.x)/2, (lh.y+rh.y)/2)

    issues, scores = [], []

    sh_tilt = abs(horizontal_angle_deg(ls, rs))
    if sh_tilt < 3:
        sh_status, sh_tip = "Good", "Shoulders level"
        scores.append(100)
    elif sh_tilt < 7:
        sh_status, sh_tip = "Slight tilt", f"Shoulder tilt {sh_tilt:.1f}° — relax both shoulders equally"
        scores.append(70); issues.append(sh_tip)
    else:
        sh_status, sh_tip = "Uneven!", f"Shoulder tilt {sh_tilt:.1f}° — one shoulder raised, correct it"
        scores.append(30); issues.append(sh_tip)

    head_tilt = abs(horizontal_angle_deg(le, re))
    if head_tilt < 4:
        ht_status, ht_tip = "Good", "Head straight"
        scores.append(100)
    elif head_tilt < 10:
        ht_status, ht_tip = "Slight tilt", f"Head tilted {head_tilt:.1f}° — straighten your neck"
        scores.append(65); issues.append(ht_tip)
    else:
        ht_status, ht_tip = "Tilted!", f"Head tilted {head_tilt:.1f}° — significant lean detected"
        scores.append(25); issues.append(ht_tip)

    ear_offset = mid_shoulder.x - mid_ear.x
    if abs(ear_offset) < 0.03:
        fh_status, fh_tip = "Good", "Head position neutral"
        scores.append(100)
    elif abs(ear_offset) < 0.07:
        fh_status, fh_tip = "Slight forward", "Head slightly forward — pull chin back"
        scores.append(60); issues.append(fh_tip)
    else:
        fh_status, fh_tip = "Forward head!", "Head far forward — tuck chin, align ear over shoulder"
        scores.append(20); issues.append(fh_tip)

    neck_ang = angle_deg(mid_ear, mid_shoulder, mid_hip)
    if neck_ang > 160:
        na_status, na_tip = "Good", "Spine aligned"
        scores.append(100)
    elif neck_ang > 140:
        na_status, na_tip = "Slight lean", f"Spine angle {neck_ang:.0f}° — sit/stand more upright"
        scores.append(65); issues.append(na_tip)
    else:
        na_status, na_tip = "Hunched!", f"Spine angle {neck_ang:.0f}° — you are hunching significantly"
        scores.append(20); issues.append(na_tip)

    return {
        "shoulder_tilt": (sh_tilt,    sh_status, sh_tip),
        "head_tilt":     (head_tilt,  ht_status, ht_tip),
        "forward_head":  (ear_offset, fh_status, fh_tip),
        "neck_angle":    (neck_ang,   na_status, na_tip),
        "posture_score": round(sum(scores) / len(scores), 1),
        "top_issue":     issues[0] if issues else "Posture looks great!",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  HUD HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def score_color(score):
    if score >= 75:   return (80, 220, 100)
    elif score >= 50: return (200, 200, 60)
    else:             return (80, 100, 220)


def status_color(status):
    s = status.lower()
    if "good" in s:   return (80, 220, 100)
    if "slight" in s: return (60, 200, 220)
    return (80, 100, 220)


def draw_bar(frame, x, y, w, h, score, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
    fill = int(w * score / 100)
    cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (110, 110, 110), 1)


def get_face_tip(ratio_details, sym_score):
    if sym_score < 60:
        return "Tip: face camera straight on for best accuracy"
    worst_name, (actual, ideal, score) = min(
        ratio_details.items(), key=lambda x: x[1][2]
    )
    if score >= 80:
        return "Face proportions look great!"
    direction = "above" if actual > ideal else "below"
    return f"Tip: {worst_name} is {direction} ideal  ({actual:.2f} vs {ideal:.2f})"


# ─── Face HUD ─────────────────────────────────────────────────────────────────
def draw_face_hud(frame, bq, ratio_score, sym_score, ratio_details, gender):
    panel_w, panel_h = 480, 420
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    cv2.putText(frame, "INTELLIGENT MIRROR", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150, 150, 150), 1)
    cv2.putText(frame, f"Mode: {gender.upper()}   |   G=gender  S=AI tips  Q=quit",
                (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (110, 110, 110), 1)

    bq_color = score_color(bq)
    cv2.putText(frame, f"BQ  {bq:.1f} / 100", (16, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, bq_color, 2)
    draw_bar(frame, 16, 120, 444, 13, bq, bq_color)

    cv2.putText(frame, f"Ratio     {ratio_score:.1f}/100", (16, 162),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (175, 175, 175), 1)
    draw_bar(frame, 260, 153, 184, 9, ratio_score, score_color(ratio_score))

    cv2.putText(frame, f"Symmetry  {sym_score:.1f}/100", (16, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (175, 175, 175), 1)
    draw_bar(frame, 260, 181, 184, 9, sym_score, score_color(sym_score))

    cv2.line(frame, (16, 208), (462, 208), (55, 55, 55), 1)

    y = 232
    for name, (actual, ideal, score) in ratio_details.items():
        col = score_color(score)
        cv2.putText(frame, f"{name:<12}  {actual:.3f} / {ideal:.3f}", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (155, 155, 155), 1)
        draw_bar(frame, 370, y - 13, 90, 8, score, col)
        y += 28

    tip = get_face_tip(ratio_details, sym_score)
    cv2.putText(frame, tip, (16, panel_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (130, 195, 255), 1)


# ─── Posture HUD ──────────────────────────────────────────────────────────────
def draw_posture_hud(frame, posture, frame_h):
    panel_w, panel_h = 480, 200
    panel_y = frame_h - panel_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_y), (panel_w, frame_h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    cv2.putText(frame, "POSTURE ANALYSIS", (16, panel_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (150, 150, 150), 1)
    ps = posture["posture_score"]
    ps_col = score_color(ps)
    cv2.putText(frame, f"{ps:.0f}/100", (300, panel_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, ps_col, 2)
    draw_bar(frame, 16, panel_y + 32, 444, 8, ps, ps_col)
    cv2.line(frame, (16, panel_y + 46), (462, panel_y + 46), (55, 55, 55), 1)

    metrics = [
        ("Shoulders", posture["shoulder_tilt"][1]),
        ("Head Tilt",  posture["head_tilt"][1]),
        ("Head Fwd",   posture["forward_head"][1]),
        ("Spine",      posture["neck_angle"][1]),
    ]
    y = panel_y + 68
    for label, status in metrics:
        cv2.putText(frame, f"{label:<12} {status}", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color(status), 1)
        y += 26

    cv2.line(frame, (16, panel_y + 158), (462, panel_y + 158), (55, 55, 55), 1)
    cv2.putText(frame, posture["top_issue"], (16, panel_y + 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (130, 195, 255), 1)


# ─── AI Suggestions HUD (right-side panel) ────────────────────────────────────
def draw_llm_panel(frame, worker, frame_w, frame_h):
    """
    Draws the AI suggestions panel on the RIGHT side of the frame.
    Only visible after pressing S.
    """
    panel_w = 340
    panel_h = 320
    panel_x = frame_w - panel_w - 10
    panel_y = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (8, 16, 8), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (40, 100, 40), 1)

    # Header
    label = "AI TIPS  (thinking...)" if worker.is_thinking else "AI TIPS"
    cv2.putText(frame, label, (panel_x + 12, panel_y + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (100, 220, 100) if not worker.is_thinking else (60, 160, 220), 1)
    cv2.line(frame, (panel_x + 12, panel_y + 36),
             (panel_x + panel_w - 12, panel_y + 36), (40, 80, 40), 1)

    # Tips — first 2 are face, last 1 is posture
    y = panel_y + 58
    max_chars = 38
    for i, tip in enumerate(worker.suggestion):
        # Section label
        if i == 0:
            cv2.putText(frame, "FACE", (panel_x + 12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 180, 255), 1)
            y += 18
        elif i == 2:
            cv2.line(frame, (panel_x + 12, y - 4),
                     (panel_x + panel_w - 12, y - 4), (40, 80, 40), 1)
            cv2.putText(frame, "POSTURE", (panel_x + 12, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 220, 160), 1)
            y += 26

        wrapped = textwrap.wrap(tip, width=max_chars)
        for line in wrapped:
            if y > panel_y + panel_h - 20:
                break
            cv2.putText(frame, line, (panel_x + 12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (190, 230, 190), 1)
            y += 22
        y += 6

    # Footer
    if not worker.is_thinking and LLM_AVAILABLE:
        cv2.putText(frame, "Press S to refresh", (panel_x + 12, panel_y + panel_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 130, 80), 1)
    elif not LLM_AVAILABLE:
        cv2.putText(frame, "Model not loaded — see terminal",
                    (panel_x + 12, panel_y + panel_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 180), 1)


# ══════════════════════════════════════════════════════════════════════════════
#  EMA SMOOTHER
# ══════════════════════════════════════════════════════════════════════════════

class EMA:
    def __init__(self, alpha=0.08):
        self.alpha = alpha
        self.value = None

    def update(self, new_val):
        if self.value is None:
            self.value = new_val
        else:
            self.value = self.alpha * new_val + (1 - self.alpha) * self.value
        return round(self.value, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    gender       = "female"
    show_llm     = False   # toggle with S key
    last_bq      = 0.0
    last_posture = None

    # Load LLM in background so startup isn't blocked
    threading.Thread(target=load_llm, daemon=True).start()
    llm_worker = LLMWorker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    # EMA smoothers
    ema_bq      = EMA(alpha=0.08)
    ema_ratio   = EMA(alpha=0.08)
    ema_sym     = EMA(alpha=0.08)
    ema_ratios  = {}
    ema_posture = EMA(alpha=0.08)

    print("[Mirror] Starting...")
    print("  Q = quit   G = toggle gender   S = get AI suggestions")
    print("  Make sure your shoulders are visible for posture detection.")

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh, mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_h, frame_w = frame.shape[:2]

            face_results = face_mesh.process(rgb)
            pose_results = pose.process(rgb)

            # ── Face BQ ──────────────────────────────────────────────────────
            if face_results.multi_face_landmarks:
                for face_lm in face_results.multi_face_landmarks:
                    bq, ratio_score, sym_score, ratio_details = compute_bq(
                        face_lm, gender
                    )
                    s_bq    = ema_bq.update(bq)
                    s_ratio = ema_ratio.update(ratio_score)
                    s_sym   = ema_sym.update(sym_score)

                    s_ratio_details = {}
                    worst_ratio_name = "N/A"
                    worst_ratio_score = 101
                    for name, (actual, ideal, score) in ratio_details.items():
                        if name not in ema_ratios:
                            ema_ratios[name] = EMA(alpha=0.08)
                        s_score = ema_ratios[name].update(score)
                        s_ratio_details[name] = (actual, ideal, s_score)
                        if s_score < worst_ratio_score:
                            worst_ratio_score = s_score
                            worst_ratio_name  = name

                    last_bq = s_bq
                    draw_face_hud(frame, s_bq, s_ratio, s_sym,
                                  s_ratio_details, gender)

                    # ── White facial boundaries ───────────────────────────
                    h_f, w_f = frame.shape[:2]
                    lm = face_lm.landmark

                    def lm_px(idx):
                        """Convert normalized landmark to pixel coords."""
                        return (int(lm[idx].x * w_f), int(lm[idx].y * h_f))

                    WHITE = (255, 255, 255)
                    T = 1   # line thickness

                    # ── Face oval ─────────────────────────────────────────
                    OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,
                            397,365,379,378,400,377,152,148,176,149,150,136,
                            172,58,132,93,234,127,162,21,54,103,67,109,10]
                    for i in range(len(OVAL) - 1):
                        cv2.line(frame, lm_px(OVAL[i]), lm_px(OVAL[i+1]), WHITE, T)

                    # ── Left eye ──────────────────────────────────────────
                    L_EYE = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7,33]
                    for i in range(len(L_EYE) - 1):
                        cv2.line(frame, lm_px(L_EYE[i]), lm_px(L_EYE[i+1]), WHITE, T)

                    # ── Right eye ─────────────────────────────────────────
                    R_EYE = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382,362]
                    for i in range(len(R_EYE) - 1):
                        cv2.line(frame, lm_px(R_EYE[i]), lm_px(R_EYE[i+1]), WHITE, T)

                    # ── Left eyebrow ──────────────────────────────────────
                    L_BROW = [70,63,105,66,107,55,65,52,53,46]
                    for i in range(len(L_BROW) - 1):
                        cv2.line(frame, lm_px(L_BROW[i]), lm_px(L_BROW[i+1]), WHITE, T)

                    # ── Right eyebrow ─────────────────────────────────────
                    R_BROW = [300,293,334,296,336,285,295,282,283,276]
                    for i in range(len(R_BROW) - 1):
                        cv2.line(frame, lm_px(R_BROW[i]), lm_px(R_BROW[i+1]), WHITE, T)

                    # ── Nose bridge + tip ─────────────────────────────────
                    NOSE_BRIDGE = [168,6,197,195,5,4]
                    for i in range(len(NOSE_BRIDGE) - 1):
                        cv2.line(frame, lm_px(NOSE_BRIDGE[i]), lm_px(NOSE_BRIDGE[i+1]), WHITE, T)

                    NOSE_BOTTOM = [294,278,344,440,275,4,45,220,115,48,64,98,294]
                    for i in range(len(NOSE_BOTTOM) - 1):
                        cv2.line(frame, lm_px(NOSE_BOTTOM[i]), lm_px(NOSE_BOTTOM[i+1]), WHITE, T)

                    # ── Outer lips ────────────────────────────────────────
                    LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291,
                                  409,270,269,267,0,37,39,40,185,61]
                    for i in range(len(LIPS_OUTER) - 1):
                        cv2.line(frame, lm_px(LIPS_OUTER[i]), lm_px(LIPS_OUTER[i+1]), WHITE, T)

                    # ── Inner lips ────────────────────────────────────────
                    LIPS_INNER = [78,191,80,81,82,13,312,311,310,415,308,
                                  324,318,402,317,14,87,178,88,95,78]
                    for i in range(len(LIPS_INNER) - 1):
                        cv2.line(frame, lm_px(LIPS_INNER[i]), lm_px(LIPS_INNER[i+1]), WHITE, T)
            else:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (340, 52), (12, 12, 12), -1)
                cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
                cv2.putText(frame, "No face detected — centre yourself",
                            (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (80, 80, 240), 1)
                worst_ratio_name = "N/A"

            # ── Posture ───────────────────────────────────────────────────────
            if pose_results.pose_landmarks:
                posture = compute_posture(pose_results.pose_landmarks)
                posture["posture_score"] = ema_posture.update(
                    posture["posture_score"]
                )
                last_posture = posture

            if last_posture:
                draw_posture_hud(frame, last_posture, frame_h)
            else:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, frame_h - 40),
                              (420, frame_h), (10, 10, 20), -1)
                cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
                cv2.putText(frame,
                            "Posture: step back so shoulders are visible",
                            (12, frame_h - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 80, 240), 1)

            # ── AI Suggestions panel ──────────────────────────────────────────
            if show_llm:
                draw_llm_panel(frame, llm_worker, frame_w, frame_h)

            cv2.imshow("Intelligent Mirror  [Q=quit  G=gender  S=AI tips]", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                gender = "male" if gender == "female" else "female"
                print(f"[Mirror] Gender → {gender}")
            elif key == ord('s'):
                show_llm = True
                posture_issue = (last_posture["top_issue"]
                                 if last_posture else "unknown")
                posture_score = (last_posture["posture_score"]
                                 if last_posture else 0)
                llm_worker.request(
                    bq=last_bq,
                    posture_score=posture_score,
                    gender=gender,
                    worst_ratio=worst_ratio_name,
                    posture_issue=posture_issue,
                )

    cap.release()
    cv2.destroyAllWindows()
    print("[Mirror] Closed.")


if __name__ == "__main__":
    main()
