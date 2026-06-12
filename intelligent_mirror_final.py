import cv2
import mediapipe as mp
import math
import threading
import time
import textwrap
import os

# ── MediaPipe solution handles ─────────────────────────────────────────────────
mp_face_mesh      = mp.solutions.face_mesh     # 468-point face landmark model
mp_pose           = mp.solutions.pose          # 33-point body pose model
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

PHI = 1.618  # Golden ratio — used as ideal reference in face split display

# ── Face landmark index map (MediaPipe Face Mesh indices) ──────────────────────
# Readable names → numbered landmark indices so code stays self-documenting
LM = {
    "forehead":10,"chin":152,"left_cheek":234,"right_cheek":454,
    "left_eye_out":133,"right_eye_out":362,"left_eye_in":173,"right_eye_in":398,
    "nose_tip":2,"upper_lip":13,"left_nose":129,"right_nose":358,
    "left_mouth":61,"right_mouth":291,
}

# ── Left-right mirror pairs used for symmetry scoring ─────────────────────────
# Each tuple is (left_landmark_index, right_landmark_index)
SYMMETRY_PAIRS = [(133,362),(173,398),(129,358),(61,291),(234,454),(70,300),(105,334)]

# ── Body pose landmark indices (MediaPipe Pose) ────────────────────────────────
POSE_LM = {"left_shoulder":11,"right_shoulder":12,"left_ear":7,"right_ear":8,"left_hip":23,"right_hip":24}

# ── Ordered landmark index sequences for drawing white face outlines ───────────
# Each sub-list is drawn as a connected polyline (consecutive pairs joined by lines)
FACE_CONTOURS = [
    [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10],  # face oval
    [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7,33],                                                                                  # left eye
    [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382,362],                                                                              # right eye
    [70,63,105,66,107,55,65,52,53,46],                                                                                                                  # left eyebrow
    [300,293,334,296,336,285,295,282,283,276],                                                                                                          # right eyebrow
    [168,6,197,195,5,4],                                                                                                                                # nose bridge
    [294,278,344,440,275,4,45,220,115,48,64,98,294],                                                                                                   # nose bottom
    [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,61],                                                                       # outer lips
    [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95,78],                                                                       # inner lips
]

# ── LLM config ─────────────────────────────────────────────────────────────────
MODEL_PATH    = "./gemma-2-2b-it-Q4_K_M.gguf"  # path to downloaded .gguf model
LLM_AVAILABLE = False   # set to True only after model loads successfully
llm           = None    # holds the loaded Llama model instance

def load_llm():
    """Load the offline LLM model at startup in a background thread."""
    global llm, LLM_AVAILABLE
    try:
        from llama_cpp import Llama
        if not os.path.exists(MODEL_PATH):
            print(f"[LLM] Model not found at {MODEL_PATH}"); return
        print("[LLM] Loading...")
        # n_ctx=512: small context window keeps inference fast
        # n_threads=4: use 4 CPU cores; verbose=False: suppress internal logs
        llm = Llama(model_path=MODEL_PATH, n_ctx=512, n_threads=4, verbose=False)
        LLM_AVAILABLE = True
        print("[LLM] Ready.")
    except ImportError:
        print("[LLM] llama-cpp-python not installed.")

def build_prompt(bq, posture_score, gender, worst_ratio, posture_issue):
    """Build the instruction prompt sent to the LLM with current scan results."""
    return (f"You are a personal appearance and wellness coach.\n"
            f"A {gender} user scanned themselves in an intelligent mirror.\n"
            f"BQ: {bq:.0f}/100, Weakest facial proportion: {worst_ratio}, "
            f"Posture: {posture_score:.0f}/100, Issue: {posture_issue}\n"
            f"Give exactly 3 short actionable tips. Tips 1 and 2 about face/grooming "
            f"related to {worst_ratio}. Tip 3 about fixing: {posture_issue}. "
            f"One sentence each. Number them 1, 2, 3. No scores.")

class LLMWorker:
    """Runs LLM inference in a background thread so the camera never freezes."""
    def __init__(self):
        self.suggestion   = ["Press S to get AI suggestions"]
        self.is_thinking  = False
        self._last_prompt = None  # caches last prompt to skip duplicate requests

    def request(self, bq, posture_score, gender, worst_ratio, posture_issue):
        """Queue a tip generation — ignored if already running or same inputs."""
        if self.is_thinking or not LLM_AVAILABLE: return
        prompt = build_prompt(bq, posture_score, gender, worst_ratio, posture_issue)
        if prompt == self._last_prompt: return  # nothing changed, skip
        self._last_prompt = prompt
        # daemon=True: thread auto-kills when main program exits
        threading.Thread(target=self._run, args=(prompt,), daemon=True).start()

    def _run(self, prompt):
        """Background thread: calls LLM, parses numbered tips from raw output."""
        self.is_thinking = True
        self.suggestion  = ["Generating AI tips..."]
        try:
            # stop=["4."]: halt generation after tip 3 so it doesn't keep going
            raw   = llm(prompt, max_tokens=180, temperature=0.7, stop=["4.","\n\n\n"])["choices"][0]["text"].strip()
            lines = [l.strip() for l in raw.split("\n") if l.strip()]
            # keep only lines starting with a digit or dash (numbered tips)
            tips  = [l for l in lines if l and (l[0].isdigit() or l.startswith("-"))]
            self.suggestion = tips if tips else [raw]  # fallback: show raw text
        except Exception as e:
            self.suggestion = [f"LLM error: {e}"]
        finally:
            self.is_thinking = False  # always reset even if exception occurred

# ── Math helpers ───────────────────────────────────────────────────────────────

def dist(p1, p2):
    """Euclidean distance between two MediaPipe landmarks (normalized 0–1 coords)."""
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)

def angle_deg(a, b, c):
    """Angle at point B formed by A-B-C, in degrees. Uses dot product formula."""
    ax,ay = a.x-b.x, a.y-b.y
    cx,cy = c.x-b.x, c.y-b.y
    dot   = ax*cx + ay*cy
    mag   = math.sqrt(ax**2+ay**2) * math.sqrt(cx**2+cy**2)
    # clamp to [-1,1] to avoid floating point error crashing acos
    return math.degrees(math.acos(max(-1.0, min(1.0, dot/mag)))) if mag else 0.0

def horiz_angle(p1, p2):
    """Angle of line p1→p2 relative to horizontal axis, in degrees (-180 to +180)."""
    return math.degrees(math.atan2(-(p2.y-p1.y), p2.x-p1.x))

def tilt(p1, p2):
    """
    How many degrees the line p1→p2 deviates from horizontal.
    Always returns 0–90 regardless of direction.
    Fixes the 177.9° = 'perfectly flat' bug from raw horiz_angle.
    """
    raw = abs(horiz_angle(p1, p2)) % 180
    return 180 - raw if raw > 90 else raw

def mkpt(x, y):
    """Create a lightweight point object with .x and .y attributes."""
    return type('P',(),{'x':x,'y':y})()

# ── Stage 2: BQ Scoring ────────────────────────────────────────────────────────

def compute_symmetry(landmarks):
    """
    Measures left-right facial balance across 7 landmark pairs.
    Each pair is compared against the face midline (avg of forehead + chin x).
    Returns a score 0–100 (100 = perfect symmetry).
    """
    lm  = landmarks.landmark
    mid = (lm[LM["forehead"]].x + lm[LM["chin"]].x) / 2
    # deviation = how unequally each pair sits around the midline, normalised
    devs = [abs(abs(lm[li].x-mid) - abs(lm[ri].x-mid)) / max(abs(lm[li].x-mid), abs(lm[ri].x-mid), 1e-6)
            for li,ri in SYMMETRY_PAIRS]
    return round(max(0, 100*(1 - sum(devs)/len(devs)*4)), 1)

def compute_face_split(landmarks):
    """
    Splits the face vertically at the midline.
    Measures left_half (midline → left cheek) and right_half (midline → right cheek).
    ratio = left/right: ideal = 1.000 (equal halves), shown alongside PHI for context.
    Returns a dict with measurements and a split_score 0–100.
    """
    lm  = landmarks.landmark
    mid = (lm[LM["forehead"]].x + lm[LM["chin"]].x) / 2
    lh  = abs(mid - lm[LM["left_cheek"]].x)
    rh  = abs(lm[LM["right_cheek"]].x - mid)
    rat = lh / (rh + 1e-6)  # +1e-6 prevents division by zero
    return {
        "left_half":round(lh,4), "right_half":round(rh,4), "ratio":round(rat,3),
        "split_score":round(max(0,100*(1-abs(rat-1.0)*5)),1),  # 20% deviation → 0
        "midline_x":mid, "forehead_y":lm[LM["forehead"]].y, "chin_y":lm[LM["chin"]].y,
    }

def compute_bq(landmarks, gender="female"):
    """
    Final BQ = 60% symmetry + 40% face split score.
    Returns (bq, split_score, sym_score, split_dict).
    """
    sym   = compute_symmetry(landmarks)
    split = compute_face_split(landmarks)
    bq    = round(0.6*sym + 0.4*split["split_score"], 1)
    return bq, split["split_score"], sym, split

# ── Stage 3: Posture Analysis ──────────────────────────────────────────────────

def compute_posture(pose_lm):
    """
    Analyses 4 posture metrics from MediaPipe Pose keypoints.
    Uses midpoints of left/right pairs so single-side dominance doesn't skew results.
    Thresholds are tightened vs real-world expectations — 'Good' requires near-perfect alignment.
    Returns a dict with per-metric tuples (value, status, tip) and overall posture_score.
    """
    lm    = pose_lm.landmark
    ls,rs = lm[POSE_LM["left_shoulder"]], lm[POSE_LM["right_shoulder"]]
    le,re = lm[POSE_LM["left_ear"]],      lm[POSE_LM["right_ear"]]
    lh,rh = lm[POSE_LM["left_hip"]],      lm[POSE_LM["right_hip"]]
    # midpoints reduce sensitivity to individual landmark noise
    ms = mkpt((ls.x+rs.x)/2,(ls.y+rs.y)/2)  # mid-shoulder
    me = mkpt((le.x+re.x)/2,(le.y+re.y)/2)  # mid-ear
    mh = mkpt((lh.x+rh.x)/2,(lh.y+rh.y)/2) # mid-hip

    issues, scores = [], []

    # 1. Shoulder tilt — angle of shoulder line vs horizontal
    sh_t = tilt(ls,rs)
    if sh_t < 2:   sh_s,sh_tip="Good","Shoulders level"; scores.append(100)
    elif sh_t < 5: sh_s,sh_tip="Slight tilt",f"Shoulder tilt {sh_t:.1f}° — relax shoulders equally"; scores.append(70); issues.append(sh_tip)
    else:          sh_s,sh_tip="Uneven!",f"Shoulder tilt {sh_t:.1f}° — one shoulder raised"; scores.append(30); issues.append(sh_tip)

    # 2. Head tilt — angle of ear-to-ear line vs horizontal
    ht = tilt(le,re)
    if ht < 2:   ht_s,ht_tip="Good","Head straight"; scores.append(100)
    elif ht < 6: ht_s,ht_tip="Slight tilt",f"Head tilted {ht:.1f}° — straighten neck"; scores.append(65); issues.append(ht_tip)
    else:        ht_s,ht_tip="Tilted!",f"Head tilted {ht:.1f}° — significant lean"; scores.append(25); issues.append(ht_tip)

    # 3. Forward head — ear x-offset vs shoulder x-position
    # Ideal: ear directly above shoulder → offset near 0
    eo = me.x - ms.x
    if abs(eo) < 0.015:   fh_s,fh_tip="Good","Head position neutral"; scores.append(100)
    elif abs(eo) < 0.045: fh_s,fh_tip="Slight forward","Head slightly forward — pull chin back"; scores.append(60); issues.append(fh_tip)
    else:                 fh_s,fh_tip="Forward head!","Head far forward — tuck chin"; scores.append(20); issues.append(fh_tip)

    # 4. Spine angle — ear → shoulder → hip (ideal: ~180° straight line)
    na = angle_deg(me,ms,mh)
    if na > 168:   na_s,na_tip="Good","Spine aligned"; scores.append(100)
    elif na > 150: na_s,na_tip="Slight lean",f"Spine {na:.0f}° — stand more upright"; scores.append(65); issues.append(na_tip)
    else:          na_s,na_tip="Hunched!",f"Spine {na:.0f}° — significant hunch"; scores.append(20); issues.append(na_tip)

    return {
        "shoulder_tilt":(sh_t,sh_s,sh_tip), "head_tilt":(ht,ht_s,ht_tip),
        "forward_head":(eo,fh_s,fh_tip),    "neck_angle":(na,na_s,na_tip),
        "posture_score":round(sum(scores)/len(scores),1),
        "top_issue":issues[0] if issues else "Posture looks great!",
    }

# ── HUD drawing helpers ────────────────────────────────────────────────────────

def sc(score):
    """Score → BGR color: green ≥75, yellow ≥50, red below."""
    return (80,220,100) if score>=75 else ((200,200,60) if score>=50 else (80,100,220))

def stc(status):
    """Status string → BGR color: green=good, cyan=slight, red=bad."""
    s = status.lower()
    return (80,220,100) if "good" in s else ((60,200,220) if "slight" in s else (80,100,220))

def bar(frame, x, y, w, h, score, col):
    """Draw a filled progress bar: dark background → colored fill → grey border."""
    cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,50),-1)              # background
    cv2.rectangle(frame,(x,y),(x+int(w*score/100),y+h),col,-1)      # fill proportional to score
    cv2.rectangle(frame,(x,y),(x+w,y+h),(110,110,110),1)            # border

def txt(frame, text, pos, scale, col, thick=1):
    """Shorthand for cv2.putText with FONT_HERSHEY_SIMPLEX."""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick)

def draw_face_hud(frame, bq, ss, sym, split, gender):
    """Draw the face analysis panel (top-left): BQ, symmetry, split ratio, L/R bar, tip."""
    # Semi-transparent dark background via alpha blend
    ov = frame.copy()
    cv2.rectangle(ov,(0,0),(480,430),(12,12,12),-1)
    cv2.addWeighted(ov,0.68,frame,0.32,0,frame)

    txt(frame,"INTELLIGENT MIRROR",(16,30),0.75,(150,150,150))
    txt(frame,f"Mode:{gender.upper()}  G=gender S=tips Q=quit",(16,54),0.48,(110,110,110))

    # BQ score — large number + progress bar
    bc = sc(bq)
    txt(frame,f"BQ  {bq:.1f} / 100",(16,108),1.5,bc,2)
    bar(frame,16,120,444,13,bq,bc)

    # Sub-scores with individual bars
    txt(frame,f"Symmetry   {sym:.1f}/100",(16,162),0.65,(175,175,175))
    bar(frame,260,153,184,9,sym,sc(sym))
    txt(frame,f"Face Split  {ss:.1f}/100",(16,190),0.65,(175,175,175))
    bar(frame,260,181,184,9,ss,sc(ss))
    cv2.line(frame,(16,208),(462,208),(55,55,55),1)

    # Face split ratio — large display
    rc = sc(split["split_score"])
    txt(frame,"FACE SPLIT RATIO",(16,238),0.55,(130,130,130))
    txt(frame,f"{split['left_half']:.3f}  :  {split['right_half']:.3f}",(16,278),1.1,rc,2)  # left : right
    txt(frame,f"Ratio  {split['ratio']:.3f}",(16,316),1.0,rc,2)
    txt(frame,f"Ideal  1.000   |   Golden Ratio  {PHI:.3f}",(16,340),0.44,(120,120,120))

    # Visual split bar — blue=left half, orange=right half, white center tick
    lh,rh = split["left_half"], split["right_half"]
    bw,bx,by,bh2 = 444,16,356,16
    lf = int(bw*lh/(lh+rh+1e-6))  # left fill width proportional to left_half
    cv2.rectangle(frame,(bx,by),(bx+lf,by+bh2),(200,100,60),-1)         # left (orange)
    cv2.rectangle(frame,(bx+lf,by),(bx+bw,by+bh2),(60,100,200),-1)      # right (blue)
    cv2.rectangle(frame,(bx,by),(bx+bw,by+bh2),(110,110,110),1)         # border
    cv2.line(frame,(bx+bw//2,by),(bx+bw//2,by+bh2),(255,255,255),1)     # center reference tick
    txt(frame,"L",(bx+4,by+bh2-3),0.38,(255,255,255))
    txt(frame,"R",(bx+bw-16,by+bh2-3),0.38,(255,255,255))

    # Contextual tip based on current scores
    tip = ("Tip: face camera straight on" if sym<60
           else "Face split is well balanced!" if split["split_score"]>=80
           else f"Tip: {'left' if split['ratio']>1 else 'right'} half wider ({split['ratio']:.3f} vs 1.000)")
    txt(frame,tip,(16,420),0.50,(130,195,255))

def draw_posture_hud(frame, posture, fh):
    """Draw the posture panel (bottom-left): score, 4 metrics with values, top tip."""
    ph,py = 200, fh-200  # panel height and top-y (anchored to frame bottom)
    ov = frame.copy()
    cv2.rectangle(ov,(0,py),(480,fh),(10,10,20),-1)
    cv2.addWeighted(ov,0.70,frame,0.30,0,frame)

    ps = posture["posture_score"]
    txt(frame,"POSTURE ANALYSIS",(16,py+24),0.70,(150,150,150))
    txt(frame,f"{ps:.0f}/100",(300,py+24),0.80,sc(ps),2)
    bar(frame,16,py+32,444,8,ps,sc(ps))
    cv2.line(frame,(16,py+46),(462,py+46),(55,55,55),1)

    # Each metric row: label, status (color-coded), measured value
    metrics = [
        ("Shoulders",*posture["shoulder_tilt"][:2],f"{posture['shoulder_tilt'][0]:.1f}°"),
        ("Head Tilt",*posture["head_tilt"][:2],    f"{posture['head_tilt'][0]:.1f}°"),
        ("Head Fwd", *posture["forward_head"][:2], f"{abs(posture['forward_head'][0]):.3f}"),
        ("Spine",    *posture["neck_angle"][:2],   f"{posture['neck_angle'][0]:.0f}°"),
    ]
    y = py+68
    for label,_,status,value in metrics:
        txt(frame,f"{label:<12} {status:<14} {value}",(16,y),0.52,stc(status))
        y += 26

    cv2.line(frame,(16,py+158),(462,py+158),(55,55,55),1)
    txt(frame,posture["top_issue"],(16,py+180),0.50,(130,195,255))

def draw_llm_panel(frame, worker, fw, fh):
    """
    Draw the AI tips panel (top-right). Panel height grows dynamically to fit all
    wrapped tip text — posture tip is always fully visible, never clipped.
    """
    pw,px,py      = 360, fw-370, 10
    max_c,lh,pad  = 40, 22, 12    # max chars per line, line height, padding
    tips = worker.suggestion

    # Pre-calculate total height needed before drawing anything
    def wh(t): return len(textwrap.wrap(t,width=max_c))*lh+6  # height of one wrapped tip
    ch  = 66 + 20 + sum(wh(t) for t in tips[:2])  # header + FACE label + 2 face tips
    ch += 36 + 20                                   # divider + POSTURE label
    ch += wh(tips[2]) if len(tips)>2 else lh        # posture tip or placeholder
    ch += 28                                         # footer
    ph  = max(ch+pad*2, 200)                         # minimum 200px tall

    # Semi-transparent dark green background + border
    ov = frame.copy()
    cv2.rectangle(ov,(px,py),(px+pw,py+ph),(8,16,8),-1)
    cv2.addWeighted(ov,0.78,frame,0.22,0,frame)
    cv2.rectangle(frame,(px,py),(px+pw,py+ph),(40,100,40),1)

    lbl = "AI TIPS  (thinking...)" if worker.is_thinking else "AI TIPS"
    txt(frame,lbl,(px+pad,py+26),0.65,(60,160,220) if worker.is_thinking else (100,220,100))
    cv2.line(frame,(px+pad,py+36),(px+pw-pad,py+36),(40,80,40),1)

    y = py+56
    txt(frame,"FACE",(px+pad,y),0.42,(100,180,255)); y+=20
    for tip in tips[:2]:  # first 2 tips are face/grooming
        for line in textwrap.wrap(tip,width=max_c):
            txt(frame,line,(px+pad,y),0.46,(190,230,190)); y+=lh
        y+=6  # gap between tips

    # Divider + posture section
    y+=4; cv2.line(frame,(px+pad,y),(px+pw-pad,y),(40,80,40),1); y+=16
    txt(frame,"POSTURE",(px+pad,y),0.42,(100,220,160)); y+=20
    if len(tips)>2:
        for line in textwrap.wrap(tips[2],width=max_c):
            txt(frame,line,(px+pad,y),0.46,(190,230,190)); y+=lh
    else:
        # LLM sometimes returns fewer than 3 tips — show fallback message
        txt(frame,"No posture tip yet — press S again",(px+pad,y),0.42,(120,120,120)); y+=lh

    y+=8
    footer = "Press S to refresh" if (not worker.is_thinking and LLM_AVAILABLE) else "Model not loaded — see terminal"
    txt(frame,footer,(px+pad,y),0.38,(80,130,80) if LLM_AVAILABLE else (80,80,180))

# ── EMA Smoother ───────────────────────────────────────────────────────────────

class EMA:
    """
    Exponential Moving Average — smooths a value over time.
    new_display = alpha * raw + (1-alpha) * previous
    Lower alpha = more stable but slower to react to changes.
    """
    def __init__(self, a=0.08): self.a=a; self.v=None
    def update(self, x):
        self.v = x if self.v is None else self.a*x+(1-self.a)*self.v  # seed on first frame
        return round(self.v,1)

# ── Snapshot Engine ────────────────────────────────────────────────────────────

SAMPLE_FRAMES = 10   # collect this many frames before locking best result
LOCK_SECONDS  = 10   # hold the locked result for this many seconds

class SnapshotEngine:
    """
    Manages the sample → lock → resample cycle.
    SAMPLING: collects frames, picks the one with the highest BQ.
    LOCKED:   freezes that result on screen for LOCK_SECONDS, then re-samples.
    """
    def __init__(self):
        self.state="SAMPLING"; self.candidates=[]; self.locked=None
        self.lock_t=None; self.n=0

    def feed(self, *args):
        """Feed one frame's results. Triggers lock when SAMPLE_FRAMES collected."""
        if self.state=="SAMPLING":
            self.candidates.append(args); self.n+=1
            if self.n>=SAMPLE_FRAMES:
                # Pick best frame = highest BQ (args[0])
                self.locked=max(self.candidates,key=lambda x:x[0])
                self.lock_t=time.time(); self.state="LOCKED"
                self.candidates=[]; self.n=0
                print(f"[Mirror] Locked BQ={self.locked[0]:.1f}")
        elif time.time()-self.lock_t>=LOCK_SECONDS:
            # Lock expired — start collecting again
            self.state="SAMPLING"; self.candidates=[]; self.n=0
            print("[Mirror] Re-sampling...")

    def data(self):
        """Returns locked result tuple if in LOCKED state, else None."""
        return self.locked if self.state=="LOCKED" else None

    def secs_left(self):
        """Seconds remaining in current lock period."""
        return max(0,LOCK_SECONDS-(time.time()-self.lock_t)) if self.state=="LOCKED" else 0

    def progress(self):
        """Sampling progress 0.0 → 1.0 for the progress bar."""
        return self.n/SAMPLE_FRAMES

# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    gender,show_llm = "female",False

    # Start LLM loading immediately in background — camera opens while model loads
    threading.Thread(target=load_llm,daemon=True).start()
    worker = LLMWorker()
    snap   = SnapshotEngine()

    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened(): print("[ERROR] No webcam"); return

    # Open both models as context managers so they release cleanly on exit
    with mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,
            min_detection_confidence=0.5,min_tracking_confidence=0.5) as fm, \
         mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5,
            model_complexity=1) as pose:  # model_complexity 1 = balanced speed/accuracy

        last_lm = None  # cache last valid face landmarks for boundary drawing
        worst   = "N/A" # tracks worst face metric for LLM prompt

        while cap.isOpened():
            ret,frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame,1)                          # mirror flip — feels natural
            rgb   = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)      # MediaPipe expects RGB, OpenCV gives BGR
            fh,fw = frame.shape[:2]

            # Run both models on the same RGB frame each iteration
            fr = fm.process(rgb)    # face mesh results
            pr = pose.process(rgb)  # pose results

            raw_bq=raw_r=raw_s=raw_split=raw_post=None  # reset each frame

            if fr.multi_face_landmarks:
                last_lm = fr.multi_face_landmarks[0]  # cache for boundary drawing
                raw_bq,raw_r,raw_s,raw_split = compute_bq(last_lm,gender)
            if pr.pose_landmarks:
                raw_post = compute_posture(pr.pose_landmarks)
            if raw_bq and raw_post:
                snap.feed(raw_bq,raw_r,raw_s,raw_split,raw_post)  # feed only when both available

            display = snap.data()  # None if still sampling, tuple if locked

            # ── Draw white face boundaries on live frame every frame ──────────
            if last_lm:
                lm = last_lm.landmark
                def px(i): return (int(lm[i].x*fw),int(lm[i].y*fh))  # normalized → pixel coords
                for contour in FACE_CONTOURS:
                    for i in range(len(contour)-1):
                        cv2.line(frame,px(contour[i]),px(contour[i+1]),(255,255,255),1)

                # Dashed vertical split line — drawn live, follows face movement
                mid_x = int((lm[LM["forehead"]].x+lm[LM["chin"]].x)/2*fw)
                ty    = int(lm[LM["forehead"]].y*fh)-10
                by2   = int(lm[LM["chin"]].y*fh)+10
                yc = ty
                while yc<by2:
                    # draw 8px segment, skip 8px → dashed effect
                    cv2.line(frame,(mid_x,yc),(mid_x,min(yc+8,by2)),(255,255,255),1)
                    yc+=16

            # ── Display locked snapshot scores in HUD ─────────────────────────
            if display:
                db,dr,ds,dsp,dp = display
                # Build worst-ratio string for LLM prompt
                worst = (f"face split {dsp['ratio']:.3f} "
                         f"(L{dsp['left_half']:.3f} R{dsp['right_half']:.3f})")
                draw_face_hud(frame,db,dr,ds,dsp,gender)
                draw_posture_hud(frame,dp,fh)

                # If face lost during locked state, overlay a warning
                if raw_bq is None:
                    ov=frame.copy()
                    cv2.rectangle(ov,(0,56),(420,96),(40,0,0),-1)
                    cv2.addWeighted(ov,0.75,frame,0.25,0,frame)
                    txt(frame,"No face detected — centre yourself",(12,84),0.55,(80,80,240))

                # Green countdown bar at bottom — shows time until next resample
                sl = snap.secs_left()
                bw = fw-32
                cv2.rectangle(frame,(16,fh-14),(16+bw,fh-6),(40,40,40),-1)
                cv2.rectangle(frame,(16,fh-14),(16+int(bw*sl/LOCK_SECONDS),fh-6),(60,180,80),-1)
                txt(frame,f"Next scan in {sl:.0f}s",(16,fh-18),0.42,(120,120,120))

            else:
                # ── Sampling state — show collection progress ─────────────────
                prog = snap.progress()
                bw   = fw-32
                ov=frame.copy()
                cv2.rectangle(ov,(0,0),(420,52),(12,12,12),-1)
                cv2.addWeighted(ov,0.68,frame,0.32,0,frame)
                msg = (f"Analysing... ({snap.n}/{SAMPLE_FRAMES} frames)"
                       if raw_bq else "No face detected — centre yourself")
                txt(frame,msg,(12,32),0.55,(60,200,220) if raw_bq else (80,80,240))

                # Blue progress bar — fills as frames are collected
                cv2.rectangle(frame,(16,fh-14),(16+bw,fh-6),(40,40,40),-1)
                cv2.rectangle(frame,(16,fh-14),(16+int(bw*prog),fh-6),(60,160,220),-1)
                txt(frame,"Sampling best frame...",(16,fh-18),0.42,(120,120,120))

            if show_llm: draw_llm_panel(frame,worker,fw,fh)  # AI panel — only after S pressed

            cv2.imshow("Intelligent Mirror  [Q=quit G=gender S=tips]",frame)
            key = cv2.waitKey(1)&0xFF  # 1ms wait per frame; &0xFF for cross-platform safety
            if key==ord('q'): break
            elif key==ord('g'):
                gender="male" if gender=="female" else "female"
                snap.__init__()  # reset snapshot so it re-samples with new gender targets
                print(f"[Mirror] Gender→{gender}")
            elif key==ord('s'):
                if display:
                    show_llm=True
                    db,_,_,_,dp=display
                    worker.request(db,dp["posture_score"],gender,worst,dp["top_issue"])
                else: print("[Mirror] Wait for snapshot lock first.")

    # Always release webcam and close window — prevents camera staying locked
    cap.release(); cv2.destroyAllWindows(); print("[Mirror] Closed.")

if __name__=="__main__": main()