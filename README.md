# 🪞 Intelligent Mirror

A fully offline AI-powered desktop application that uses your webcam to analyze **facial symmetry**, **face split ratio**, and **posture** in real time — then generates personalized improvement tips using a local LLM. No internet required after setup.

---

## ✨ Features

- **Beauty Quotient (BQ) Score** — Weighted combination of symmetry (60%) and face split score (40%), displayed 0–100
- **Face Split Ratio** — Splits the face vertically at the midline, measures left vs right half widths, compares against ideal 1.000
- **Live Split Line** — Dashed white vertical line drawn on the face that follows movement in real time
- **Symmetry Analysis** — Measures left-right facial balance across 7 landmark pairs
- **White Facial Boundaries** — Outlines drawn over eyes, eyebrows, nose, lips, and face oval using MediaPipe landmarks
- **Snapshot System** — Collects 10 frames, locks the best (highest BQ) result for 30 seconds, then re-samples — zero flickering
- **Posture Detection** — Analyses shoulder tilt, head tilt, forward head posture, and spine angle with tightened real-world thresholds
- **Offline AI Tips** — Press `S` to generate 3 personalized tips (2 face + 1 posture) from a local LLM — zero internet required
- **Dynamic AI Panel** — Panel height grows automatically to fit all tip text — posture tip always fully visible

---

## 🖥️ Layout

```
┌──────────────────────────────────────────────────────────────────┐
│ INTELLIGENT MIRROR                          ┌─────────────────┐ │
│ Mode: FEMALE  |  G=gender  S=tips  Q=quit   │ AI TIPS         │ │
│                                             │ ─────────────── │ │
│ BQ  67.3 / 100  ████████████░░░░░░          │ FACE            │ │
│ Symmetry   82.1/100  ████████░░             │ 1. tip text...  │ │
│ Face Split 71.4/100  ███████░░░             │ 2. tip text...  │ │
│ ────────────────────────────────────        │ ─────────────── │ │
│ FACE SPLIT RATIO                            │ POSTURE         │ │
│ 0.312  :  0.298                             │ 3. tip text...  │ │
│ Ratio  1.047                                │ Press S refresh │ │
│ Ideal  1.000  |  Golden Ratio  1.618        └─────────────────┘ │
│ [L ████████████████|████████████ R]                             │
│ Tip: left half wider — face camera squarely                     │
│                                                                  │
│         [live webcam feed + white face boundaries]              │
│         [dashed vertical split line follows live face]          │
│                                                                  │
│ POSTURE ANALYSIS                              72/100             │
│ ████████████████░░░░                                            │
│ Shoulders   Slight tilt   3.2°                                  │
│ Head Tilt   Good          1.1°                                  │
│ Head Fwd    Slight fwd    0.022                                  │
│ Spine       Slight lean   154°                                   │
│ ──────────────────────────────────────────────────────          │
│ Shoulder tilt 3.2° — relax both shoulders equally               │
│ ─────────────────── Next scan in 24s ───────────────────────────│
└──────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3 |
| Webcam + Drawing | OpenCV (`cv2`) |
| Face Landmarks (468 pts) | MediaPipe Face Mesh |
| Body Pose (33 pts) | MediaPipe Pose |
| Geometry Math | Python `math` (built-in) |
| Snapshot Engine | Custom class (sample → lock → resample) |
| Background Threading | Python `threading` (built-in) |
| Text Wrapping | Python `textwrap` (built-in) |
| Offline LLM | `llama-cpp-python` + Gemma 2B GGUF |

---

## ⚙️ Installation

### 1. Clone or download the project

```bash
git clone https://github.com/yourusername/intelligent-mirror
cd intelligent-mirror
```

### 2. Install Python dependencies

```bash
pip3 install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3) Mac users:** If `mediapipe` fails, run:
> ```bash
> pip3 uninstall mediapipe -y
> pip3 install mediapipe==0.10.9
> ```

### 3. Download the AI model (one-time, ~1.5 GB)

The model is not in the repository (exceeds GitHub's 100 MB limit). Download it once:

```bash
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('bartowski/gemma-2-2b-it-GGUF', 'gemma-2-2b-it-Q4_K_M.gguf', local_dir='.')
"
```

> The app works without the model — AI tips will be disabled until it is present.

### 4. Run

```bash
python3 intelligent_mirror_final.py
```

---

## 🎮 Controls

| Key | Action |
|---|---|
| `G` | Toggle Male / Female mode — resets snapshot and re-samples immediately |
| `S` | Generate AI tips (2 face + 1 posture) — only works when snapshot is locked |
| `Q` | Quit |

---

## 📐 How BQ Score is Calculated

```
BQ = (0.60 × Symmetry Score) + (0.40 × Face Split Score)
```

### Symmetry Score (60%)

Seven left-right landmark pairs compared against the face midline:

| Pair | Landmarks used |
|---|---|
| Eye outer corners | 133, 362 |
| Eye inner corners | 173, 398 |
| Nose wings | 129, 358 |
| Mouth corners | 61, 291 |
| Cheeks | 234, 454 |
| Eyebrow outer | 70, 300 |
| Eyebrow inner | 105, 334 |

Formula: `max(0, 100 × (1 - avg_deviation × 4))`

### Face Split Score (40%)

The face is divided vertically at the midline (average of forehead and chin x-position):

```
left_half  = distance from midline → left cheek
right_half = distance from midline → right cheek
ratio      = left_half / right_half
```

- Ideal ratio = **1.000** (perfectly equal halves)
- Golden ratio = **1.618** (shown as reference only)
- Score formula: `max(0, 100 × (1 - |ratio - 1.0| × 5))`

The dashed white vertical line on the face shows exactly where this split is measured, and updates live every frame.

---

## 🧍 Posture Analysis

Four metrics computed from MediaPipe Pose keypoints. All angles are normalized to 0–90° to avoid the raw `atan2` range causing false readings.

| Metric | Method | Good | Slight | Bad |
|---|---|---|---|---|
| Shoulder tilt | Angle of shoulder line vs horizontal | < 2° | 2–5° | > 5° |
| Head tilt | Angle of ear line vs horizontal | < 2° | 2–6° | > 6° |
| Forward head | Ear x-offset vs shoulder x | < 0.015 | 0.015–0.045 | > 0.045 |
| Spine angle | Ear → shoulder → hip angle | > 168° | 150–168° | < 150° |

Each metric shows its actual measured value in the HUD (e.g. `Shoulders   Slight tilt   3.2°`).

> **Tip:** Step back from the camera so both shoulders are visible in frame.

---

## 📸 Snapshot System

Instead of displaying raw per-frame scores (which flicker), the app runs a **sample → lock → resample** cycle:

```
SAMPLING (10 frames) → picks highest BQ → LOCKED (30 seconds) → repeat
```

- **SAMPLING**: shows `Analysing... (3/10 frames)` with a blue progress bar at the bottom
- **LOCKED**: scores frozen, green countdown bar shows `Next scan in 24s`
- Pressing `G` resets the snapshot and re-samples immediately with the new gender targets
- `S` for AI tips only fires during the LOCKED state

To adjust timing, change these two constants near the top of the snapshot section:

```python
SAMPLE_FRAMES = 10    # frames to collect before locking
LOCK_SECONDS  = 30    # seconds to hold the locked result
```

---

## 🤖 Offline LLM

The AI tip generator uses `llama-cpp-python` to run a quantized GGUF model fully on-device.

- **Model:** `gemma-2-2b-it-Q4_K_M.gguf` (~1.5 GB)
- **RAM required:** ~3–4 GB
- **Speed:** ~5–15 seconds per generation on CPU (faster on Apple Silicon with Metal)
- **Internet:** Only needed once to download the model file

The LLM runs in a **background thread** — camera feed never freezes during generation.

**What it receives:** BQ score, face split ratio, posture score, top posture issue

**What it returns:**
- Tips 1 & 2 — face appearance / grooming advice
- Tip 3 — specific posture correction

To use a different model, update this line in the file:

```python
MODEL_PATH = "./gemma-2-2b-it-Q4_K_M.gguf"
```

**Alternative models:**

| Model | Size | Quality |
|---|---|---|
| `gemma-2-2b-it-Q4_K_M.gguf` | ~1.5 GB | Fast, good — recommended |
| `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | ~2 GB | Better quality |

---

## 📁 Project Structure

```
intelligent-mirror/
│
├── intelligent_mirror_final.py    # Main application (all stages integrated)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Excludes model and cache from Git
├── README.md                      # This file
│
└── gemma-2-2b-it-Q4_K_M.gguf     # AI model — download separately, NOT in Git
```

### What gets pushed to Git

| File | Push? | Reason |
|---|---|---|
| `intelligent_mirror_final.py` | ✅ Yes | Main source code |
| `requirements.txt` | ✅ Yes | Dependency list |
| `.gitignore` | ✅ Yes | Keeps repo clean |
| `README.md` | ✅ Yes | Documentation |
| `*.gguf` model file | ❌ No | ~1.5 GB — exceeds GitHub's 100 MB limit |
| `.cache/huggingface/` | ❌ No | Auto-generated download cache |

---

## 🗂️ Git Setup

```bash
# Create .gitignore
echo "*.gguf
*.bin
*.safetensors
.cache/
__pycache__/
*.pyc
.DS_Store" > .gitignore

# Push to GitHub
git init
git add intelligent_mirror_final.py requirements.txt .gitignore README.md
git commit -m "Initial commit: Intelligent Mirror"
git remote add origin https://github.com/yourusername/intelligent-mirror.git
git push -u origin main
```

---

## 🔧 Troubleshooting

**`module 'mediapipe' has no attribute 'solutions'`**
```bash
pip3 uninstall mediapipe -y
pip3 install mediapipe==0.10.9
```

**`zsh: command not found: python` or `pip`**
```bash
python3 intelligent_mirror_final.py
pip3 install -r requirements.txt
```

**`[LLM] Model not found`**
- Make sure the `.gguf` file is in the same folder as the Python script
- Or update `MODEL_PATH` in the script to the full absolute path

**Posture panel not showing**
- Step back so both shoulders are visible in the frame
- MediaPipe Pose needs to see at least your upper body

**Scores not updating**
- Wait for the 30-second green countdown to finish
- Or press `G` to force an immediate re-sample

**AI tips panel missing posture tip**
- Press `S` again — the LLM occasionally returns fewer than 3 tips
- The panel shows `"No posture tip yet — press S again"` as a fallback

---

## 📊 Score Color Guide

| Color | Range | Meaning |
|---|---|---|
| 🟢 Green | 75–100 | Great |
| 🟡 Yellow | 50–74 | Average |
| 🔴 Red | 0–49 | Needs improvement |

---

## 📝 Notes

- All processing is **100% local** — no data leaves your machine at any point
- The app fully works without the LLM model; AI tips are optional
- For best results: face the camera straight on, ensure good lighting, and position yourself so your full face and shoulders are visible in frame
- The split line and face boundaries draw on the **live feed** every frame — they always follow your current position even while scores are locked

---

## 👤 Author

Built as a professor evaluation project demonstrating real-time computer vision, facial landmark analysis, pose estimation, snapshot-based stable scoring, and offline LLM integration.

**Stack:** Python · OpenCV · MediaPipe · llama-cpp-python
