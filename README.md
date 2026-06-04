# 🪞 Intelligent Mirror

A fully offline AI-powered desktop application that uses your webcam to analyze **facial proportions**, **symmetry**, and **posture** in real time — then generates personalized improvement tips using a local LLM. No internet required after setup.

---

## ✨ Features

- **Beauty Quotient (BQ) Score** — Scores your facial proportions (0–100) by comparing 5 key ratios against golden ratio (φ = 1.618) ideals
- **Symmetry Analysis** — Measures left-right facial balance across 7 landmark pairs
- **Gender-Aware Scoring** — Separate ideal targets for male and female faces (toggle with `G`)
- **Posture Detection** — Detects shoulder tilt, head tilt, forward head posture, and spine alignment in real time
- **White Facial Boundaries** — Draws precise outlines over eyes, eyebrows, nose, lips, and face oval
- **EMA Smoothing** — Exponential Moving Average prevents score flickering from frame-to-frame jitter
- **Offline AI Tips** — Press `S` to generate 3 personalized tips (2 face + 1 posture) from a local LLM — zero internet required

---

## 🖥️ Demo Layout

```
┌─────────────────────────────────────────────────┐
│ INTELLIGENT MIRROR           [AI TIPS panel →]  │
│ Mode: FEMALE | G=gender S=AI tips Q=quit        │
│                                                  │
│ BQ  67.3 / 100  ████████████░░░░░░              │
│ Ratio     58.2/100  ████░░░░░                   │
│ Symmetry  82.1/100  ████████░░                  │
│ ─────────────────────────────────────────────── │
│ Face H/W    1.521 / 1.618  ░░░░░                │
│ Width/Eye   1.689 / 1.618  ██████               │
│ H/NoseLip   2.743 / 2.900  ████░░               │
│ Nose/Mouth  0.601 / 0.618  ███████              │
│ Eye/Face W  0.371 / 0.382  ████████             │
│ ─────────────────────────────────────────────── │
│ Tip: Face H/W is below ideal (1.52 vs 1.62)    │
│                                                  │
│                    [live webcam feed]            │
│                                                  │
│ POSTURE ANALYSIS                     74/100      │
│ ██████████████░░░░                              │
│ Shoulders   Good                                 │
│ Head Tilt   Slight tilt                         │
│ Head Fwd    Good                                 │
│ Spine       Good                                 │
│ ─────────────────────────────────────────────── │
│ Head tilted 6.2° — straighten your neck        │
└─────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3 |
| Webcam + Drawing | OpenCV (`cv2`) |
| Face Landmarks (468 pts) | MediaPipe Face Mesh |
| Body Pose (33 pts) | MediaPipe Pose |
| Golden Ratio Math | Python `math` (built-in) |
| Score Smoothing | Custom EMA class |
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

> **Note for Apple Silicon (M1/M2/M3) Mac users:** If `mediapipe` fails, try:
> ```bash
> pip3 uninstall mediapipe -y
> pip3 install mediapipe==0.10.9
> ```

### 3. Download the AI model (one-time, ~1.5 GB)

The model file is not included in the repository (too large for Git). Download it once:

```bash
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('bartowski/gemma-2-2b-it-GGUF', 'gemma-2-2b-it-Q4_K_M.gguf', local_dir='.')
"
```

> The model is saved in the project folder. The app works without it — AI tips will simply be disabled.

### 4. Run

```bash
python3 intelligent_mirror_stage4.py
```

---

## 🎮 Controls

| Key | Action |
|---|---|
| `G` | Toggle between Male / Female scoring mode |
| `S` | Generate AI tips (2 face + 1 posture) |
| `Q` | Quit |

---

## 📐 How BQ Score is Calculated

The BQ (Beauty Quotient) score is a weighted combination of two components:

```
BQ = (0.60 × Ratio Score) + (0.40 × Symmetry Score)
```

### Ratio Score (60%)
Five facial measurements are compared to gender-specific golden ratio ideals:

| Ratio | What it measures | Female ideal | Male ideal |
|---|---|---|---|
| Face H/W | Face height ÷ face width | 1.618 | 1.45 |
| Width/Eye | Face width ÷ eye span | 1.618 | 1.50 |
| H/NoseLip | Face height ÷ nose-to-lip | 2.90 | 2.60 |
| Nose/Mouth | Nose width ÷ mouth width | 0.618 | 0.680 |
| Eye/Face W | Eye span ÷ face width | 0.382 | 0.360 |

Each ratio is scored: `max(0, 100 × (1 - deviation × 2))`

### Symmetry Score (40%)
Seven left-right landmark pairs are measured against the face midline:
- Eye outer corners, eye inner corners
- Nose wings, mouth corners
- Cheeks, eyebrow outer, eyebrow inner

Score: `max(0, 100 × (1 - avg_deviation × 4))`

---

## 🧍 How Posture is Analysed

Four metrics are computed from MediaPipe Pose body keypoints:

| Metric | Method | Good threshold |
|---|---|---|
| Shoulder tilt | Angle of shoulder line vs horizontal | < 3° |
| Head tilt | Angle of ear line vs horizontal | < 4° |
| Forward head | Ear x-offset vs shoulder x-position | < 0.03 |
| Spine angle | Ear → shoulder → hip angle | > 160° |

> **Tip:** Step back from the camera so both shoulders are visible for posture detection to work.

---

## 🤖 Offline LLM (Stage 4)

The AI tip generator uses `llama-cpp-python` to run a quantized GGUF model locally.

- **Model used:** `gemma-2-2b-it-Q4_K_M.gguf` (~1.5 GB)
- **RAM required:** ~3–4 GB
- **Runs on:** CPU (with Metal GPU acceleration on Apple Silicon)
- **Internet required:** Only once to download the model

The LLM runs in a **background thread** so the camera feed never freezes. It receives your BQ score, posture score, weakest facial proportion, and top posture issue — and returns 3 actionable tips.

To use a different model, change this line in the file:
```python
MODEL_PATH = "./gemma-2-2b-it-Q4_K_M.gguf"
```

**Alternative models:**

| Model | Size | Quality |
|---|---|---|
| `gemma-2-2b-it-Q4_K_M.gguf` | ~1.5 GB | Fast, good |
| `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | ~2 GB | Better quality |

---

## 📁 Project Structure

```
intelligent-mirror/
│
├── intelligent_mirror_stage4.py   # Main application (all stages integrated)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Excludes model file and cache from Git
├── README.md                      # This file
│
└── gemma-2-2b-it-Q4_K_M.gguf     # AI model — download separately, NOT in Git
```

### What gets pushed to Git

| File | Push? | Reason |
|---|---|---|
| `intelligent_mirror_stage4.py` | ✅ Yes | Main source code |
| `requirements.txt` | ✅ Yes | Dependency list |
| `.gitignore` | ✅ Yes | Keeps repo clean |
| `README.md` | ✅ Yes | Documentation |
| `*.gguf` model file | ❌ No | 1.5 GB — exceeds GitHub's 100 MB limit |
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
git add intelligent_mirror_stage4.py requirements.txt .gitignore README.md
git commit -m "Initial commit: Intelligent Mirror all stages"
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
# Use python3 and pip3 instead
python3 intelligent_mirror_stage4.py
pip3 install opencv-python mediapipe
```

**`[LLM] Model not found`**
- Make sure the `.gguf` file is in the same folder as the Python script
- Or update `MODEL_PATH` in the script to the full path of your model file

**Posture panel not showing**
- Step back from the camera so both shoulders are visible in the frame
- MediaPipe Pose needs to see at least your upper body

**BQ score still fluctuating too much**
- Find `EMA(alpha=0.08)` in the file and lower alpha to `0.05`

---

## 📊 Score Color Guide

| Color | Range | Meaning |
|---|---|---|
| 🟢 Green | 75 – 100 | Great |
| 🟡 Yellow | 50 – 74 | Average |
| 🔴 Red | 0 – 49 | Needs improvement |

---

## 📝 Notes

- All processing happens **100% locally** — no data leaves your machine
- The app works without the LLM model; AI tips are optional
- For best results: face the camera straight on, ensure good lighting, and position yourself so your full face and shoulders are visible

---

## 👤 Author

Built as a professor evaluation project demonstrating real-time computer vision, facial landmark analysis, pose estimation, and offline LLM integration.

**Stack:** Python · OpenCV · MediaPipe · llama-cpp-python
