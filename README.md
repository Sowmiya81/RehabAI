# 🏋️ RehabAI - AI Movement Analysis Coach

<div align="center">

![RehabAI Banner](docs/images/ui_home.jpg)

**AI-powered squat analysis using Computer Vision, RAG, and LLM-based coaching**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-purple.svg)](https://ai.google.dev/)

</div>

---

## 🎯 What It Does

RehabAI analyzes your squat form and provides personalized coaching backed by research literature. Upload a video → Get instant biomechanics analysis → Receive evidence-based corrective exercises.

**Core Pipeline:**
1. **Computer Vision** (MediaPipe) detects pose and calculates joint angles
2. **RAG System** (ChromaDB + Sentence Transformers) retrieves relevant research
3. **LLM Agent** (Gemini 2.5 Flash + LangGraph) generates personalized coaching

---

## 🖼️ Screenshots

| Home Interface | Analysis Results (Good Form) |
|:--------------:|:----------------------------:|
| ![Home](docs/images/ui_home.jpg) | ![Good](docs/images/good_results.jpg) |

| Analysis Results (Issues Detected) | AI Coaching Plan |
|:----------------------------------:|:----------------:|
| ![Bad](docs/images/bad_results.jpg) | ![Coaching](docs/images/coaching_plan.jpg) |

---

## 🏗️ Architecture

┌─────────────────┐
│ Gradio Web UI │
└────────┬────────┘
│
┌────▼─────────────────────────┐
│ LangGraph Orchestrator │
│ (Agentic Reasoning Loop) │
└──┬──────────┬────────────┬───┘
│ │ │
┌───▼───┐ ┌──▼───┐ ┌────▼─────┐
│MediaPipe│ │ChromaDB│ │Gemini 2.5│
│ Pose │ │ RAG │ │ Flash │
└─────────┘ └────────┘ └──────────┘

text

**Tech Stack:**
- **CV**: MediaPipe Pose
- **RAG**: ChromaDB + Sentence Transformers (all-MiniLM-L6-v2)
- **LLM**: Google Gemini 2.5 Flash
- **Orchestration**: LangGraph
- **UI**: Gradio 4.0

---

## 🚀 Quick Start

### Installation

```bash
# Clone and setup
git clone https://github.com/yourusername/RehabAI.git
cd RehabAI
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Add API key
echo "GOOGLE_API_KEY=your_key_here" > .env
Run
bash
python app.py
Navigate to http://localhost:7860

Usage
Upload squat video (or record via webcam)

Click "Analyze Movement"

Review detected issues, ROM metrics, and coaching plan

Camera Setup:

Front view, 6-10 feet away

Full body visible (head to feet)

Good lighting

3-5 slow reps

📁 Project Structure
text
RehabAI/
├── app.py                          # Gradio web interface
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (API keys)
│
├── src/
│   ├── agents/
│   │   ├── orchestrator.py        # LangGraph agent workflow
│   │   ├── tools.py               # Agent tool functions
│   │   └── movement_agent.py      # Movement analysis agent
│   │
│   ├── pose/
│   │   ├── detector.py            # MediaPipe pose detection
│   │   ├── biomechanics.py        # Angle calculations, ROM
│   │   └── visualization.py       # Pose visualization
│   │
│   └── rag/
│       ├── embeddings.py          # Sentence transformer embeddings
│       ├── vector_store.py        # ChromaDB vector database
│       └── retriever.py           # Hybrid retrieval system
│
├── data/
│   ├── literature/                # Research papers (text chunks)
│   └── vector_db/                 # ChromaDB persistence
│
├── tests/
│   └── evaluation/
│       └── test_custom_eval.py    # Custom LLM-based evaluation
│
└── docs/
    └── images/                    # README screenshots
🔬 Key Features
Biomechanics Analysis
Detects: knee valgus, asymmetries, limited ROM

Measures: knee/hip flexion angles (left/right)

Quality score: 0-10 based on form issues

RAG System
Hybrid search (semantic + keyword)

Retrieves top-3 relevant research papers

Citations included in coaching plan

LLM Coaching
Personalized corrective exercises

Progressive difficulty

Evidence-based recommendations

Safety warnings

Privacy
Videos processed locally

Temporary files auto-deleted

Only biomechanics data (angles) sent to API

📊 Performance
Metric	Time
Video Processing	10-15s
RAG Retrieval	<1s
Coaching Generation	5-8s
Total	15-25s
🧪 Evaluation
Custom LLM-based evaluation using Gemini as judge:

bash
pytest tests/evaluation/test_custom_eval.py -v
Metrics:

Answer Relevancy: 8+/10

Faithfulness: 7+/10

Context Quality: 7+/10

Safety: PASS

Why Custom Evaluation?

Faster: 1 API call per test (vs 20+ for DeepEval)

Cost-effective: Works within Gemini free tier

Transparent: Clear scoring with interpretable results

🔮 Future Work
Support more exercises (deadlift, lunge, overhead press)

Multi-angle analysis

Real-time webcam feedback with live corrections

Progress tracking dashboard over time

PDF report export with visualizations

⚠️ Disclaimer
Educational purposes only. Not a replacement for professional medical advice. Consult a healthcare provider if you have injuries or medical conditions.

📄 License
MIT License - See LICENSE

<div align="center">
Made with ❤️ by [Your Name]

GitHub • LinkedIn

</div> ```