# EchoSense — AI Causal Story Engine

> "Don't just ask WHAT happened in your data. Ask WHY."

EchoSense automatically discovers **causal relationships** in any dataset and generates an **investigative report** explaining *why* patterns occur.

## Live Demo
Upload any CSV → Get causal chains + anomalies + AI report in 60 seconds.

## Tech Stack
| Module | Technology |
|--------|-----------|
| Causal AI | DoWhy + networkx |
| Anomaly Detection | Isolation Forest + Z-Score + IQR |
| NLP Profiler | regex + keyword voting |
| GenAI Report | Groq API (llama3-70b) |
| Backend | FastAPI + SQLAlchemy |
| Database | SQLite / PostgreSQL |
| Frontend | Streamlit |

## Project Structure


## How to Run
```bash
# Terminal 1 — Backend
python backend\api\main.py

# Terminal 2 — Frontend
streamlit run frontend\app.py
```

## Key Features
- **Causal Discovery** — finds WHY patterns occur (not just correlation)
- **Anomaly Detection** — 3-method ensemble with statistical validation
- **Auto Domain Detection** — automatically identifies dataset domain
- **AI Report** — generates investigative article from findings

## Interview Talking Points
**Q: Why DoWhy instead of correlation?**
DoWhy's backdoor criterion controls for confounders and gives Average Treatment Effect — what actually changes if we intervene on a variable.

**Q: How did you validate causal claims?**
Placebo refutation test — shuffle the cause variable randomly. If effect disappears, the claim is valid.

**Built by:** Radhana