# EchoSense — AI Causal Story Engine

> "Don't just ask WHAT happened in your data. Ask WHY."

EchoSense automatically discovers **causal relationships** in any dataset and generates an **investigative report** explaining *why* patterns occur — and now remembers its own past investigations to ground new ones in precedent.

## Live Demo
Upload any CSV → Get causal chains + anomalies + AI report in 60 seconds.

## Tech Stack
| Module | Technology |
|--------|-----------|
| Causal AI | DoWhy + networkx |
| Anomaly Detection | Isolation Forest + Z-Score + IQR |
| NLP Profiler | regex + keyword voting |
| GenAI Report | Groq API (llama3-70b) |
| Investigation Memory (RAG) | sentence-transformers (all-MiniLM-L6-v2) + FAISS |
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
- **Investigation Memory (RAG)** — retrieves similar past investigations before writing a new report, so findings are grounded in precedent instead of starting cold on every dataset
- **AI Report** — generates investigative article from findings

---

## Investigation Memory: Retrieval-Augmented Report Generation

DoWhy needs domain assumptions to build a causal graph, and by default those assumptions come from scratch on every single dataset — the tool has no way to know it has seen something like this before. EchoSense closes that gap with a second, dynamic RAG layer (separate from the static domain-knowledge retriever used for NLP profiling): every completed investigation whose causal claim survives placebo refutation testing is embedded and stored, and the strongest causal chain plus target variable from a new dataset is used to retrieve the *k* most similar past cases before the LLM writes its report — the same way a human analyst gets faster and more accurate with experience on similar data.

Design choices:
- **Store only validated findings.** A case is only written to memory *after* refutation testing passes — an unvalidated claim is never retrieved and repeated as if it were established.
- **Context, never evidence.** Retrieved cases are passed to the LLM explicitly labeled as background only; the system prompt instructs it to never state a past finding as confirmed for the current dataset.
- **Cosine similarity over normalized embeddings** (FAISS `IndexFlatIP`) on a domain + causal-chain summary string, so retrieval matches on "what this investigation is about," not surface-level column names.

### Validation: run across 3 diverse real-world datasets

To verify the pipeline generalizes (not just tuned to one dataset) and that RAG actually persists and retrieves across runs, EchoSense was run end-to-end on 3 datasets spanning unrelated domains:

| Dataset | Domain | Rows | Causal edges confirmed | Causal chains | Anomalies flagged |
|---|---|---|---|---|---|
| Zomato Delivery Operations | Logistics | 45,584 | 15 | 2 (100% confidence) | 6 |
| IBM HR Analytics (Attrition) | HR | 1,470 | — | 5 (100% confidence) | 6 |
| Heart Disease (UCI) | Healthcare | 1,025 | — | 4 (100% confidence) | 7 |
| **Total** | **3 domains** | **48,079** | — | **11 chains** | **19** |

Re-running the Zomato dataset after its first pass confirmed the memory layer works end-to-end in the live API, not just in isolation: the first run stores its validated causal chain (e.g. `Delivery_person_Ratings → multiple_deliveries → Delivery_person_Age → Time_taken`) to the FAISS-backed store, and subsequent runs on the same domain retrieve it as grounding context before generating their own report.

One implementation detail worth flagging honestly: reported confidence values cluster at 100% because confidence is computed as `1 − p_value`, and most edges here have p-values well under 0.001 — a genuine artifact of clean, high-signal datasets and display rounding, not a validation bug (refutation testing via random permutation of the cause variable is still what separates the 11 reported chains from the edges rejected as spurious).

---

## How to Run
```bash
# Terminal 1 — Backend
python backend\api\main.py

# Terminal 2 — Frontend
streamlit run frontend\app.py
```

## Interview Talking Points
**Q: Why DoWhy instead of correlation?**
DoWhy's backdoor criterion controls for confounders and gives Average Treatment Effect — what actually changes if we intervene on a variable.

**Q: How did you validate causal claims?**
Placebo refutation test — shuffle the cause variable randomly. If effect disappears, the claim is valid.

**Q: Why add a RAG layer to a causal discovery tool specifically?**
Because DoWhy's assumptions are rebuilt from zero on every dataset. A memory of past validated investigations lets a new one start from precedent instead of a cold start — and because only refutation-passed cases are stored and retrieved as context (never as evidence), the memory can't compound its own past mistakes.

**Built by:** Radhana