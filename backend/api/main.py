import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from config import settings
from db.database import get_db, init_db
from db import crud
from modules.profiler import ColumnProfiler
from modules.anomaly_ml import AnomalyDetector
from modules.causal import CausalAnalyzer
from modules.report import ReportWriter

app = FastAPI(title="EchoSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

profiler  = ColumnProfiler()
detector  = AnomalyDetector(contamination=0.05)
analyzer  = CausalAnalyzer(min_correlation=0.1)
writer    = ReportWriter()


@app.on_event("startup")
def startup():
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    init_db()
    print("EchoSense API ready.")


@app.get("/")
def root():
    return {"status": "ok", "service": "EchoSense API v1.0"}


@app.post("/upload")
async def upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Sirf CSV files allowed hain.")

    contents = await file.read()
    path = os.path.join(settings.UPLOAD_DIR, f"{int(time.time())}_{file.filename}")

    with open(path, "wb") as f:
        f.write(contents)

    df = pd.read_csv(path)
    ds = crud.create_dataset(db, file.filename, path, len(df), len(df.columns), len(contents))
    pr = profiler.profile_dataframe(df)
    crud.update_dataset_domain(db, ds.id, pr["domain"], pr["domain_confidence"])
    crud.save_column_profiles(db, ds.id, pr["column_profiles"])

    return {
        "dataset_id": ds.id,
        "filename":   file.filename,
        "rows":       len(df),
        "domain":     pr["domain"],
        "columns":    pr["column_profiles"],
    }


@app.get("/datasets")
def list_datasets(db: Session = Depends(get_db)):
    datasets = crud.get_all_datasets(db)
    return {
        "datasets": [
            {"id": d.id, "filename": d.filename,
             "rows": d.row_count, "domain": d.domain}
            for d in datasets
        ]
    }


@app.post("/analyze/{dataset_id}")
def analyze(dataset_id: int,
            target_col: str = Query(None),
            db: Session = Depends(get_db)):

    ds = crud.get_dataset(db, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    run = crud.create_analysis_run(db, dataset_id)
    t0  = time.time()

    try:
        df    = pd.read_csv(ds.filepath)
        anom  = detector.detect(df)
        crud.save_anomalies(db, dataset_id, anom)

        caus  = analyzer.analyze(df, target_col=target_col)
        crud.save_causal_edges(db, dataset_id, caus["causal_edges"])

        pr    = profiler.profile_dataframe(df)
        rep   = writer.generate(
            domain=ds.domain or "general",
            anomalies=anom,
            causal_edges=caus["causal_edges"],
            causal_chains=caus["causal_chains"],
            dataset_info=pr["summary"],
            target_variable=caus.get("target_variable"),
        )

        valid = [e for e in caus["causal_edges"] if not e.get("is_spurious")]
        sig   = [a for a in anom if a.get("p_value", 1) < 0.05]

        crud.complete_analysis_run(
            db, run.id, len(sig), len(valid),
            ds.domain, rep["text"], time.time() - t0
        )

        return {
            "run_id":        run.id,
            "status":        "completed",
            "anomaly_count": len(sig),
            "causal_chains": caus["causal_chains"],
            "report":        rep["text"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{dataset_id}")
def results(dataset_id: int, db: Session = Depends(get_db)):
    ds    = crud.get_dataset(db, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    anom  = crud.get_anomalies(db, dataset_id)
    edges = crud.get_causal_edges(db, dataset_id)
    return {
        "anomalies": [{"col": a.column_name, "desc": a.description} for a in anom],
        "causal_edges": [{"cause": e.cause_variable, "effect": e.effect_variable,
                          "strength": e.causal_strength} for e in edges],
    }


@app.get("/history")
def history(db: Session = Depends(get_db)):
    runs = crud.get_analysis_history(db)
    return {
        "runs": [{"id": r.id, "status": r.status,
                  "anomalies": r.anomaly_count} for r in runs]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)