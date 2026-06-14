import sys
sys.path.insert(0, "backend")

import numpy as np
import pandas as pd
from modules.anomaly_ml import AnomalyDetector

print("\n  EchoSense — Week 3 Test\n")

np.random.seed(42)
n = 200

df = pd.DataFrame({
    "delivery_time": np.concatenate([
        np.random.normal(30, 5, 180),
        np.random.normal(95, 10, 20),  # ye anomalies hain
    ]),
    "distance_km":  np.random.uniform(1, 10, n),
    "cancellation": np.random.choice([0, 1], n, p=[0.85, 0.15]),
})

detector = AnomalyDetector(contamination=0.1)
results = detector.detect(df)

sig = [r for r in results if r.get("p_value", 1) < 0.05]

print(f"Total anomaly groups: {len(results)}")
print(f"Significant ones:     {len(sig)}\n")

for r in sig:
    print(f"  [{r['anomaly_type']}] {r['column_name']}")
    print(f"    Rows affected: {len(r['row_indices'])}")
    print(f"    p-value: {r['p_value']}")
    print(f"    → {r['description']}\n")

print(f"{'='*50}")
print("  ✓ WEEK 3 COMPLETE!")
print("  Next: Week 4 — Causal AI (DoWhy)")
print(f"{'='*50}\n")