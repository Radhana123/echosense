import sys
sys.path.insert(0, "backend")

import numpy as np
import pandas as pd
from modules.causal import CausalAnalyzer

print("\n  EchoSense — Week 4 Test\n")

np.random.seed(42)
n = 500

rain     = np.random.binomial(1, 0.3, n)
delay    = 30 + 45 * rain + np.random.normal(0, 8, n)
rating   = np.clip(5 - 0.04 * delay + np.random.normal(0, 0.5, n), 1, 5)
cancel   = (rating < 2.5).astype(int)
distance = np.random.uniform(1, 10, n)

df = pd.DataFrame({
    "rain":          rain,
    "delivery_time": delay,
    "rating":        rating,
    "cancellation":  cancel,
    "distance_km":   distance,
})

result = CausalAnalyzer(min_correlation=0.1).analyze(df, target_col="cancellation")

valid    = [e for e in result["causal_edges"] if not e.get("is_spurious")]
spurious = [e for e in result["causal_edges"] if e.get("is_spurious")]

print(f"Target variable:  {result['target_variable']}")
print(f"Valid edges:      {len(valid)}")
print(f"Spurious rejected:{len(spurious)}\n")

print("VALID CAUSAL EDGES:")
for e in valid:
    print(f"  ✓ {e['cause']} → {e['effect']}  (conf={e['confidence']})")

print("\nCAUSAL CHAINS:")
for ch in result["causal_chains"]:
    print(f"  {ch['chain_str']}  (conf={ch['min_confidence']})")

print(f"\n{'='*50}")
print("  ✓ WEEK 4 COMPLETE!")
print("  Next: Week 5 — DL + RAG")
print(f"{'='*50}\n")