import sys
sys.path.insert(0, "backend")

import pandas as pd
from modules.profiler import ColumnProfiler

print("\n  EchoSense — Week 2 Test\n")

# Test data — Zomato jaisa
test_data = {
    "order_id":       ["ORD001","ORD002","ORD003","ORD004","ORD005"],
    "weather":        ["Clear","Rain","Rain","Clear","Rain"],
    "delivery_time":  [28, 67, 94, 31, 88],
    "distance_km":    [3.2, 3.1, 2.9, 4.0, 3.3],
    "cancellation":   [0, 0, 1, 0, 1],
    "customer_email": ["a@b.com","c@d.com","e@f.com","g@h.com","i@j.com"],
}

df = pd.DataFrame(test_data)
result = ColumnProfiler().profile_dataframe(df)

print(f"Domain: {result['domain'].upper()}")
print(f"Confidence: {result['domain_confidence']}")
print(f"Total rows: {result['summary']['total_rows']}")
print(f"Numeric cols: {result['summary']['numeric_cols']}")
print()
print("Column Profiles:")
for p in result["column_profiles"]:
    target = " ← TARGET" if p.get("is_target") else ""
    print(f"  {p['column_name']:20} {p['inferred_type']:14} {p['semantic_type']}{target}")

print(f"\n{'='*50}")
print("  ✓ WEEK 2 COMPLETE!")
print("  Next: Week 3 — Anomaly Detection")
print(f"{'='*50}\n")