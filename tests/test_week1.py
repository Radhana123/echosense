import sys
sys.path.insert(0, "backend")

from db.database import init_db, SessionLocal
from db.models import Dataset, ColumnProfile

def line(msg):
    print(f"\n{'─'*50}\n  {msg}\n{'─'*50}")

print("\n  EchoSense — Week 1 Test\n")

# Test 1
line("Test 1: Tables banao")
init_db()
print("✓ Tables ready!")

# Test 2
line("Test 2: Dataset save karo")
db = SessionLocal()
ds = Dataset(
    filename="zomato_test.csv",
    filepath="data/samples/zomato_test.csv",
    row_count=50000,
    col_count=8,
    domain="logistics",
    domain_confidence=0.87
)
db.add(ds)
db.commit()
db.refresh(ds)
print(f"✓ Saved! ID: {ds.id}, domain: {ds.domain}")

# Test 3
line("Test 3: Column profiles save karo")
profiles = [
    ColumnProfile(dataset_id=ds.id, column_name="weather",
                  inferred_type="categorical", semantic_type="general",
                  null_percent=0.0, unique_count=3,
                  sample_values=["Rain","Clear","Cloud"], stats={}),
    ColumnProfile(dataset_id=ds.id, column_name="delivery_time",
                  inferred_type="numeric", semantic_type="time_duration",
                  null_percent=0.0, unique_count=892,
                  sample_values=["28","67","94"],
                  stats={"mean":42.3,"std":18.7,"min":10,"max":120}),
    ColumnProfile(dataset_id=ds.id, column_name="cancellation",
                  inferred_type="binary", semantic_type="target_variable",
                  null_percent=0.0, unique_count=2,
                  sample_values=["0","1"], stats={}),
]
for p in profiles:
    db.add(p)
db.commit()
print(f"✓ {len(profiles)} profiles saved!")

# Test 4
line("Test 4: Query karo")
found = db.query(Dataset).filter(Dataset.domain=="logistics").first()
print(f"✓ Found: {found.filename} ({found.row_count} rows)")
cols = db.query(ColumnProfile).filter(
    ColumnProfile.dataset_id==ds.id
).all()
for c in cols:
    print(f"   {c.column_name:22} {c.inferred_type:14} {c.semantic_type}")

db.close()
print(f"\n{'='*50}")
print("  ✓ WEEK 1 COMPLETE!")
print("  Next: Week 2 — NLP Column Profiler")
print(f"{'='*50}\n")