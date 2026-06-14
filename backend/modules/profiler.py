import re
import pandas as pd
import numpy as np
from typing import Optional

DOMAIN_KEYWORDS = {
    "logistics":  ["delivery", "order", "shipment", "dispatch", "courier",
                   "distance", "route", "rider", "cancellation", "tracking"],
    "healthcare": ["patient", "disease", "diagnosis", "blood", "glucose",
                   "bmi", "hospital", "doctor", "cholesterol", "heart"],
    "finance":    ["revenue", "profit", "transaction", "balance", "loan",
                   "credit", "payment", "interest", "stock", "investment"],
    "ecommerce":  ["product", "cart", "purchase", "seller", "discount",
                   "review", "refund", "category", "inventory", "rating"],
    "hr":         ["employee", "salary", "attrition", "department", "tenure",
                   "performance", "promotion", "overtime", "jobtitle", "manager"],
}

SEMANTIC_PATTERNS = {
    "target_variable": r"(target|label|cancel|churn|fraud|default|outcome|result)",
    "revenue_metric":  r"(revenue|sales|amount|price|cost|payment|profit|income)",
    "time_duration":   r"(duration|time|minutes|hours|delay|wait|eta)",
    "contact":         r"(email|phone|mobile|address|contact)",
    "date_column":     r"(date|timestamp|created|updated|day|month|year)",
    "id_column":       r"(_id$|^id$|_key$|_code$|identifier)",
    "rating_metric":   r"(rating|score|stars|satisfaction|review)",
    "geographic":      r"(city|state|country|region|lat|lng|zone|area)",
    "demographic":     r"(age|gender|sex|dob|birth|education)",
}


class ColumnProfiler:

    def profile_dataframe(self, df: pd.DataFrame) -> dict:
        profiles = [self._profile_column(df[col], col) for col in df.columns]
        domain, conf = self._detect_domain(df.columns.tolist())
        target = self._find_target(profiles)
        if target:
            for p in profiles:
                if p["column_name"] == target:
                    p["is_target"] = True
        return {
            "domain": domain,
            "domain_confidence": round(conf, 3),
            "column_profiles": profiles,
            "summary": self._summary(df, profiles),
        }

    def _profile_column(self, series: pd.Series, col_name: str) -> dict:
        null_pct = round(series.isnull().mean() * 100, 2)
        inferred = self._infer_dtype(series)
        semantic = self._infer_semantic(col_name)
        stats = {}
        if inferred == "numeric":
            stats = {
                "mean":   round(float(series.mean()), 3),
                "std":    round(float(series.std()),  3),
                "min":    round(float(series.min()),  3),
                "max":    round(float(series.max()),  3),
                "median": round(float(series.median()), 3),
            }
        return {
            "column_name":   col_name,
            "inferred_type": inferred,
            "semantic_type": semantic,
            "null_percent":  null_pct,
            "unique_count":  int(series.nunique()),
            "sample_values": [str(v) for v in series.dropna().head(5).tolist()],
            "stats":         stats,
            "is_target":     False,
        }

    def _infer_dtype(self, series: pd.Series) -> str:
        if pd.api.types.is_numeric_dtype(series):
            vals = set(series.dropna().unique())
            return "binary" if vals.issubset({0, 1, 0.0, 1.0}) else "numeric"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "date"
        unique_lower = set(series.dropna().astype(str).str.lower().unique())
        if unique_lower.issubset({"yes","no","true","false","y","n","1","0"}):
            return "binary"
        total = len(series.dropna())
        if series.nunique() <= 20 or (total > 0 and series.nunique()/total < 0.05):
            return "categorical"
        return "text"

    def _infer_semantic(self, col_name: str) -> str:
        col = col_name.lower()
        for semantic, pattern in SEMANTIC_PATTERNS.items():
            if re.search(pattern, col, re.IGNORECASE):
                return semantic
        return "general"

    def _detect_domain(self, cols: list) -> tuple:
        col_text = " ".join(cols).lower()
        scores = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            scores[domain] = sum(1 for kw in keywords if kw in col_text)
        total = sum(scores.values())
        if total == 0:
            return "general", 0.5
        best = max(scores, key=scores.get)
        return best, scores[best] / total

    def _find_target(self, profiles: list) -> Optional[str]:
        for p in profiles:
            if p["semantic_type"] == "target_variable" and p["inferred_type"] == "binary":
                return p["column_name"]
        binary = [p["column_name"] for p in profiles if p["inferred_type"] == "binary"]
        return binary[0] if binary else None

    def _summary(self, df: pd.DataFrame, profiles: list) -> dict:
        total_cells = df.shape[0] * df.shape[1]
        total_nulls = sum(p["null_percent"] * df.shape[0] / 100 for p in profiles)
        return {
            "total_rows":      df.shape[0],
            "total_columns":   df.shape[1],
            "overall_null_rate": round(total_nulls/total_cells*100, 2) if total_cells > 0 else 0,
            "numeric_cols":    sum(1 for p in profiles if p["inferred_type"] == "numeric"),
            "categorical_cols":sum(1 for p in profiles if p["inferred_type"] == "categorical"),
            "binary_cols":     sum(1 for p in profiles if p["inferred_type"] == "binary"),
        }


if __name__ == "__main__":
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
    print(f"\nDomain: {result['domain'].upper()} (confidence: {result['domain_confidence']})")
    print(f"Summary: {result['summary']}\n")
    print("Column Profiles:")
    for p in result["column_profiles"]:
        target = " ← TARGET" if p.get("is_target") else ""
        print(f"  {p['column_name']:20} type:{p['inferred_type']:13} sem:{p['semantic_type']}{target}")