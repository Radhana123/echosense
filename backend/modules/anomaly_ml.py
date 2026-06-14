import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats


class AnomalyDetector:

    def __init__(self, contamination=0.05):
        self.contamination = contamination

    def detect(self, df: pd.DataFrame) -> list:
        results = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return []

        iso = self._isolation_forest(df, numeric_cols)
        if iso:
            results.append(iso)

        for col in numeric_cols:
            z   = self._zscore(df, col)
            iqr = self._iqr(df, col)
            if z and iqr:
                agreed = list(set(z["row_indices"]) & set(iqr["row_indices"]))
                if agreed:
                    results.append({
                        "column_name":      col,
                        "anomaly_type":     "outlier",
                        "detection_method": "ensemble_zscore_iqr",
                        "row_indices":      agreed,
                        "anomaly_score":    round((z["anomaly_score"] + iqr["anomaly_score"]) / 2, 3),
                        "p_value":          z["p_value"],
                        "description":      f"'{col}' mein {len(agreed)} outliers mile.",
                    })

        results.extend(self._missing_check(df))
        return results

    def _isolation_forest(self, df, cols):
        X = df[cols].fillna(df[cols].median())
        if len(X) < 20:
            return None
        X_scaled = StandardScaler().fit_transform(X)
        iso = IsolationForest(contamination=self.contamination, random_state=42)
        preds = iso.fit_predict(X_scaled)
        idx = [int(i) for i, p in enumerate(preds) if p == -1]
        if not idx:
            return None
        _, pv = stats.mannwhitneyu(
            iso.decision_function(X_scaled)[preds == -1],
            iso.decision_function(X_scaled)[preds == 1],
            alternative="less"
        )
        return {
            "column_name":      "multiple_columns",
            "anomaly_type":     "multivariate_outlier",
            "detection_method": "isolation_forest",
            "row_indices":      idx,
            "anomaly_score":    0.8,
            "p_value":          round(float(pv), 5),
            "description":      f"{len(idx)} rows globally anomalous hain (p={pv:.4f})",
        }

    def _zscore(self, df, col, threshold=3.0):
        s = df[col].dropna()
        if len(s) < 10:
            return None
        z = np.abs(stats.zscore(s))
        mask = z > threshold
        idx = [int(i) for i in s.index[mask]]
        if not idx:
            return None
        max_z = float(z[mask].max())
        pv = float(2 * (1 - stats.norm.cdf(max_z)))
        return {
            "row_indices":   idx,
            "anomaly_score": round(min(max_z / 10, 1.0), 3),
            "p_value":       round(pv, 5),
        }

    def _iqr(self, df, col, factor=1.5):
        s = df[col].dropna()
        if len(s) < 10:
            return None
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            return None
        mask = (s < Q1 - factor * IQR) | (s > Q3 + factor * IQR)
        idx = [int(i) for i in s.index[mask]]
        if not idx:
            return None
        return {
            "row_indices":   idx,
            "anomaly_score": 0.7,
            "p_value":       0.01,
        }

    def _missing_check(self, df):
        results = []
        for col in df.columns:
            null_pct = df[col].isnull().mean() * 100
            if null_pct > 30:
                results.append({
                    "column_name":      col,
                    "anomaly_type":     "missing_pattern",
                    "detection_method": "null_analysis",
                    "row_indices":      [int(i) for i in df[df[col].isnull()].index[:50]],
                    "anomaly_score":    round(null_pct / 100, 3),
                    "p_value":          0.001,
                    "description":      f"'{col}' mein {null_pct:.1f}% missing values hain.",
                })
        return results


if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "delivery_time": np.concatenate([
            np.random.normal(30, 5, 180),
            np.random.normal(95, 10, 20),
        ]),
        "distance_km":  np.random.uniform(1, 10, n),
        "cancellation": np.random.choice([0, 1], n, p=[0.85, 0.15]),
    })

    detector = AnomalyDetector(contamination=0.1)
    results = detector.detect(df)

    print(f"\n{len(results)} anomaly groups detected:\n")
    for r in results:
        sig = "SIGNIFICANT" if r.get("p_value", 1) < 0.05 else "not significant"
        print(f"  [{sig}]  {r['column_name']}")
        print(f"    Method: {r['detection_method']}")
        print(f"    Rows:   {len(r['row_indices'])}")
        print(f"    Score:  {r['anomaly_score']}  p={r.get('p_value')}")
        print(f"    → {r['description']}\n")