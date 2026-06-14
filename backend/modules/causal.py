import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from typing import Optional


class CausalAnalyzer:

    def __init__(self, min_correlation=0.15):
        self.min_corr = min_correlation

    def analyze(self, df: pd.DataFrame, target_col=None) -> dict:
        num_df = self._prepare(df)
        if num_df.shape[1] < 2:
            return {"causal_edges": [], "causal_chains": []}

        target = target_col if target_col in num_df.columns else num_df.columns[-1]
        pairs  = self._candidate_pairs(num_df, target)
        edges  = [e for c, ef in pairs for e in [self._estimate(num_df, c, ef)] if e]
        edges  = self._refute(num_df, edges)
        chains = self._build_chains(edges, target)
        edges.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        return {
            "causal_edges":  edges,
            "causal_chains": chains,
            "target_variable": target,
            "summary": {
                "causal_edges_found":  len([e for e in edges if not e.get("is_spurious")]),
                "spurious_rejected":   len([e for e in edges if e.get("is_spurious")]),
            }
        }

    def _prepare(self, df):
        res = pd.DataFrame()
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                res[col] = df[col].fillna(df[col].median())
            elif df[col].nunique() <= 10:
                m = {v: i for i, v in enumerate(df[col].dropna().unique())}
                res[col] = df[col].map(m).fillna(-1)
        return res

    def _candidate_pairs(self, df, target):
        pairs = [
            (c, target)
            for c in df.columns if c != target
            and abs(df[c].corr(df[target])) >= self.min_corr
        ]
        non = [c for c in df.columns if c != target]
        for a, b in combinations(non, 2):
            if abs(df[a].corr(df[b])) >= self.min_corr:
                pairs.append((a, b) if df[a].var() < df[b].var() else (b, a))
        return pairs[:15]

    def _estimate(self, df, cause, effect):
        try:
            from dowhy import CausalModel
            g = f'digraph{{"{cause}"->"{effect}";}}'
            m = CausalModel(data=df, treatment=cause, outcome=effect, graph=g)
            est = m.estimate_effect(
                m.identify_effect(proceed_when_unidentifiable=True),
                method_name="backdoor.linear_regression",
                target_units="ate"
            )
            ate = float(est.value)
        except:
            slope, _, _, _, _ = stats.linregress(df[cause], df[effect])
            ate = float(slope)

        corr, pv = stats.pearsonr(df[cause], df[effect])
        if abs(corr) < self.min_corr or pv > 0.1:
            return None

        return {
            "cause":      cause,
            "effect":     effect,
            "strength":   round(ate, 4),
            "confidence": round(max(0, min(1, 1 - pv)), 4),
            "p_value":    round(float(pv), 5),
            "correlation":round(corr, 4),
            "is_spurious":False,
        }

    def _refute(self, df, edges):
        for e in edges:
            pl = df.copy()
            pl[e["cause"]] = np.random.permutation(pl[e["cause"]].values)
            slope, _, _, _, _ = stats.linregress(pl[e["cause"]], pl[e["effect"]])
            ratio = abs(slope) / (abs(e["strength"]) + 1e-8)
            e["is_spurious"]      = ratio > 0.8
            e["refutation_ratio"] = round(ratio, 3)
        return edges

    def _build_chains(self, edges, target):
        valid = [e for e in edges if not e.get("is_spurious")]
        eff2cause = {}
        for e in valid:
            eff2cause.setdefault(e["effect"], []).append(e)

        chains = []

        def dfs(node, chain, visited):
            if node not in eff2cause:
                if len(chain) >= 2:
                    chains.append(list(reversed(chain)))
                return
            for e in eff2cause[node]:
                c = e["cause"]
                if c not in visited:
                    visited.add(c)
                    chain.append({"from": c, "to": node,
                                  "strength": e["strength"],
                                  "confidence": e["confidence"]})
                    dfs(c, chain, visited)
                    chain.pop()
                    visited.remove(c)

        dfs(target, [], {target})
        chains.sort(key=len, reverse=True)

        return [
            {
                "nodes":          [ch[0]["from"]] + [s["to"] for s in ch],
                "chain_str":      " → ".join([ch[0]["from"]] + [s["to"] for s in ch]),
                "min_confidence": round(min(s["confidence"] for s in ch), 3),
            }
            for ch in chains[:5]
        ]


if __name__ == "__main__":
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

    print("\nRunning causal analysis...")
    result = CausalAnalyzer(min_correlation=0.1).analyze(df, target_col="cancellation")

    print(f"\nTarget: {result['target_variable']}")
    print(f"Summary: {result['summary']}\n")

    print("CAUSAL EDGES:")
    for e in result["causal_edges"]:
        status = "SPURIOUS ✗" if e.get("is_spurious") else "VALID ✓"
        print(f"  [{status}] {e['cause']} → {e['effect']}")
        print(f"    strength={e['strength']}  conf={e['confidence']}  p={e['p_value']}")

    print("\nCAUSAL CHAINS:")
    for ch in result["causal_chains"]:
        print(f"  {ch['chain_str']}  (conf={ch['min_confidence']})")