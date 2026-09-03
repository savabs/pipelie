"""A day in the life of a data scientist who is about to ship a wrong number."""
import numpy as np, pandas as pd, pipelie
rng = np.random.default_rng(42)
n = 4000

# A customer table assembled from two source systems, as they always are.
df = pd.DataFrame({
    "customer_id":  [f"C{i:05d}" for i in range(n)],
    "signup_date":  ["2024-03-11"] * (n//2) + ["3/11/2024"] * (n - n//2),
    "region":       rng.choice(["North","north ","South","East","West"], n),
    "revenue":      rng.lognormal(6, 1, n).round(2),
    "churn_risk":   np.arange(1_700_000_000, 1_700_000_000 + n),   # never computed
    "credit_limit": rng.choice([5000.0, -999], n, p=[0.88, 0.12]), # -999 = unknown
    "churned":      rng.integers(0, 2, n),
})
# The reporting job ran twice on Tuesday.
df = pd.concat([df, df.head(300)], ignore_index=True)
# Support notes only exist for customers who called -- and callers churn more.
df["support_note"] = np.where(df["churned"] == 1, "escalated", None)

print("=" * 78)
print("STEP 1 — the checks most teams actually run")
print("=" * 78)
print(f"rows: {len(df):,}   columns: {df.shape[1]}")
print("\nnulls per column:")
print(df.isna().sum().to_string())
print("\ndtypes all as expected:", list(df.dtypes.astype(str).unique()))
print("\nrevenue looks sane:", f"mean={df.revenue.mean():,.0f}  min={df.revenue.min():,.0f}  max={df.revenue.max():,.0f}")
print("\n>>> Every one of those passes. Schema fine. Nulls low. Ranges plausible.")
print(">>> This is the point at which the number gets shipped.\n")

print("=" * 78)
print("STEP 2 — the number you are about to report")
print("=" * 78)
rate = df.churned.mean()
top = df.nlargest(5, "churn_risk")[["customer_id", "churn_risk"]]
print(f"churn rate: {rate:.1%}")
print(f"average credit limit: {df.credit_limit.mean():,.0f}")
print(f"highest-risk customers to call first:\n{top.to_string(index=False)}\n")

print("=" * 78)
print("STEP 3 — pipelie")
print("=" * 78)
print(pipelie.audit(df, target="churned", key=["customer_id"]))
