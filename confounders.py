# confounders.py
import os
import numpy as np
import pandas as pd

def build_covariates(
    therapy_df: pd.DataFrame,
    dataset_d_path: str = "out/D_clean.csv",
    dataset_o_path: str = "out/O_clean.csv",
) -> pd.DataFrame:
    D = pd.read_csv(dataset_d_path, low_memory=False, dtype={"ENROLID": str, "NDCNUM": str})
    O = pd.read_csv(dataset_o_path, low_memory=False, dtype={"ENROLID": str})

    therapy_df = therapy_df.copy()
    therapy_df["ENROLID"] = therapy_df["ENROLID"].astype(str)
    D["ENROLID"] = D["ENROLID"].astype(str)
    O["ENROLID"] = O["ENROLID"].astype(str)

    demo_age = D.groupby("ENROLID")["AGE"].median(numeric_only=True).rename("AGE")
    if "SEX" in D.columns:
        sex_mode = (
            D.groupby("ENROLID")["SEX"]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
            .rename("SEX")
        )
        demo = pd.concat([demo_age, sex_mode], axis=1).reset_index()
    else:
        demo = demo_age.reset_index()
        demo["SEX"] = np.nan

    dx_cols = [c for c in O.columns if c.upper().startswith("DX")]
    for c in dx_cols:
        O[c] = (
            O[c].astype(str)
            .str.strip()
            .replace({"": np.nan, "nan": np.nan, "NONE": np.nan, "0": np.nan, "0.0": np.nan}, regex=False)
        )
        O[c] = O[c].astype(str).str.replace(".", "", regex=False).str.upper()

    diag_long = (
        O.melt(id_vars=["ENROLID"], value_vars=dx_cols, var_name="DX_position", value_name="ICD")
        .dropna(subset=["ICD"])
    )

    diag_long_subset = diag_long[diag_long["ENROLID"].isin(therapy_df["ENROLID"])]
    top_icd = diag_long_subset["ICD"].value_counts().head(20).index

    O_features = (
        pd.get_dummies(
            diag_long_subset[diag_long_subset["ICD"].isin(top_icd)][["ENROLID", "ICD"]],
            columns=["ICD"],
            prefix="ICD"
        )
        .groupby("ENROLID")
        .max()
        .reset_index()
    )

    D["THERCLS"] = pd.to_numeric(D.get("THERCLS"), errors="coerce")
    drug_features = (
        D.groupby(["ENROLID", "THERCLS"])["NDCNUM"]
        .count()
        .unstack(fill_value=0)
    )
    drug_features = (drug_features > 0).astype(int).reset_index()
    drug_features = drug_features.rename(columns=lambda c: f"THERCLS_{int(c)}" if isinstance(c, float) or isinstance(c, int) else c)

    cov = therapy_df.merge(demo, on="ENROLID", how="left")
    cov = cov.merge(O_features, on="ENROLID", how="left")
    cov = cov.merge(drug_features, on="ENROLID", how="left")

    indicator_cols = [c for c in cov.columns if c.startswith("ICD_") or c.startswith("THERCLS_")]
    cov[indicator_cols] = cov[indicator_cols].fillna(0).astype(int)

    os.makedirs("out", exist_ok=True)
    cov.to_csv("out/confounders.csv", index=False)

    print("[confounders] Shape:", cov.shape, "| ICD cols:", len([c for c in cov.columns if c.startswith('ICD_')]),
          "| Drug cls cols:", len([c for c in cov.columns if c.startswith('THERCLS_')]))
    return cov


if __name__ == "__main__":
    ther = pd.read_csv("out/therapy_cohort.csv", dtype={"ENROLID": str})
    cov = build_covariates(ther)
    print(cov.head(), "\nDone.")
