# preprocess.py

from pathlib import Path
import pandas as pd
import numpy as np

KEEP_D = [
    "SEQNUM","ENROLID","SVCDATE","NDCNUM","THERGRP","THERCLS","DAYSUPP","QTY",
    "AWP","COPAY","COINS","DEDUCT","NETPAY","PAID","GENERIC","BRAND",
    "AGE","YEAR","FEMALE","SEX"
]
KEEP_O = [
    "SEQNUM","ENROLID","SVCDATE","PROCTYP","REVCODE","STDPLAC",
    "PROC1","PROC2","PROC3","PROC4","PROC5",
    "DX1","DX2","DX3","DX4","DX5","DX6","DX7","DX8","DX9","DX10","DX11","DX12","DX13","DX14","DX15",
    "AGE","YEAR","FEMALE","SEX","NETPAY","PAID","COINS","COPAY","DEDUCT"
]
KEEP_I = [
    "SEQNUM","ENROLID","ADMITDATE","ADMDATE","DISCHDATE","DISDATE","DRG","MDC","DAYS",
    "PROC1","PROC2","PROC3","PROC4","PROC5","PROC6","PROC7","PROC8","PROC9","PROC10","PROC11","PROC12","PROC13","PROC14","PROC15",
    "DX1","DX2","DX3","DX4","DX5","DX6","DX7","DX8","DX9","DX10","DX11","DX12","DX13","DX14","DX15",
    "HOSPPAY","TOTCHG","NETPAY","AGE","YEAR","FEMALE","SEX"
]

ESSENTIAL_D = ["ENROLID"]
ESSENTIAL_O = ["ENROLID"]
ESSENTIAL_I = ["ENROLID"]

ZEROABLE_PAY_COLS = ["COPAY","COINS","DEDUCT","NETPAY","PAID","HOSPPAY","TOTCHG"]

OUT_DIR = Path("out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def to_datetime_safe(s):
    return pd.to_datetime(s, errors="coerce")

def normalize_code_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.upper()
    s = s.replace({"": np.nan, "NAN": np.nan, "NONE": np.nan, "0": np.nan, "0.0": np.nan})
    s = s.astype(str).str.replace(".", "", regex=False)
    s.loc[s.str.lower()=="nan"] = np.nan
    return s

def normalize_diag_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in [c for c in df.columns if c.upper().startswith("DX")]:
        df[c] = normalize_code_series(df[c])
    return df

def normalize_proc_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in [c for c in df.columns if c.upper().startswith("PROC")]:
        df[c] = normalize_code_series(df[c])
    return df

def drop_and_impute(
    df: pd.DataFrame,
    keep_cols,
    essential_cols,
    *,
    require_drug_signal: bool = False
) -> pd.DataFrame:
    cols_present = [c for c in keep_cols if c in df.columns]
    df = df[cols_present].copy()

    if "ENROLID" in df.columns:
        df["ENROLID"] = df["ENROLID"].astype(str).str.strip()

    for c in ZEROABLE_PAY_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if "AGE" in df.columns:
        df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
        mask_age = (df["AGE"] < 0) | (df["AGE"] > 115)
        df.loc[mask_age, "AGE"] = np.nan

    if "FEMALE" in df.columns:
        df["FEMALE"] = pd.to_numeric(df["FEMALE"], errors="coerce")
    if "SEX" in df.columns:
        df["SEX"] = pd.to_numeric(df["SEX"], errors="coerce")
        if "FEMALE" not in df.columns:
            df["FEMALE"] = np.where(df["SEX"]==2, 1, np.where(df["SEX"]==1, 0, np.nan))

    for c in ["AGE","FEMALE"]:
        if c in df.columns:
            df[f"{c}_MISSING"] = df[c].isna().astype(int)

    has_id = pd.Series(True, index=df.index)
    for c in essential_cols:
        if c in df.columns:
            has_id &= df[c].notna() & (df[c].astype(str).str.strip() != "")
        else:
            has_id &= False

    if require_drug_signal:
        def _has_any_drug_signal(row):
            return (
                (("NDCNUM" in row.index) and pd.notna(row["NDCNUM"]) and str(row["NDCNUM"]).strip() != "") or
                (("THERGRP" in row.index) and pd.notna(row["THERGRP"])) or
                (("THERCLS" in row.index) and pd.notna(row["THERCLS"]))
            )
        if {"NDCNUM","THERGRP","THERCLS"}.intersection(df.columns):
            has_drug = df.apply(_has_any_drug_signal, axis=1)
        else:
            has_drug = pd.Series(False, index=df.index)
        df = df.loc[has_id & has_drug].copy()
    else:
        df = df.loc[has_id].copy()

    if "SEQNUM" in df.columns:
        df = df.drop_duplicates(subset=["SEQNUM"])

    return df

def _first_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("Не се пронајдени dataset CSV фајлови во наведени патеки.")

base_idea = Path(".idea").joinpath("datasets_csv")
base_csv = Path("csv")

d_path = _first_existing([base_idea/"dataset_d.csv", base_csv/"dataset_d.csv"])
o_path = _first_existing([base_idea/"dataset_o.csv", base_csv/"dataset_o.csv"])
i_path = _first_existing([base_idea/"dataset_i.csv", base_csv/"dataset_i.csv"])

D = pd.read_csv(d_path, low_memory=False, dtype={"ENROLID": str, "NDCNUM": str})
O = pd.read_csv(o_path, low_memory=False, dtype={"ENROLID": str})
I = pd.read_csv(i_path, low_memory=False, dtype={"ENROLID": str})

try:
    ids_oi = set(pd.concat([
        O[["ENROLID"]],
        I[["ENROLID"]]
    ], ignore_index=True)["ENROLID"].astype(str))
    before_rows = len(D)
    D = D[D["ENROLID"].astype(str).isin(ids_oi)].copy()
    after_rows = len(D)
    print(f"[preprocess] Filtered D to O/I ENROLIDs: {before_rows} -> {after_rows} rows | ids={len(set(D['ENROLID']))}")
except Exception as _e:
    print(f"[preprocess] Skipping D alignment due to error: {_e}")

if "SVCDATE" in D.columns: D["SVCDATE"] = to_datetime_safe(D["SVCDATE"])
if "SVCDATE" in O.columns: O["SVCDATE"] = to_datetime_safe(O["SVCDATE"])
if "ADMITDATE" in I.columns: I["ADMITDATE"] = to_datetime_safe(I["ADMITDATE"])
if "DISCHDATE" in I.columns: I["DISCHDATE"] = to_datetime_safe(I["DISCHDATE"])
if "ADMDATE" in I.columns and ("ADMITDATE" not in I.columns or I["ADMITDATE"].isna().all()):
    I["ADMITDATE"] = to_datetime_safe(I["ADMDATE"])
if "DISDATE" in I.columns and ("DISCHDATE" not in I.columns or I["DISCHDATE"].isna().all()):
    I["DISCHDATE"] = to_datetime_safe(I["DISDATE"])

if "THERGRP" in D.columns: D["THERGRP"] = pd.to_numeric(D["THERGRP"], errors="coerce")
if "THERCLS" in D.columns: D["THERCLS"] = pd.to_numeric(D["THERCLS"], errors="coerce")

O = normalize_diag_cols(O); I = normalize_diag_cols(I)
O = normalize_proc_cols(O); I = normalize_proc_cols(I)

Dc = drop_and_impute(D, KEEP_D, ESSENTIAL_D, require_drug_signal=True)
Oc = drop_and_impute(O, KEEP_O, ESSENTIAL_O, require_drug_signal=False)
Ic = drop_and_impute(I, KEEP_I, ESSENTIAL_I, require_drug_signal=False)

def _norm_enrolid(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.replace(r"[^\d]", "", regex=True)
    return s

for _df in (Dc, Oc, Ic):
    if "ENROLID" in _df.columns:
        _df["ENROLID"] = _norm_enrolid(_df["ENROLID"])

dx_cols_o = [c for c in Oc.columns if c.upper().startswith("DX")]
dx_cols_i = [c for c in Ic.columns if c.upper().startswith("DX")]

diag_o = (
    Oc[["ENROLID","SVCDATE"] + dx_cols_o]
      .melt(id_vars=["ENROLID","SVCDATE"], value_vars=dx_cols_o,
            var_name="DX_POS", value_name="ICD")
      .dropna(subset=["ICD"])
      .rename(columns={"SVCDATE":"DATE"})
)
diag_o["SOURCE"] = "O"

diag_i = (
    Ic[["ENROLID","ADMITDATE"] + dx_cols_i]
      .rename(columns={"ADMITDATE":"DATE"})
      .melt(id_vars=["ENROLID","DATE"], value_vars=dx_cols_i,
            var_name="DX_POS", value_name="ICD")
      .dropna(subset=["ICD"])
)
diag_i["SOURCE"] = "I"

diagnoses_long = pd.concat([diag_o, diag_i], ignore_index=True, sort=False)

drugs_diag = Dc[["ENROLID","NDCNUM"]].merge(
    diagnoses_long[["ENROLID","ICD","SOURCE","DX_POS"]],
    on="ENROLID", how="inner"
)

Dc.to_csv(OUT_DIR/"D_clean.csv", index=False)
Oc.to_csv(OUT_DIR/"O_clean.csv", index=False)
Ic.to_csv(OUT_DIR/"I_clean.csv", index=False)
diagnoses_long.to_csv(OUT_DIR/"diagnoses_long.csv", index=False)
drugs_diag.to_csv(OUT_DIR/"drugs_diag.csv", index=False)

print("Preprocessing done. Files are in ./out/")
print({
    "D_clean": Dc.shape,
    "O_clean": Oc.shape,
    "I_clean": Ic.shape,
    "diagnoses_long": diagnoses_long.shape,
    "drugs_diag": drugs_diag.shape
})
