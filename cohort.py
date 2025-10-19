# cohort.py
import os
import pandas as pd
from typing import Iterable

def _normalize_icd_cols(df: pd.DataFrame, icd_like_cols: Iterable[str]) -> pd.DataFrame:
    for c in icd_like_cols:
        if c in df.columns:
            s = df[c].astype(str).str.strip()
            s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
            s = s.str.replace(".", "", regex=False).str.upper()
            df[c] = s
    return df


def _icd_mask(series: pd.Series, icd_codes) -> pd.Series:
    if not icd_codes:
        return pd.Series([False] * len(series), index=series.index)
    return series.str.startswith(tuple(icd_codes))


def _extract_dx_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.upper().startswith("DX")]
    if cols:
        return cols
    cols = [c for c in df.columns if "DX" in c.upper()]
    return cols

def _from_outpatient(outpatient_path: str, icd_codes, debug: bool = False) -> pd.DataFrame:
    if not os.path.exists(outpatient_path):
        if debug:
            print(f"[outpatient] Патеката не постои: {outpatient_path}")
        return pd.DataFrame({"ENROLID": []})

    O = pd.read_csv(outpatient_path, dtype={"ENROLID": str}, low_memory=False)
    dx_cols = _extract_dx_cols(O)

    if debug:
        print(f"[outpatient] rows={len(O)}, dx_cols={len(dx_cols)} -> {dx_cols[:8]}")

    if not dx_cols:
        return pd.DataFrame({"ENROLID": []})

    O = _normalize_icd_cols(O, dx_cols)

    long_dx = (
        O.melt(id_vars=["ENROLID"], value_vars=dx_cols, var_name="DX_pos", value_name="ICD")
         .dropna(subset=["ICD"])
    )

    if debug and not long_dx.empty:
        print("[outpatient] Топ ICD (по нормализација):")
        print(long_dx["ICD"].value_counts().head(10))

    mask = _icd_mask(long_dx["ICD"], icd_codes)
    ids = long_dx.loc[mask, "ENROLID"].dropna().unique()

    return pd.DataFrame({"ENROLID": ids})


def _from_inpatient(inpatient_path: str, icd_codes, debug: bool = False) -> pd.DataFrame:
    if not os.path.exists(inpatient_path):
        if debug:
            print(f"[inpatient] Патеката не постои: {inpatient_path}")
        return pd.DataFrame({"ENROLID": []})

    I = pd.read_csv(inpatient_path, dtype={"ENROLID": str}, low_memory=False)
    dx_cols = _extract_dx_cols(I)

    if debug:
        print(f"[inpatient] rows={len(I)}, dx_cols={len(dx_cols)} -> {dx_cols[:8]}")

    if not dx_cols:
        return pd.DataFrame({"ENROLID": []})

    I = _normalize_icd_cols(I, dx_cols)

    long_dx = (
        I.melt(id_vars=["ENROLID"], value_vars=dx_cols, var_name="DX_pos", value_name="ICD")
         .dropna(subset=["ICD"])
    )

    if debug and not long_dx.empty:
        print("[inpatient] Топ ICD (по нормализација):")
        print(long_dx["ICD"].value_counts().head(10))

    mask = _icd_mask(long_dx["ICD"], icd_codes)
    ids = long_dx.loc[mask, "ENROLID"].dropna().unique()

    return pd.DataFrame({"ENROLID": ids})


def _from_drugs_diag(icd_codes, debug: bool = False) -> pd.DataFrame:
    dd_path = "out/drugs_diag.csv"
    if not os.path.exists(dd_path):
        if debug:
            print(f"[drugs_diag] Нема датотека: {dd_path}")
        return pd.DataFrame({"ENROLID": []})

    dd = pd.read_csv(dd_path, dtype={"ENROLID": str, "NDCNUM": str, "ICD": str}, low_memory=False)
    dd["ICD"] = (
        dd["ICD"].astype(str)
                 .str.strip()
                 .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
                 .str.replace(".", "", regex=False)
                 .str.upper()
    )

    if debug and not dd.empty:
        print(f"[drugs_diag] rows={len(dd)}")
        print("[drugs_diag] Топ ICD (по нормализација):")
        print(dd["ICD"].value_counts().head(10))

    mask = _icd_mask(dd["ICD"], icd_codes)
    ids = dd.loc[mask, "ENROLID"].dropna().unique()

    return pd.DataFrame({"ENROLID": ids})

def _union_ids(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    all_ids = pd.Series(dtype=str)
    for df in dfs:
        if df is not None and not df.empty and "ENROLID" in df.columns:
            all_ids = pd.concat([all_ids, df["ENROLID"].astype(str)], ignore_index=True)
    if all_ids.empty:
        return pd.DataFrame({"ENROLID": []})
    all_ids = all_ids.dropna().astype(str).str.strip()
    all_ids = all_ids[all_ids != ""]
    return pd.DataFrame({"ENROLID": sorted(set(all_ids))})

def build_cohort(
    inpatient_path: str,
    outpatient_path: str,
    pharma_path: str,
    icd_codes=None,
    prefer_source: str = "outpatient",
    debug: bool = False
) -> pd.DataFrame:
    if icd_codes is None:
        icd_codes = ["U071"]

    prefer_source = (prefer_source or "outpatient").lower()
    valid_opts = {"outpatient", "inpatient", "drugs_diag", "both", "auto"}
    if prefer_source not in valid_opts:
        raise ValueError("prefer_source мора да биде една од: "
                         "'outpatient', 'inpatient', 'drugs_diag', 'both', 'auto'.")

    if prefer_source == "outpatient":
        cohort_df = _from_outpatient(outpatient_path, icd_codes, debug=debug)

    elif prefer_source == "inpatient":
        cohort_df = _from_inpatient(inpatient_path, icd_codes, debug=debug)

    elif prefer_source == "drugs_diag":
        cohort_df = _from_drugs_diag(icd_codes, debug=debug)

    elif prefer_source == "both":
        op = _from_outpatient(outpatient_path, icd_codes, debug=debug)
        ip = _from_inpatient(inpatient_path, icd_codes, debug=debug)
        cohort_df = _union_ids([op, ip])

    elif prefer_source == "auto":
        if os.path.exists("out/drugs_diag.csv"):
            cohort_df = _from_drugs_diag(icd_codes, debug=debug)
        else:
            cohort_df = _from_outpatient(outpatient_path, icd_codes, debug=debug)

    if cohort_df is None or cohort_df.empty:
        cohort_df = pd.DataFrame({"ENROLID": []})
    else:
        cohort_df = cohort_df.dropna().drop_duplicates()
        cohort_df["ENROLID"] = cohort_df["ENROLID"].astype(str).str.strip()

    print(f"[cohort] ICD filters: {icd_codes} | prefer_source='{prefer_source}' -> n={len(cohort_df)} пациенти")

    os.makedirs("out", exist_ok=True)
    cohort_df.to_csv("out/cohort_ids.csv", index=False)
    return cohort_df

if __name__ == "__main__":
    # df = build_cohort(
    #     inpatient_path="out/I_clean.csv",
    #     outpatient_path="out/O_clean.csv",
    #     pharma_path="out/D_clean.csv",
    #     icd_codes=["U071"],
    #     prefer_source="outpatient",
    #     debug=False
    # )
    # print(df.head(), "\nTotal:", len(df))

    df2 = build_cohort(
        inpatient_path="out/I_clean.csv",
        outpatient_path="out/O_clean.csv",
        pharma_path="out/D_clean.csv",
        icd_codes=["I10"],
        prefer_source="both",
        debug=True
    )
    print(df2.head(), "\nTotal:", len(df2))
