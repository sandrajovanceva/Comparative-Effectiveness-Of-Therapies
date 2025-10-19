# outcomes.py
import pandas as pd
import numpy as np

def _norm_enrolid(x: pd.Series) -> pd.Series:
    s = x.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.replace(r"[^\d]", "", regex=True)
    return s

def _normalize_icd_cols(df: pd.DataFrame) -> pd.DataFrame:
    dx_cols = [c for c in df.columns if c.upper().startswith("DX")]
    if not dx_cols: return df
    for c in dx_cols:
        df[c] = (df[c].astype(str)
                      .str.strip()
                      .str.replace(".", "", regex=False)
                      .str.upper()
                      .replace({"": np.nan, "NAN": np.nan, "NONE": np.nan, "0": np.nan, "0.0": np.nan}))
    return df

def define_outcomes_for_ids(I: pd.DataFrame, O: pd.DataFrame, target_ids: pd.Series) -> pd.DataFrame:
    I = I.copy(); O = O.copy()
    I["ENROLID"] = _norm_enrolid(I["ENROLID"])
    O["ENROLID"] = _norm_enrolid(O["ENROLID"])

    target_ids = _norm_enrolid(pd.Series(target_ids))
    I = I[I["ENROLID"].isin(set(target_ids))]
    O = O[O["ENROLID"].isin(set(target_ids))]

    ad_col = "ADMITDATE" if "ADMITDATE" in I.columns else ("ADMDATE" if "ADMDATE" in I.columns else None)
    ds_col = "DISCHDATE" if "DISCHDATE" in I.columns else ("DISDATE" if "DISDATE" in I.columns else None)

    keep_cols = ["ENROLID"]
    if ad_col: keep_cols.append(ad_col)
    if ds_col: keep_cols.append(ds_col)
    if "DX1" in I.columns: keep_cols.append("DX1")

    admissions = I[keep_cols].copy()
    if ad_col: admissions["ADMITDATE"] = pd.to_datetime(admissions[ad_col], errors="coerce")
    if ds_col: admissions["DISDATE"]   = pd.to_datetime(admissions[ds_col], errors="coerce")

    admissions = admissions.sort_values(["ENROLID", "ADMITDATE"])
    admissions["NEXT_ADMIT"] = admissions.groupby("ENROLID")["ADMITDATE"].shift(-1)

    delta = (admissions["NEXT_ADMIT"] - admissions["DISDATE"]).dt.days
    admissions["READMIT_30D"] = (delta >= 0) & (delta <= 30)

    readmit = (admissions.groupby("ENROLID")["READMIT_30D"]
               .max()
               .reset_index()
               .astype({"READMIT_30D": bool}))
    all_ids = pd.DataFrame({"ENROLID": target_ids.unique()})
    readmit = all_ids.merge(readmit, on="ENROLID", how="left").fillna({"READMIT_30D": False})

    O = _normalize_icd_cols(O)
    dx_cols = [c for c in O.columns if c.upper().startswith("DX")]
    if dx_cols:
        long_dx = (O.melt(id_vars=["ENROLID"], value_vars=dx_cols,
                          var_name="DX_pos", value_name="ICD")
                     .dropna(subset=["ICD"]))
        complications_dx = {
            "E11": "Diabetes complication",
            "I21": "Acute MI"
        }
        long_dx["complication"] = np.nan
        for pref, label in complications_dx.items():
            long_dx.loc[long_dx["ICD"].str.startswith(pref), "complication"] = label

        comp = (long_dx.dropna(subset=["complication"])
                        .groupby("ENROLID")["complication"]
                        .agg(lambda x: sorted(set(x)))
                        .reset_index())
    else:
        comp = pd.DataFrame({"ENROLID": [], "complication": []})

    outcomes = readmit.merge(comp, on="ENROLID", how="left")
    return outcomes

def attach_therapy_info(outcomes: pd.DataFrame, D: pd.DataFrame) -> pd.DataFrame:
    out = outcomes.copy(); D = D.copy()
    out["ENROLID"] = _norm_enrolid(out["ENROLID"])
    D["ENROLID"] = _norm_enrolid(D["ENROLID"])

    for col in ["THERGRP","THERCLS"]:
        if col in D.columns:
            D[col] = pd.to_numeric(D[col], errors="coerce")

    d_keep = D[["ENROLID","THERGRP","THERCLS"]].dropna(subset=["ENROLID"])
    thergrp = (d_keep.dropna(subset=["THERGRP"])
                     .groupby("ENROLID")["THERGRP"]
                     .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
                     .reset_index())
    thercls = (d_keep.dropna(subset=["THERCLS"])
                     .groupby("ENROLID")["THERCLS"]
                     .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
                     .reset_index())

    out = out.merge(thergrp, on="ENROLID", how="left")
    out = out.merge(thercls, on="ENROLID", how="left")

    coverage = ((out["THERGRP"].notna()) | (out["THERCLS"].notna())).mean()*100
    print(f"[attach_therapy_info] Coverage: {coverage:.1f}%")

    return out

def compute_outcomes_strict(matched_df: pd.DataFrame,
                            inpatient_path: str = "out/I_clean.csv",
                            outpatient_path: str = "out/O_clean.csv",
                            d_clean_path: str = "out/D_clean.csv",
                            attach_therapy: bool = False,
                            verbose: bool = True) -> pd.DataFrame:
    I = pd.read_csv(inpatient_path, dtype={"ENROLID": str}, low_memory=False)
    O = pd.read_csv(outpatient_path, dtype={"ENROLID": str}, low_memory=False)

    ids = _norm_enrolid(matched_df["ENROLID"])
    if verbose:
        print(f"[outcomes] matched_df shape={matched_df.shape}, unique ENROLID={ids.nunique()}")

    I_ids = set(_norm_enrolid(I["ENROLID"]).unique())
    O_ids = set(_norm_enrolid(O["ENROLID"]).unique())
    ids_set = set(ids.unique())
    if verbose:
        print(f"[outcomes] Intersections: matched∩I={len(ids_set & I_ids)}, matched∩O={len(ids_set & O_ids)}")

    outs = define_outcomes_for_ids(I, O, ids)
    if verbose:
        print(f"[outcomes] outs shape={outs.shape} (колони: {list(outs.columns)})")
        print(outs.head(5))

    tmp = matched_df.copy()
    tmp["ENROLID"] = _norm_enrolid(tmp["ENROLID"])
    res = tmp.merge(outs, on="ENROLID", how="left")

    if "READMIT_30D" in res.columns:
        res["READMIT_30D"] = res["READMIT_30D"].fillna(False).astype(bool)

    if attach_therapy:
        D = pd.read_csv(d_clean_path, dtype={"ENROLID": str}, low_memory=False)
        if verbose:
            print(f"[outcomes] D_clean shape={D.shape}")
        res = attach_therapy_info(res, D)

    if verbose:
        print(f"[outcomes] final results shape={res.shape}")
        print(res.head(10))

    return res

