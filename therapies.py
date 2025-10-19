# therapies.py

import os
from typing import Optional, Sequence, List, Tuple, Dict, Set
import pandas as pd

SEPSIS_THERAPIES: Dict[str, List[float]] = {
    "IV_Fluids_Electrolytes": [6.0, 11.0, 16.0, 17.0, 29.0],
    "Cardio_Vasoactive":      [59.0, 60.0, 69.0, 70.0],
    "Corticosteroids":        [46.0, 52.0, 53.0],
    "AntiInfectives":         [166.0, 173.0, 174.0],
    "Anticoag_or_Heme":       [128.0],
    "Endocrine_Metabolic":    [160.0, 162.0],
    "Analgesics_or_Others":   [51.0, 47.0, 62.0, 68.0, 75.0],
}
SEPSIS_ALLOWED_THERCLS: Set[float] = set(
    cls for _label, classes in SEPSIS_THERAPIES.items() for cls in classes
)

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _label_topk(classes_counts: pd.Series, top_k: Optional[int]) -> List[Tuple[float, str]]:
    idx = classes_counts.index if top_k is None else classes_counts.index[:top_k]
    alphabet = [chr(i) for i in range(ord("A"), ord("Z")+1)]
    return [(float(cls), (alphabet[i] if i < len(alphabet) else f"C{i+1}")) for i, cls in enumerate(idx)]

def _discover_drug_name_columns(df: pd.DataFrame) -> List[str]:
    candidates: List[str] = []
    wanted = {"PRODNME", "PRODNAME", "PRODUCT", "GNIND", "GNN", "GNNME", "GENERIC", "DRUGNAME", "DRUG_NAME"}
    upper_cols = {c.upper(): c for c in df.columns}
    for u in wanted:
        if u in upper_cols:
            candidates.append(upper_cols[u])
    if not candidates:
        candidates = [c for c in df.columns if any(k in c.upper() for k in ["NAME", "PROD", "GEN"])]
    return candidates

def select_therapies(
    cohort_df: pd.DataFrame,
    dataset_d_path: str = "out/D_clean.csv",
    thergrp_targets: Optional[Sequence[float]] = None,
    top_k_classes: Optional[int] = 1,
    min_per_arm: Optional[int] = None,
    index_strategy: str = "first",
    debug: bool = False,
    include_thercls: Optional[Sequence[float]] = None,
    sepsis_only: bool = False,
    rest_as_B: bool = True,
    rest_min_per_arm: Optional[int] = 2,
    discovery_min_n: Optional[int] = None,
    pool_across_groups: bool = False,
) -> pd.DataFrame:

    D = pd.read_csv(dataset_d_path, low_memory=False, dtype={"ENROLID": str, "NDCNUM": str})
    D["THERGRP"] = _to_num(D.get("THERGRP"))
    D["THERCLS"] = _to_num(D.get("THERCLS"))

    cohort_ids = set(cohort_df["ENROLID"].astype(str).unique())
    if debug:
        print(f"[therapies] cohort_ids: {len(cohort_ids)}")

    Din = D[D["ENROLID"].isin(cohort_ids)][["ENROLID", "THERGRP", "THERCLS"]].copy()
    Din = Din.dropna(subset=["ENROLID", "THERGRP"])  # барај барем валиден THERGRP
    if debug:
        uniq_pat = Din["ENROLID"].nunique()
        print(f"[therapies] Din пред THERCLS филтри: {uniq_pat} уникатни пациенти | rows={len(Din)}")

    if thergrp_targets is not None:
        tgt = set(float(g) for g in thergrp_targets)
        Din = Din[Din["THERGRP"].isin(tgt)].copy()

    allowed: Optional[Set[float]] = None

    if include_thercls is not None:
        allowed = set(float(x) for x in include_thercls)

    if sepsis_only:
        base = set(SEPSIS_ALLOWED_THERCLS)
        allowed = (allowed | base) if allowed is not None else set(base)

    if discovery_min_n is not None and discovery_min_n > 0:
        cls_counts = Din.dropna(subset=["THERCLS"]).groupby("THERCLS")["ENROLID"].nunique()
        discovered = set(float(c) for c, n in cls_counts.items() if n >= discovery_min_n)
        if allowed is not None:
            before = len(allowed)
            allowed |= discovered
            if debug:
                print(f"[therapies] discovery_min_n={discovery_min_n} -> додадени {len(allowed)-before} класи")
        else:
            allowed = discovered
            if debug:
                print(f"[therapies] discovery_min_n={discovery_min_n} -> иницијални {len(allowed)} класи")

    if allowed is not None:
        known = Din[Din["THERCLS"].isin(allowed)]
        sepsis_grps = set(known["THERGRP"].dropna().unique())

        rest_nan = Din[Din["THERGRP"].isin(sepsis_grps) & Din["THERCLS"].isna()]
        Din = pd.concat([known, rest_nan], ignore_index=True)

        if debug:
            print(f"[therapies] THERCLS allowed -> {sorted(allowed)} | rows={len(Din)} | уникатни пациенти={Din['ENROLID'].nunique()}")

    if Din.empty:
        print("[therapies] Празно: нема групи/класи што ги задоволуваат условите.")
        out = pd.DataFrame(columns=["ENROLID", "THERGRP", "THERCLS", "therapy"])
        os.makedirs("out", exist_ok=True)
        out.to_csv("out/therapy_cohort.csv", index=False)
        return out

    if debug:
        grp_patients = Din.groupby("THERGRP")["ENROLID"].nunique().sort_values(ascending=False)
        print("[therapies] Patients by THERGRP:\n", grp_patients.head(50))

    selections: List[pd.DataFrame] = []

    if pool_across_groups:
        cls_counts_all = Din.dropna(subset=["THERCLS"]).groupby("THERCLS")["ENROLID"].nunique().sort_values(ascending=False)
        if cls_counts_all.empty:
            print("[therapies] Празно: нема класи за глобално pooled селекција.")
            out = pd.DataFrame(columns=["ENROLID", "THERGRP", "THERCLS", "therapy"])
            os.makedirs("out", exist_ok=True)
            out.to_csv("out/therapy_cohort.csv", index=False)
            return out

        labeled = _label_topk(cls_counts_all, top_k_classes)
        parts = []
        for cls, lbl in labeled:
            tmp = (
                Din[Din["THERCLS"] == cls][["ENROLID", "THERGRP", "THERCLS"]]
                .drop_duplicates(subset=["ENROLID", "THERGRP", "THERCLS"])
                .assign(therapy=lbl)
            )
            parts.append(tmp)

        sel = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        if rest_as_B and top_k_classes == 1:
            top1_cls = labeled[0][0]
            rest = (
                Din[(Din["THERCLS"] != top1_cls) | (Din["THERCLS"].isna())][["ENROLID","THERGRP","THERCLS"]]
                .drop_duplicates(subset=["ENROLID", "THERGRP", "THERCLS"])
                .assign(therapy="B")
            )
            sel = pd.concat([sel, rest], ignore_index=True)

            if rest_min_per_arm is not None:
                nA = sel.loc[sel["therapy"] == "A", "ENROLID"].nunique()
                nB = sel.loc[sel["therapy"] == "B", "ENROLID"].nunique()
                if (nB < rest_min_per_arm) or (min_per_arm is not None and nA < min_per_arm):
                    if debug:
                        print(f"[therapies] Skip pooled (A={nA}, B={nB} (rest) не го минува прагот)")
                    sel = pd.DataFrame(columns=["ENROLID","THERGRP","THERCLS","therapy"])

        if not sel.empty:
            selections.append(sel)

    else:
        for g, gframe in Din.groupby("THERGRP"):
            cls_counts = gframe.dropna(subset=["THERCLS"]).groupby("THERCLS")["ENROLID"].nunique().sort_values(ascending=False)
            if cls_counts.empty:
                continue

            labeled = _label_topk(cls_counts, top_k_classes)
            parts = []

            for cls, lbl in labeled:
                tmp = (
                    gframe[gframe["THERCLS"] == cls][["ENROLID", "THERGRP", "THERCLS"]]
                    .drop_duplicates(subset=["ENROLID", "THERGRP", "THERCLS"])
                    .assign(therapy=lbl)
                )
                parts.append(tmp)

            sel = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

            if rest_as_B and top_k_classes == 1:
                top1_cls = labeled[0][0]
                rest = (
                    gframe[(gframe["THERCLS"] != top1_cls) | (gframe["THERCLS"].isna())][["ENROLID","THERGRP","THERCLS"]]
                    .drop_duplicates(subset=["ENROLID", "THERGRP", "THERCLS"])
                    .assign(therapy="B")
                )
                sel = pd.concat([sel, rest], ignore_index=True)

                if rest_min_per_arm is not None:
                    nA = sel.loc[sel["therapy"] == "A", "ENROLID"].nunique()
                    nB = sel.loc[sel["therapy"] == "B", "ENROLID"].nunique()
                    if (nB < rest_min_per_arm) or (min_per_arm is not None and nA < min_per_arm):
                        if debug:
                            print(f"[therapies] Skip THERGRP={g} (A={nA}, B={nB} < прагот)")
                        continue

            if (min_per_arm is not None) and (top_k_classes is not None) and (top_k_classes >= 2):
                n_by_lbl = sel.groupby("therapy")["ENROLID"].nunique()
                if (n_by_lbl < min_per_arm).any():
                    if debug:
                        print(f"[therapies] Skip THERGRP={g} (min_per_arm не задоволено)")
                    continue

            if not sel.empty:
                selections.append(sel)

    if not selections:
        print("[therapies] Празно: нема групи/класи што ги задоволуваат условите.")
        out = pd.DataFrame(columns=["ENROLID", "THERGRP", "THERCLS", "therapy"])
        os.makedirs("out", exist_ok=True)
        out.to_csv("out/therapy_cohort.csv", index=False)
        return out

    therapy_all = pd.concat(selections, ignore_index=True)

    if index_strategy == "first":
        priorities = {lbl: i for i, lbl in enumerate(sorted(therapy_all["therapy"].unique()))}
        therapy_all = (
            therapy_all.assign(_pri=therapy_all["therapy"].map(priorities))
                       .sort_values(["ENROLID", "THERGRP", "_pri"])
                       .drop_duplicates(subset=["ENROLID", "THERGRP"], keep="first")
                       .drop(columns=["_pri"])
        )

    os.makedirs("out", exist_ok=True)
    therapy_all.to_csv("out/therapy_cohort.csv", index=False)

    summary = (
        therapy_all.groupby(["THERGRP", "therapy"])["ENROLID"]
                   .nunique()
                   .reset_index()
                   .rename(columns={"ENROLID": "n_patients"})
                   .sort_values(["THERGRP", "therapy"])
    )
    print("[therapies] Selection done.")
    print(summary.to_string(index=False))
    print("Unique patients overall:", therapy_all["ENROLID"].nunique())

    remaining_thercls = (
        therapy_all.dropna(subset=["THERCLS"])["THERCLS"].unique().tolist()
    )
    if debug:
        try:
            import numpy as np
            print("Therapies selected (unique THERCLS):", [np.float64(x) for x in remaining_thercls])
        except Exception:
            print("Therapies selected (unique THERCLS):", remaining_thercls)

    return therapy_all

def discover_classes_for_cohort(
    cohort_ids_csv: str = "out/cohort_ids.csv",
    dataset_d_path: str = "out/D_clean.csv",
    top_n_groups: int = 30,
    top_n_names_per_class: int = 5,
) -> None:

    cohort_df = pd.read_csv(cohort_ids_csv, dtype={"ENROLID": str})
    D = pd.read_csv(dataset_d_path, dtype={"ENROLID": str, "NDCNUM": str}, low_memory=False)

    D = D[D["ENROLID"].isin(set(cohort_df["ENROLID"].astype(str)))].copy()
    D["THERGRP"] = _to_num(D.get("THERGRP"))
    D["THERCLS"] = _to_num(D.get("THERCLS"))

    name_cols = _discover_drug_name_columns(D)
    keep_cols = ["THERGRP", "THERCLS"] + name_cols

    print("\n== Топ THERGRP/THERCLS по број пациенти (top-N групи) ==")
    grp_sizes = (
        D[["THERGRP","ENROLID"]].dropna()
         .groupby("THERGRP")["ENROLID"].nunique()
         .sort_values(ascending=False).head(top_n_groups)
    )
    print(grp_sizes.to_string())

    print("\n== Топ THERGRP/THERCLS комбинации (по број пациенти) ==")
    cls_sizes = (
        D[["THERGRP","THERCLS","ENROLID"]].dropna()
         .groupby(["THERGRP","THERCLS"])["ENROLID"].nunique()
         .sort_values(ascending=False).head(top_n_groups)
    )
    print(cls_sizes.to_string())

    if name_cols:
        print("\n== Топ имиња по класа (за препознавање на SEPSIS-терапии) ==")
        peek = (
            D[keep_cols].dropna(subset=["THERGRP","THERCLS"])
             .groupby(["THERGRP","THERCLS"])[name_cols[0]]
             .agg(lambda s: s.value_counts().head(top_n_names_per_class).index.tolist())
             .reset_index()
             .head(top_n_groups)
        )
        print(peek.to_string(index=False))
    else:
        print("\n[Предупредување] Не се пронајдени колони со имиња на лекови (на пр. PRODNME/GNN).")
        print("Ако твојот сет нема имиња, користи документација за THERGRP/THERCLS за мапирање на класи.")

if __name__ == "__main__":
    discover_classes_for_cohort(
        cohort_ids_csv="out/cohort_ids.csv",
        dataset_d_path="out/D_clean.csv",
        top_n_groups=30,
        top_n_names_per_class=5,
    )

    # cohort_ids = pd.read_csv("out/cohort_ids.csv", dtype={"ENROLID": str})
    # df = select_therapies(
    #     cohort_df=cohort_ids,
    #     dataset_d_path="out/D_clean.csv",
    #     thergrp_targets=None,
    #     top_k_classes=1,
    #     min_per_arm=None,
    #     index_strategy="first",
    #     debug=True,
    #     sepsis_only=True,
    #     rest_as_B=True,
    #     rest_min_per_arm=2,
    #     discovery_min_n=10,
    #     pool_across_groups=False
    # )
    # print(df.head(), "\nTotal rows:", len(df), "| Unique patients:", df["ENROLID"].nunique())
