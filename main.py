# main.py
import os
import cohort
import therapies
import confounders
import causal_inference
import outcomes
import report

def main():
    os.makedirs("results", exist_ok=True)

    print("\nКреирање cohort од inpatient/outpatient")
    cohort_df = cohort.build_cohort(
        inpatient_path="out/I_clean.csv",
        outpatient_path="out/O_clean.csv",
        pharma_path="out/D_clean.csv",
        icd_codes=["I10"],
        prefer_source="both"
    )
    print("Cohort shape:", cohort_df.shape)

    try:
        import pandas as pd
        Ddbg = pd.read_csv("out/D_clean.csv", dtype={"ENROLID": str}, low_memory=False)
        cohort_ids = set(cohort_df["ENROLID"].astype(str))
        Ddbg_ids = set(Ddbg["ENROLID"].astype(str))
        inter = cohort_ids & Ddbg_ids
        print(f"[debug] Cohort ENROLIDs: {len(cohort_ids)} | D_clean ENROLIDs: {len(Ddbg_ids)} | overlap: {len(inter)}")
        from therapies import SEPSIS_ALLOWED_THERCLS
        Ddbg["THERGRP"] = pd.to_numeric(Ddbg.get("THERGRP"), errors="coerce")
        Ddbg["THERCLS"] = pd.to_numeric(Ddbg.get("THERCLS"), errors="coerce")
        Din = Ddbg[Ddbg["ENROLID"].isin(inter)][["ENROLID","THERGRP","THERCLS"]]
        n_with_grp = Din["ENROLID"].nunique()
        Din_known = Din.dropna(subset=["THERGRP"])  # барем THERGRP
        n_with_known_grp = Din_known["ENROLID"].nunique()
        Din_cls = Din.dropna(subset=["THERCLS"])  # барем THERCLS
        n_with_known_cls = Din_cls["ENROLID"].nunique()
        top_grp = (
            Din_known.groupby("THERGRP")["ENROLID"].nunique().sort_values(ascending=False).head(10)
        )
        top_cls = (
            Din_cls.groupby("THERCLS")["ENROLID"].nunique().sort_values(ascending=False).head(10)
        )
        print(f"[debug] In overlap: any THERGRP={n_with_grp}, known THERGRP={n_with_known_grp}, known THERCLS={n_with_known_cls}")
        print("[debug] Top THERGRP by unique ENROLID:\n" + top_grp.to_string())
        print("[debug] Top THERCLS by unique ENROLID:\n" + top_cls.to_string())
    except Exception as e:
        print(f"[debug] diagnostics skipped due to error: {e}")

    dataset_d_for_therapies = "out/D_clean.csv"
    include_thercls_auto = None
    try:
        import pandas as pd
        Dx = pd.read_csv("out/diagnoses_long.csv", dtype={"ENROLID": str, "ICD": str}, low_memory=False)
        D = pd.read_csv("out/D_clean.csv", dtype={"ENROLID": str, "NDCNUM": str}, low_memory=False)

        Dx["ICD"] = (
            Dx["ICD"].astype(str).str.strip().str.upper().str.replace(".", "", regex=False)
        )
        Dx = Dx[Dx["ICD"].str.startswith("I10", na=False)].copy()
        Dx["DATE"] = pd.to_datetime(Dx.get("DATE"), errors="coerce")
        D["SVCDATE"] = pd.to_datetime(D.get("SVCDATE"), errors="coerce")

        WINDOW_DAYS = int(os.environ.get("I10_LINK_WINDOW_DAYS", 7))
        linked = (
            D[["ENROLID","NDCNUM","THERGRP","THERCLS","SVCDATE"]]
              .merge(Dx[["ENROLID","DATE"]], on="ENROLID", how="inner")
        )
        linked["delta_days"] = (linked["SVCDATE"] - linked["DATE"]).dt.days.abs()
        linked = linked[linked["delta_days"] <= WINDOW_DAYS]

        linked = (
            linked.sort_values(["ENROLID","NDCNUM","SVCDATE","delta_days"])
                  .drop_duplicates(subset=["ENROLID","NDCNUM","SVCDATE"], keep="first")
        )

        if not linked.empty:
            D_linked = linked[["ENROLID","NDCNUM","THERGRP","THERCLS","SVCDATE"]].copy()
            D_linked.to_csv("out/D_i10_linked.csv", index=False)
            dataset_d_for_therapies = "out/D_i10_linked.csv"
            cls_counts = (
                D_linked.dropna(subset=["THERCLS"]).groupby("THERCLS")["ENROLID"].nunique().sort_values(ascending=False)
            )
            MIN_N = 50
            TOP_N = 12
            kept = cls_counts[cls_counts >= MIN_N].head(TOP_N)
            include_thercls_auto = [float(c) for c in kept.index.tolist()]
            print(f"[link] I10-linked rows: {len(D_linked)} | window=±{WINDOW_DAYS}d | THERCLS total={cls_counts.size} | kept>={MIN_N} top{TOP_N} -> {len(include_thercls_auto)}")
        else:
            print("[link] No I10-linked administrations within ±7 days; using full D_clean.")
    except Exception as e:
        print(f"[link] Failed linking D to I10 by date: {e}")

    if not include_thercls_auto:
        include_thercls_auto = [53.0, 46.0, 47.0, 51.0, 52.0, 59.0, 69.0, 189.0, 166.0, 174.0]

    print("\nИзбор на терапии за споредба (пошироки правила)")
    therapy_df = therapies.select_therapies(
        cohort_df=cohort_df,
        dataset_d_path=dataset_d_for_therapies,
        thergrp_targets=None,
        top_k_classes=1,
        min_per_arm=50,
        sepsis_only=False,
        include_thercls=include_thercls_auto,
        debug=True,
        discovery_min_n=None,
        rest_as_B=True,
        rest_min_per_arm=50,
        pool_across_groups=False
    )

    print("Therapies selected (unique THERCLS):", sorted(therapy_df["THERCLS"].unique()))
    print("Therapy cohort shape:", therapy_df.shape)

    print("\нКонструирање на коваријати")
    covariates_df = confounders.build_covariates(
        therapy_df,
        dataset_d_path="out/D_clean.csv",
        dataset_o_path="out/O_clean.csv"
    )
    print("Covariates shape:", covariates_df.shape)

    print("\nCausal inference (IPTW / balancing)")
    matched_df, balance_df = causal_inference.run_matching(covariates_df)
    print("Matched cohort shape:", matched_df.shape)
    balance_df.to_csv("results/balance_smd.csv", index=False)
    print(balance_df.head(15))

    print("\nOutcome анализа (READMIT_30D)")
    results_df = outcomes.compute_outcomes_strict(
        matched_df,
        inpatient_path="out/I_clean.csv",
        outpatient_path="out/O_clean.csv"
    )
    results_df.to_csv("results/final_results.csv", index=False)
    print("Results saved to results/final_results.csv")

    print("\nГенерирање табели и графици")
    print(report.summary_table(results_df, "THERCLS", "READMIT_30D"))
    try:
        import pandas as pd
        cls_counts_results = results_df.groupby("THERCLS")["READMIT_30D"].count().sort_values(ascending=False)
        kept_cls = cls_counts_results[cls_counts_results >= 50].head(12).index.tolist()
        results_df_cls = results_df[results_df["THERCLS"].isin(kept_cls)].copy()
    except Exception:
        results_df_cls = results_df

    report.plot_outcome_distribution(results_df_cls, "THERCLS", "READMIT_30D", "Readmission Rate by Therapy Class (Top)")
    report.plot_covariate_balance(balance_df)
    report.plot_readmit_by_group(results_df, group_col="THERGRP", outcome_col="READMIT_30D")
    report.plot_readmit_by_group(results_df_cls, group_col="THERCLS", outcome_col="READMIT_30D",
                                 title="Readmission Rate by Therapy Class (Top)")

if __name__ == "__main__":
    main()
