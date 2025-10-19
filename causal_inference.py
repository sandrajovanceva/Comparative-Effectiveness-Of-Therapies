# causal_inference.py
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm
from lifelines import CoxPHFitter

def smd_table(df, treat_col, covariate_cols, weights=None):

    X = df[covariate_cols].copy()
    t = df[treat_col].astype(int).values

    cat_cols = [c for c in covariate_cols if X[c].dtype == "object"]
    num_cols = [c for c in covariate_cols if c not in cat_cols]

    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    def w_mean(x, w):
        return np.sum(w * x) / np.sum(w)

    def w_var(x, w):
        mu = w_mean(x, w)
        return np.sum(w * (x - mu) ** 2) / np.sum(w)

    results = []
    for col in X.columns:
        x = X[col].values

        if weights is None:
            w1 = np.ones_like(x[t == 1])
            w0 = np.ones_like(x[t == 0])
        else:
            w1 = weights[t == 1]
            w0 = weights[t == 0]

        m1 = w_mean(x[t == 1], w1)
        m0 = w_mean(x[t == 0], w0)
        v1 = w_var(x[t == 1], w1)
        v0 = w_var(x[t == 0], w0)

        sd = np.sqrt((v1 + v0) / 2.0) if (v1 + v0) > 0 else np.nan
        smd = (m1 - m0) / sd if (sd is not None and sd > 0) else np.nan
        results.append({"covariate": col, "mean_t1": m1, "mean_t0": m0, "SMD": np.abs(smd)})

    out = pd.DataFrame(results).sort_values("SMD", ascending=False).reset_index(drop=True)
    return out

def fit_propensity_scores(df, treat_col, covariate_cols, max_iter=300):

    y = df[treat_col].astype(int).values
    X = df[covariate_cols].copy()

    cat_cols = [c for c in covariate_cols if X[c].dtype == "object"]
    num_cols = [c for c in covariate_cols if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), num_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=max_iter, solver="lbfgs")
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X, y)

    ps = pipe.predict_proba(X)[:, 1]
    ps = pd.Series(ps, index=df.index, name="ps")
    return ps, pipe

def stabilized_iptw(treat, ps, trim_q=0.99):

    treat = np.asarray(treat).astype(int)
    ps = np.clip(np.asarray(ps), 1e-6, 1 - 1e-6)

    p_t = treat.mean()
    w = np.where(treat == 1, p_t / ps, (1 - p_t) / (1 - ps))

    upper = np.quantile(w, trim_q)
    w = np.minimum(w, upper)

    return pd.Series(w, name="iptw")

def ps_logit(ps):
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    return np.log(ps / (1 - ps))

def ps_match_1to1(df, treat_col, ps, caliper=0.2):

    from sklearn.neighbors import NearestNeighbors

    logit = ps_logit(ps)
    df_ = df.copy()
    df_["logit_ps"] = logit

    t1 = df_[df_[treat_col] == 1].copy()
    t0 = df_[df_[treat_col] == 0].copy()

    sd_logit = np.std(df_["logit_ps"].values)
    abs_caliper = caliper * sd_logit

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(t0[["logit_ps"]].values)
    dist, idx = nn.kneighbors(t1[["logit_ps"]].values)

    pairs = []
    used_controls = set()
    for i, (d, j) in enumerate(zip(dist.flatten(), idx.flatten())):
        if d <= abs_caliper:
            c_idx = t0.index[j]
            if c_idx in used_controls:
                continue
            used_controls.add(c_idx)
            t_idx = t1.index[i]
            pairs.append((t_idx, c_idx))

    matched_idx = [i for p in pairs for i in p]
    matched = df_.loc[matched_idx].copy()
    matched["match_weight"] = 1.0
    return matched

def weighted_logistic(df, outcome_col, treat_col, weight_col):

    y = df[outcome_col].astype(int)
    X = sm.add_constant(df[treat_col].astype(int))
    w = df[weight_col].values
    model = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=w)
    res = model.fit()
    return res


def weighted_cox(df, time_col, event_col, treat_col, weight_col, extra_covariates=None):

    cols = [time_col, event_col, treat_col, weight_col]
    if extra_covariates:
        cols += extra_covariates

    cph = CoxPHFitter()
    cph.fit(df[cols], duration_col=time_col, event_col=event_col, weights_col=weight_col)
    return cph

def run_iptw_pipeline(df, treat_col, covariate_cols,
                      outcome_binary=None,
                      time_col=None, event_col=None,
                      trim_q=0.99, return_all=False):

    ps, model = fit_propensity_scores(df, treat_col, covariate_cols)
    df = df.copy()
    df["ps"] = ps

    df["w"] = stabilized_iptw(df[treat_col].astype(int).values, df["ps"].values, trim_q=trim_q)

    smd_pre = smd_table(df, treat_col, covariate_cols, weights=None)
    smd_post = smd_table(df, treat_col, covariate_cols, weights=df["w"].values)

    results = {"ps_model": model, "smd_pre": smd_pre, "smd_post": smd_post}

    if outcome_binary is not None:
        logit_res = weighted_logistic(df, outcome_binary, treat_col, "w")
        results["logit"] = logit_res

    if (time_col is not None) and (event_col is not None):
        cph = weighted_cox(df, time_col, event_col, treat_col, "w")
        results["cox"] = cph

    return (df, results) if return_all else results

def run_matching(covariates_df: pd.DataFrame):

    df = covariates_df.copy()

    df["treat"] = (df["therapy"].astype(str).str.upper() == "A").astype(int)

    covariate_cols = []
    for col in ["AGE", "SEX"]:
        if col in df.columns:
            covariate_cols.append(col)
    covariate_cols += [c for c in df.columns if c.startswith("ICD_")]
    covariate_cols += [c for c in df.columns if c.startswith("THERCLS_")]

    if "AGE" in df.columns:
        df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
    if "SEX" in df.columns:
        try:
            df["SEX"] = pd.to_numeric(df["SEX"], errors="coerce")
        except Exception:
            pass

    ps, _ = fit_propensity_scores(df, "treat", covariate_cols)
    df["ps"] = ps
    df["w"] = stabilized_iptw(df["treat"].values, df["ps"].values, trim_q=0.99)

    smd_pre = smd_table(df, "treat", covariate_cols, weights=None).rename(columns={"SMD": "std_diff_before"})
    smd_post = smd_table(df, "treat", covariate_cols, weights=df["w"].values).rename(columns={"SMD": "std_diff_after"})
    balance_df = smd_pre[["covariate", "std_diff_before"]].merge(
        smd_post[["covariate", "std_diff_after"]], on="covariate", how="outer"
    ).fillna(0.0)

    keep_cols = ["ENROLID", "therapy", "THERGRP", "THERCLS", "treat", "w"] + covariate_cols
    keep_cols = [c for c in keep_cols if c in df.columns]
    matched_df = df[keep_cols].copy()

    return matched_df, balance_df
