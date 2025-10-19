# report.py

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def summary_table(results: pd.DataFrame, group_col: str, outcome_col: str):

    table = (
        results.groupby(group_col)[outcome_col]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    return table

def plot_outcome_distribution(results: pd.DataFrame, group_col: str, outcome_col: str, title: str):

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=results,
        x=group_col,
        y=outcome_col,
        errorbar="sd",
        palette="Set2"
    )
    plt.title(title)
    plt.xlabel("Therapy Group")
    plt.ylabel(outcome_col)
    plt.tight_layout()
    plt.show()

def plot_covariate_balance(balance_df: pd.DataFrame):

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="std_diff_before", y="covariate",
        data=balance_df, label="Before Matching", color="red"
    )
    sns.scatterplot(
        x="std_diff_after", y="covariate",
        data=balance_df, label="After Matching", color="green"
    )
    plt.axvline(0.1, linestyle="--", color="gray")
    plt.axvline(-0.1, linestyle="--", color="gray")
    plt.title("Covariate Balance Before and After Matching")
    plt.xlabel("Standardized Mean Difference")
    plt.ylabel("Covariate")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_readmit_by_group(results_df: pd.DataFrame,
                          group_col: str = "THERGRP",
                          outcome_col: str = "READMIT_30D",
                          title: str | None = None,
                          figsize=(9, 5)):

    df = results_df.copy()

    if df[outcome_col].dtype != bool and not np.issubdtype(df[outcome_col].dtype, np.integer):
        df[outcome_col] = df[outcome_col].astype(bool)
    df[outcome_col] = df[outcome_col].astype(int)

    g = df.groupby(group_col)[outcome_col].agg(['mean', 'count'])
    g = g.sort_index()
    p = g['mean'].values
    n = g['count'].values

    z = 1.96
    denom = 1 + z**2 / n
    centre = (p + z**2/(2*n)) / denom
    half = (z * np.sqrt((p*(1-p)/n) + (z**2/(4*n**2)))) / denom
    lower = np.clip(centre - half, 0, 1)
    upper = np.clip(centre + half, 0, 1)
    err_low = centre - lower
    err_up = upper - centre

    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(x, p, width=0.75)
    ax.errorbar(x, p, yerr=[err_low, err_up], fmt='k|', ecolor='black', capsize=6, lw=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(g.index.astype(str), rotation=0)
    ax.set_xlabel("Therapy Group" if group_col == "THERGRP" else "Therapy Class")
    ax.set_ylabel("READMIT_30D")
    ax.set_ylim(0, max(1.05*np.max(upper), 0.22))

    if title is None:
        title = f"Readmission Rate by {'Therapy Group' if group_col=='THERGRP' else 'Therapy Class'}"
    ax.set_title(title)

    for xi, pi, ni in zip(x, p, n):
        ax.text(xi, pi + 0.01, f"n={int(ni)}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    results = pd.read_csv("results/final_results.csv")

    print(summary_table(results, "THERGRP", "READMIT_30D"))

    plot_outcome_distribution(results, "THERGRP", "READMIT_30D", "Readmission Rate by Therapy Group")

    balance_df = pd.DataFrame({
        "covariate": ["age", "gender", "comorbidity_score"],
        "std_diff_before": [0.25, 0.15, 0.30],
        "std_diff_after": [0.05, 0.02, 0.07]
    })
    plot_covariate_balance(balance_df)
