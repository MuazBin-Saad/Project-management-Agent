"""Stage 1c: PM metrics (new file: src/dashboard/metrics.py).

Pure pandas on top of Neo4j query results. `q` is the dashboard's query helper:
q(cypher, **params) -> list of dicts.
"""
import numpy as np
import pandas as pd

BOTS = {"dependabot", "dependabot-preview", "pre-commit-ci"}


def pr_frame(q):
    """One row per merged, human-authored PR with cycle time, size and first-review time."""
    rows = q("""
        MATCH (d:Developer)-[:OPENED]->(p:PullRequest)
        WHERE p.mergedAt IS NOT NULL
        OPTIONAL MATCH (rv:Developer)-[v:REVIEWED]->(p) WHERE rv <> d
        WITH d, p, min(v.at) AS firstReview
        RETURN d.login AS dev, p.number AS pr, p.title AS title,
               p.createdAt.epochSeconds AS created, p.mergedAt.epochSeconds AS merged,
               coalesce(p.additions, 0) + coalesce(p.deletions, 0) AS size,
               firstReview.epochSeconds AS review_ts""")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[~df["dev"].isin(BOTS)].copy()
    df["cycle_days"] = (df["merged"] - df["created"]) / 86400
    df["review_hours"] = (df["review_ts"] - df["created"]) / 3600
    df["month"] = pd.to_datetime(df["merged"], unit="s").dt.to_period("M").astype(str)
    return df


def team_monthly(df):
    """Efficiency/flow, communication and activity per month."""
    return df.groupby("month").agg(
        prs_merged=("pr", "count"),
        median_cycle_days=("cycle_days", "median"),
        median_first_review_hours=("review_hours", "median"),
    )


def issue_response_monthly(q):
    """Median hours from issue creation to first reply by someone else, per month."""
    df = pd.DataFrame(q("""
        MATCH (i:Issue) WHERE i.firstResponseAt IS NOT NULL
        RETURN i.createdAt.epochSeconds AS c, i.firstResponseAt.epochSeconds AS r"""))
    if df.empty:
        return df
    df["hours"] = (df["r"] - df["c"]) / 3600
    df["month"] = pd.to_datetime(df["c"], unit="s").dt.to_period("M").astype(str)
    return df.groupby("month")["hours"].median().rename("median_first_response_hours")


def anomalies(df, min_prs=12, recent_n=5, z_flag=2.0):
    """Compare each dev's last N merged PRs to their own earlier history.

    Uses log(1 + cycle_days) and a robust z-score (median / MAD). Also reports how much
    bigger or smaller the recent PRs were, since large PRs are naturally slower.
    """
    out = []
    for dev, g in df.sort_values("merged").groupby("dev"):
        if len(g) < min_prs:
            continue
        recent, base = g.tail(recent_n), g.iloc[:-recent_n]
        lb, lr = np.log1p(base["cycle_days"]), np.log1p(recent["cycle_days"])
        mad = max(1.4826 * np.median(np.abs(lb - np.median(lb))), 0.25)
        z = (np.median(lr) - np.median(lb)) / mad
        size_ratio = (recent["size"].median() + 1) / (base["size"].median() + 1)
        status = "normal"
        if z >= z_flag:
            status = "slower than usual"
            if size_ratio >= 2:
                status += " (but PRs were larger)"
        elif z <= -z_flag:
            status = "faster than usual"
        out.append({
            "developer": dev, "merged_prs": len(g),
            "usual_days": round(base["cycle_days"].median(), 1),
            "recent_days": round(recent["cycle_days"].median(), 1),
            "z_score": round(z, 1), "size_ratio": round(size_ratio, 1), "status": status})
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_values("z_score", ascending=False)