"""Stage 2: fixed tools for the AI assistant (src/agent/tools.py).

All tools run parameterised Cypher / pandas, so the LLM never writes its own queries and
every answer can be traced back to data. `q` is the dashboard's query helper:
q(cypher, **params) -> list of dicts.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
from metrics import pr_frame, team_monthly, anomalies, BOTS  # noqa: E402

DAY = 86400


def _clean(obj):
    """Make numpy/pandas values JSON-friendly for the LLM."""
    return json.loads(json.dumps(obj, default=lambda o: None if pd.isna(o) else
                                 (o.item() if hasattr(o, "item") else str(o))))


class IssueIndex:
    """TF-IDF similarity over issue text, plus who fixed what (human-authored, merged PRs)."""

    def __init__(self, q):
        df = pd.DataFrame(q("""
            MATCH (i:Issue)
            OPTIONAL MATCH (p:PullRequest)-[:FIXES]->(i)
            OPTIONAL MATCH (d:Developer)-[:OPENED]->(p)
            RETURN i.number AS number, i.title AS title, i.body AS body, i.state AS state,
                   i.createdAt.epochSeconds AS created, p.number AS pr,
                   p.mergedAt.epochSeconds AS merged, d.login AS fixer"""))
        ok = df["pr"].notna() & df["merged"].notna() & df["fixer"].notna() & ~df["fixer"].isin(BOTS)
        self.fixes = df.loc[ok, ["number", "pr", "merged", "fixer"]].reset_index(drop=True)
        iss = df.drop_duplicates("number")[["number", "title", "body", "state", "created"]].reset_index(drop=True)
        self.issues = iss.set_index("number")
        text = (iss["title"].fillna("") + " " + iss["title"].fillna("") + " "
                + iss["body"].fillna("").str.slice(0, 2000))
        self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2,
                                   max_df=0.5, ngram_range=(1, 2))
        self.mat = self.vec.fit_transform(text)
        self.numbers = iss["number"].tolist()
        self.pos = {n: i for i, n in enumerate(self.numbers)}

    def sims(self, number=None, text=None):
        v = self.mat[self.pos[number]] if number is not None else self.vec.transform([text])
        return linear_kernel(v, self.mat).ravel()

    def similar(self, number=None, text=None, k=5, before=None):
        s = self.sims(number, text)
        out = []
        for idx in np.argsort(-s):
            n = self.numbers[idx]
            if n == number or s[idx] <= 0:
                continue
            if before is not None and self.issues.loc[n, "created"] >= before:
                continue
            out.append((n, float(s[idx])))
            if len(out) >= k:
                break
        return out

    def rank_assignees(self, number, k=3, as_of=None, load=None, top_similar=40):
        """Score developers for an issue.

        score = 0.60 * similarity of the issues they fixed to this one
              + 0.25 * recency (exponential decay, 90 day scale)
              - 0.15 * current open-PR load (capped)
        as_of (epoch seconds) restricts evidence to what was known at that time (for backtests).
        """
        load = load or {}
        fx = self.fixes[self.fixes["number"] != number]
        if as_of is not None:
            fx = fx[fx["merged"] < as_of]
        if fx.empty:
            return []
        last_active = fx.groupby("fixer")["merged"].max()
        s = self.sims(number=number)
        ev = fx.assign(sim=[s[self.pos[n]] for n in fx["number"]])
        ev = ev[ev["sim"] > 0].sort_values("sim", ascending=False).head(top_similar)
        if ev.empty:
            return []
        by = ev.groupby("fixer").agg(sim_sum=("sim", "sum"), n=("number", "count"))
        ref = as_of if as_of is not None else float(self.fixes["merged"].max())
        by["sim_score"] = by["sim_sum"] / by["sim_sum"].max()
        by["recency"] = np.exp(-np.clip(ref - last_active.reindex(by.index), 0, None) / DAY / 90)
        by["load"] = [min(load.get(d, 0) / 5.0, 1.0) for d in by.index]
        by["score"] = 0.60 * by["sim_score"] + 0.25 * by["recency"] - 0.15 * by["load"]
        out = []
        for dev, r in by.sort_values("score", ascending=False).head(k).iterrows():
            top = ev[ev["fixer"] == dev].head(3)["number"].tolist()
            out.append({"developer": dev, "score": round(float(r["score"]), 2),
                        "similar_issues_fixed": int(r["n"]),
                        "similarity_component": round(float(r["sim_score"]), 2),
                        "recency_component": round(float(r["recency"]), 2),
                        "open_prs_now": int(load.get(dev, 0)),
                        "evidence_issues": top})
        return out


def build_tools(q):
    from langchain_core.tools import tool

    holder = {}

    def idx():
        if "i" not in holder:
            holder["i"] = IssueIndex(q)
        return holder["i"]

    @tool
    def dev_stats(developer: str) -> dict:
        """Delivery stats for one developer (GitHub login): merged PRs, median PR cycle time in
        days, how their recent PRs compare, PR size, reviews given and issues opened."""
        df = pr_frame(q)
        g = df[df["dev"].str.lower() == developer.lower()].sort_values("merged")
        if g.empty:
            top = df["dev"].value_counts().head(8).index.tolist()
            return {"error": f"No merged PRs found for '{developer}'. Most active developers: {top}"}
        login = g["dev"].iloc[0]
        reviews = q("MATCH (d:Developer {login:$d})-[v:REVIEWED]->() RETURN count(v) AS n", d=login)[0]["n"]
        opened = q("MATCH (d:Developer {login:$d})-[:OPENED]->(i:Issue) RETURN count(i) AS n", d=login)[0]["n"]
        return _clean({
            "developer": login, "merged_prs": len(g),
            "median_cycle_days": round(g["cycle_days"].median(), 1),
            "last_5_median_cycle_days": round(g.tail(5)["cycle_days"].median(), 1),
            "median_pr_size_lines": round(g["size"].median()),
            "median_hours_to_first_review": round(g["review_hours"].median(), 1),
            "reviews_given": reviews, "issues_opened": opened})

    @tool
    def find_anomalies() -> list:
        """Developers whose recent PR cycle time is unusually slow or fast compared with their own
        history (robust z-score), including how much larger recent PRs were. These are prompts
        for a conversation, not judgements."""
        a = anomalies(pr_frame(q))
        if a.empty:
            return [{"message": "No developer has enough merged PRs for a baseline."}]
        return _clean(a.to_dict("records"))

    @tool
    def team_trend(months: int = 6) -> list:
        """Team-level monthly trend: PRs merged, median PR cycle time (days) and median time to
        first review (hours) for the last N months."""
        m = team_monthly(pr_frame(q)).tail(months).round(1).reset_index()
        return _clean(m.to_dict("records"))

    @tool
    def issue_context(number: int) -> dict:
        """Full context for one issue: title, state, labels, who opened it, its fixing PRs
        (with link source 'github' or 'inferred') and timing."""
        rows = q("""
            MATCH (i:Issue {number:$n})
            OPTIONAL MATCH (a:Developer)-[:OPENED]->(i)
            OPTIONAL MATCH (p:PullRequest)-[f:FIXES]->(i)
            OPTIONAL MATCH (pd:Developer)-[:OPENED]->(p)
            RETURN i.title AS title, i.state AS state, i.labels AS labels,
                   i.commentCount AS comments, toString(i.createdAt) AS created,
                   toString(i.closedAt) AS closed, left(i.body, 600) AS body, a.login AS opened_by,
                   collect(DISTINCT {pr: p.number, title: p.title, source: f.source,
                                     author: pd.login, merged: toString(p.mergedAt)}) AS fixing_prs""",
                 n=number)
        if not rows or rows[0]["title"] is None:
            return {"error": f"Issue #{number} not found."}
        r = rows[0]
        r["fixing_prs"] = [p for p in r["fixing_prs"] if p["pr"] is not None]
        return _clean(r)

    @tool
    def similar_issues(number: int = 0, text: str = "", k: int = 5) -> list:
        """Find past issues similar to an existing issue (pass its number) or to free text (pass
        text). Returns similarity, state, and who fixed each one and when."""
        ix = idx()
        if number:
            if number not in ix.pos:
                return [{"error": f"Issue #{number} not found."}]
            hits = ix.similar(number=number, k=k)
        elif text:
            hits = ix.similar(text=text, k=k)
        else:
            return [{"error": "Pass an issue number or some text."}]
        out = []
        for n, sc in hits:
            row = ix.issues.loc[n]
            fx = ix.fixes[ix.fixes["number"] == n]
            days = None
            if len(fx):
                days = round((fx["merged"].iloc[0] - row["created"]) / DAY, 1)
            out.append({"issue": n, "title": row["title"], "state": row["state"],
                        "similarity": round(sc, 2),
                        "fixed_by": fx["fixer"].unique().tolist(), "days_to_fix": days})
        return _clean(out)

    @tool
    def suggest_assignee(issue_number: int, k: int = 3) -> list:
        """Rank developers who could take an issue, based on similar issues they fixed, how
        recently they were active and how many PRs they currently have open. Returns the score
        components and the evidence issues so the suggestion can be checked."""
        ix = idx()
        if issue_number not in ix.pos:
            return [{"error": f"Issue #{issue_number} not found."}]
        load = {r["dev"]: r["n"] for r in q(
            "MATCH (d:Developer)-[:OPENED]->(p:PullRequest {state:'OPEN'}) RETURN d.login AS dev, count(p) AS n")}
        res = ix.rank_assignees(issue_number, k=k, load=load)
        return _clean(res or [{"message": "Not enough history to suggest an assignee."}])

    return [dev_stats, find_anomalies, team_trend, issue_context, similar_issues, suggest_assignee]