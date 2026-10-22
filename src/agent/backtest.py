"""Stage 2: backtest the assignee suggestion (src/agent/backtest.py).

For recent issues that were fixed by a merged human PR, rank developers using ONLY history
available before the issue was created, then check whether the real fixer is in the top K.
Compared with a baseline that always suggests the most prolific fixers up to that date.

Run from the project root (Neo4j must be running; use NEO4J_URI if not localhost):
    py src/agent/backtest.py [--n 150]
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools import IssueIndex  # noqa: E402

load_dotenv(ROOT / ".env", override=True)
driver = GraphDatabase.driver(os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                              auth=("neo4j", os.getenv("NEO4J_PASSWORD", "").strip()))


def q(cypher, **params):
    with driver.session() as s:
        return [r.data() for r in s.run(cypher, **params)]


def main(n):
    ix = IssueIndex(q)
    fixes = ix.fixes.merge(ix.issues[["created"]], left_on="number", right_index=True)
    fixes = fixes.sort_values("created").drop_duplicates("number")
    targets = fixes.tail(n)
    hits = {1: 0, 3: 0, 5: 0}
    base = {1: 0, 3: 0, 5: 0}
    used = 0
    for _, t in targets.iterrows():
        ranked = ix.rank_assignees(int(t["number"]), k=5, as_of=t["created"])
        if not ranked:
            continue
        used += 1
        names = [r["developer"] for r in ranked]
        prior = ix.fixes[ix.fixes["merged"] < t["created"]]["fixer"].value_counts().index.tolist()
        for k in hits:
            hits[k] += t["fixer"] in names[:k]
            base[k] += t["fixer"] in prior[:k]
    if not used:
        print("Not enough history to backtest.")
        return
    print(f"Backtest on {used} issues (history before each issue only)")
    print("top-K   model   baseline (most prolific fixers)")
    for k in (1, 3, 5):
        print(f"  {k}     {hits[k] / used:5.0%}     {base[k] / used:5.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    main(ap.parse_args().n)