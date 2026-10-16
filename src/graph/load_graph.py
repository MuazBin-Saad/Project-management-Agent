import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)
PASSWORD = os.getenv("NEO4J_PASSWORD", "").strip()

BOTS = {"dependabot", "dependabot-preview", "pre-commit-ci"}
CLOSING = re.compile(r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s*#(\d+)", re.I)
ANYREF = re.compile(r"#(\d+)")


def load(repo_dir, kind):
    items = []
    for f in sorted((repo_dir / kind).glob("page_*.json")):
        items += json.loads(f.read_text(encoding="utf-8"))["nodes"]
    return items


def login(node):
    return (node or {}).get("login")


def build(repo_dir):
    issues = load(repo_dir, "issues")
    prs = load(repo_dir, "pullRequests")
    issue_nums = {i["number"] for i in issues}
    pr_nums = {p["number"] for p in prs}

    issue_rows = [{
        "number": i["number"], "title": i["title"], "body": i["body"] or "",
        "state": i["state"], "stateReason": i.get("stateReason"),
        "createdAt": i["createdAt"], "closedAt": i["closedAt"],
        "labels": [l["name"] for l in i["labels"]["nodes"]],
        "commentCount": i["comments"]["totalCount"],
        "author": login(i["author"]),
    } for i in issues]

    pr_rows, commit_rows, seen_commits = [], [], set()
    fixes, mentions, refs = {}, set(), set()

    for p in prs:
        author = login(p["author"])
        pr_rows.append({
            "number": p["number"], "title": p["title"], "body": p["body"] or "",
            "state": p["state"], "createdAt": p["createdAt"],
            "closedAt": p["closedAt"], "mergedAt": p["mergedAt"],
            "labels": [l["name"] for l in p["labels"]["nodes"]],
            "author": author,
        })

        for c in p["commits"]["nodes"]:
            c = c["commit"]
            if c["oid"] in seen_commits:
                continue
            seen_commits.add(c["oid"])
            commit_rows.append({
                "oid": c["oid"], "message": c["messageHeadline"],
                "date": c["authoredDate"],
                "author": login((c["author"] or {}).get("user")),
                "pr": p["number"],
            })

        # official links (any author)
        for n in p["closingIssuesReferences"]["nodes"]:
            fixes[(p["number"], n["number"])] = "github"

        # text-inferred links (skip bots)
        if author in BOTS:
            continue
        text = (p["title"] or "") + "\n" + (p["body"] or "")
        fixed = {int(n) for n in CLOSING.findall(text)}
        for n in fixed:
            if n in issue_nums:
                fixes.setdefault((p["number"], n), "inferred")
        for n in {int(n) for n in ANYREF.findall(text)} - fixed:
            if n == p["number"]:
                continue
            if n in issue_nums and (p["number"], n) not in fixes:
                mentions.add((p["number"], n))
            elif n in pr_nums:
                refs.add((p["number"], n))

    return (issue_rows, pr_rows, commit_rows,
            [{"pr": a, "issue": b, "source": s} for (a, b), s in fixes.items()],
            [{"pr": a, "issue": b} for a, b in mentions],
            [{"pr": a, "other": b} for a, b in refs])


def batched(session, query, rows, size=500):
    for i in range(0, len(rows), size):
        session.run(query, rows=rows[i:i + size]).consume()


def main(repo_dir):
    issues, prs, commits, fixes, mentions, refs = build(repo_dir)
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", PASSWORD))
    with driver.session() as s:
        for q in [
            "CREATE CONSTRAINT issue_num IF NOT EXISTS FOR (n:Issue) REQUIRE n.number IS UNIQUE",
            "CREATE CONSTRAINT pr_num IF NOT EXISTS FOR (n:PullRequest) REQUIRE n.number IS UNIQUE",
            "CREATE CONSTRAINT commit_oid IF NOT EXISTS FOR (n:Commit) REQUIRE n.oid IS UNIQUE",
            "CREATE CONSTRAINT dev_login IF NOT EXISTS FOR (n:Developer) REQUIRE n.login IS UNIQUE",
        ]:
            s.run(q).consume()

        batched(s, """
            UNWIND $rows AS r
            MERGE (i:Issue {number: r.number})
            SET i.title = r.title, i.body = r.body, i.state = r.state,
                i.stateReason = r.stateReason, i.labels = r.labels,
                i.commentCount = r.commentCount,
                i.createdAt = datetime(r.createdAt), i.closedAt = datetime(r.closedAt)
            WITH i, r WHERE r.author IS NOT NULL
            MERGE (d:Developer {login: r.author})
            MERGE (d)-[:OPENED]->(i)""", issues)

        batched(s, """
            UNWIND $rows AS r
            MERGE (p:PullRequest {number: r.number})
            SET p.title = r.title, p.body = r.body, p.state = r.state,
                p.labels = r.labels, p.createdAt = datetime(r.createdAt),
                p.closedAt = datetime(r.closedAt), p.mergedAt = datetime(r.mergedAt)
            WITH p, r WHERE r.author IS NOT NULL
            MERGE (d:Developer {login: r.author})
            MERGE (d)-[:OPENED]->(p)""", prs)

        batched(s, """
            UNWIND $rows AS r
            MATCH (p:PullRequest {number: r.pr})
            MERGE (c:Commit {oid: r.oid})
            SET c.message = r.message, c.date = datetime(r.date)
            MERGE (c)-[:PART_OF]->(p)
            WITH c, r WHERE r.author IS NOT NULL
            MERGE (d:Developer {login: r.author})
            MERGE (d)-[:AUTHORED]->(c)""", commits)

        batched(s, """
            UNWIND $rows AS r
            MATCH (p:PullRequest {number: r.pr}), (i:Issue {number: r.issue})
            MERGE (p)-[f:FIXES]->(i) SET f.source = r.source""", fixes)

        batched(s, """
            UNWIND $rows AS r
            MATCH (p:PullRequest {number: r.pr}), (i:Issue {number: r.issue})
            MERGE (p)-[:MENTIONS]->(i)""", mentions)

        batched(s, """
            UNWIND $rows AS r
            MATCH (a:PullRequest {number: r.pr}), (b:PullRequest {number: r.other})
            MERGE (a)-[:REFERENCES]->(b)""", refs)

        print("Node counts:")
        for rec in s.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC"):
            print(f"  {rec['label']}: {rec['n']}")
        print("Relationship counts:")
        for rec in s.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n ORDER BY n DESC"):
            print(f"  {rec['t']}: {rec['n']}")
    driver.close()


if __name__ == "__main__":
    main(ROOT / "data" / "raw" / "pallets__flask")