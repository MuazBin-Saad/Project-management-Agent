import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)
TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
if not TOKEN:
    raise SystemExit("GITHUB_TOKEN not found in .env")
URL = "https://api.github.com/graphql"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

QUERY = """
query($owner:String!, $name:String!, $cursor:String) {
  rateLimit { cost remaining resetAt }
  repository(owner:$owner, name:$name) {
    __KIND__(first:__SIZE__, after:$cursor,
             orderBy:{field:CREATED_AT, direction:ASC}) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes { __FIELDS__ }
    }
  }
}
"""

ISSUE_FIELDS = """
number title body state stateReason createdAt updatedAt closedAt
author { login }
labels(first:10) { nodes { name } }
comments(first:5) { totalCount nodes { createdAt author { login } } }
"""

PR_FIELDS = """
number title body state createdAt updatedAt closedAt mergedAt
author { login }
labels(first:10) { nodes { name } }
closingIssuesReferences(first:10) { nodes { number } }
additions deletions changedFiles
reviews(first:20) { totalCount nodes { author { login } submittedAt state } }
commits(first:100) {
  totalCount
  nodes { commit { oid messageHeadline authoredDate author { user { login } } } }
}
"""

CONFIG = {
    "issues": (ISSUE_FIELDS, 100),
    "pullRequests": (PR_FIELDS, 30),  # smaller pages: nested commits cost more
}


def run_query(query, variables, retries=5):
    for attempt in range(retries):
        try:
            r = requests.post(
                URL,
                json={"query": query, "variables": variables},
                headers=HEADERS,
                timeout=60,
            )
            if r.status_code == 200:
                body = r.json()
                if "errors" not in body:
                    return body["data"]
                print("GraphQL errors:", body["errors"])
            else:
                print(f"HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code == 401:
                    raise SystemExit("401: token rejected, check .env")
        except requests.RequestException as e:
            print("Request failed:", e)
        wait = 2 ** attempt * 5
        print(f"Retrying in {wait}s...")
        time.sleep(wait)
    raise RuntimeError("Query failed after retries")


def wait_if_rate_limited(rate):
    if rate["remaining"] < 100:
        reset = datetime.fromisoformat(rate["resetAt"].replace("Z", "+00:00"))
        secs = max(0, (reset - datetime.now(timezone.utc)).total_seconds()) + 5
        print(f"Rate limit low, sleeping {secs:.0f}s")
        time.sleep(secs)


def fetch_all(owner, name, kind, max_pages=None):
    fields, size = CONFIG[kind]
    query = (
        QUERY.replace("__KIND__", kind)
        .replace("__SIZE__", str(size))
        .replace("__FIELDS__", fields)
    )
    out_dir = Path("data/raw") / f"{owner}__{name}" / kind
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume from the last saved page, if any
    cursor, page = None, 1
    for f in sorted(out_dir.glob("page_*.json")):
        saved = json.loads(f.read_text(encoding="utf-8"))
        info = saved["pageInfo"]
        cursor = info["endCursor"]
        page += 1
        if not info["hasNextPage"]:
            print(f"{kind}: already complete ({page - 1} pages)")
            return

    while max_pages is None or page <= max_pages:
        data = run_query(query, {"owner": owner, "name": name, "cursor": cursor})
        block = data["repository"][kind]
        path = out_dir / f"page_{page:04d}.json"
        path.write_text(json.dumps(block, ensure_ascii=False), encoding="utf-8")

        rate = data["rateLimit"]
        print(
            f"{kind} page {page}: {len(block['nodes'])} items "
            f"(total {block['totalCount']}), cost {rate['cost']}, "
            f"remaining {rate['remaining']}"
        )
        if not block["pageInfo"]["hasNextPage"]:
            break
        cursor = block["pageInfo"]["endCursor"]
        page += 1
        wait_if_rate_limited(rate)
        time.sleep(0.5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", help="owner/name, e.g. pallets/flask")
    ap.add_argument("--max-pages", type=int, default=None)
    args = ap.parse_args()
    owner, name = args.repo.split("/")
    for kind in ("issues", "pullRequests"):
        fetch_all(owner, name, kind, args.max_pages)
    print("Done.")