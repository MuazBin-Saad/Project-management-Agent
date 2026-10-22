"""Stage 1d: Streamlit dashboard with the new PM metrics tab (src/dashboard/app.py)."""
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pyvis.network import Network
from chat_tab import render as render_chat

from metrics import pr_frame, team_monthly, issue_response_monthly, anomalies

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)

st.set_page_config(page_title="Project Insight", layout="wide")


@st.cache_resource
def driver():
    pw = os.getenv("NEO4J_PASSWORD", "").strip()
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    return GraphDatabase.driver(uri, auth=("neo4j", pw))


def q(cypher, **params):
    with driver().session() as s:
        return [r.data() for r in s.run(cypher, **params)]


st.title("Project Insight: pallets/flask")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Traceability", "Problem detection", "PM metrics", "AI assistant"])

# ---------- Overview ----------
with tab1:
    counts = {r["label"]: r["n"] for r in q(
        "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n")}
    cols = st.columns(4)
    for col, label in zip(cols, ["Issue", "PullRequest", "Commit", "Developer"]):
        col.metric(label, f"{counts.get(label, 0):,}")

    rows = q("""MATCH (i:Issue)
                RETURN i.createdAt.year AS y, i.createdAt.month AS m, count(*) AS n
                ORDER BY y, m""")
    df = pd.DataFrame(rows)
    df["month"] = df["y"].astype(str) + "-" + df["m"].astype(str).str.zfill(2)
    st.subheader("Issues opened per month")
    st.bar_chart(df.set_index("month")["n"])

    st.subheader("Link sources (traceability quality)")
    st.dataframe(pd.DataFrame(q(
        "MATCH ()-[f:FIXES]->() RETURN f.source AS source, count(*) AS links")),
        hide_index=True)

# ---------- Traceability ----------
with tab2:
    linked = q("""MATCH (p:PullRequest)-[:FIXES]->(i:Issue)
                  RETURN DISTINCT i.number AS n, i.title AS t ORDER BY n DESC""")
    options = {f"#{r['n']}  {r['t'][:70]}": r["n"] for r in linked}
    choice = st.selectbox("Issue (only issues with a fixing PR)", list(options))
    n = options[choice]

    rows = q("""MATCH (i:Issue {number:$n})
                OPTIONAL MATCH (p:PullRequest)-[f:FIXES]->(i)
                OPTIONAL MATCH (c:Commit)-[:PART_OF]->(p)
                OPTIONAL MATCH (d:Developer)-[:OPENED]->(p)
                RETURN i.title AS it, p.number AS pr, p.title AS pt,
                       f.source AS src, c.oid AS oid, c.message AS msg, d.login AS dev""",
             n=n)

    net = Network(height="520px", width="100%", cdn_resources="in_line")
    net.add_node(f"I{n}", label=f"Issue #{n}", title=rows[0]["it"], color="#e74c3c", size=28)
    for r in rows:
        if r["pr"] is None:
            continue
        net.add_node(f"P{r['pr']}", label=f"PR #{r['pr']}", title=r["pt"], color="#3498db", size=22)
        net.add_edge(f"P{r['pr']}", f"I{n}", label=r["src"] or "")
        if r["dev"]:
            net.add_node(f"D{r['dev']}", label=r["dev"], color="#2ecc71", shape="box")
            net.add_edge(f"D{r['dev']}", f"P{r['pr']}", label="opened")
        if r["oid"]:
            net.add_node(f"C{r['oid']}", label=r["oid"][:7], title=r["msg"], color="#f39c12", size=12)
            net.add_edge(f"C{r['oid']}", f"P{r['pr']}", label="part of")
    components.html(net.generate_html(), height=540)
    st.caption("Red = issue, blue = PR, orange = commit, green = developer. Hover for details.")

# ---------- Problem detection ----------
with tab3:
    st.subheader("Oldest open issues with no fixing PR")
    st.dataframe(pd.DataFrame(q("""
        MATCH (i:Issue {state:"OPEN"}) WHERE NOT ()-[:FIXES]->(i)
        RETURN i.number AS issue, i.title AS title,
               duration.inDays(i.createdAt, datetime()).days AS days_open,
               i.commentCount AS comments
        ORDER BY days_open DESC LIMIT 25""")), hide_index=True)

    st.subheader("Slowest fixes (issue created to PR merged)")
    st.dataframe(pd.DataFrame(q("""
        MATCH (p:PullRequest)-[:FIXES]->(i:Issue) WHERE p.mergedAt IS NOT NULL
        RETURN i.number AS issue, i.title AS title,
               duration.inDays(i.createdAt, p.mergedAt).days AS days
        ORDER BY days DESC LIMIT 15""")), hide_index=True)

# ---------- PM metrics ----------
with tab4:
    df = pr_frame(q)
    st.caption("These numbers are prompts for a conversation, not rankings of people. "
               "Each is a proxy (see SPACE framework): check the context before acting.")
    if df.empty:
        st.warning("No merged PR data. Did the reload finish?")
    else:
        m = team_monthly(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Median PR cycle time (days)", f"{df['cycle_days'].median():.1f}")
        c2.metric("Median time to first review (h)", f"{df['review_hours'].median():.1f}")
        c3.metric("PRs merged (humans only)", f"{len(df):,}")

        st.subheader("Efficiency & flow: PR cycle time per month")
        st.line_chart(m["median_cycle_days"])
        st.caption("Open to merge. A rising trend can mean review bottlenecks, bigger PRs, or fewer maintainers.")

        st.subheader("Communication: time to first review (hours)")
        st.line_chart(m["median_first_review_hours"])

        st.subheader("Responsiveness to issues (hours to first reply)")
        r = issue_response_monthly(q)
        if len(r):
            st.line_chart(r)

        st.subheader("Activity (context only): PRs merged per month")
        st.bar_chart(m["prs_merged"])

        st.subheader("Anomaly spotlight: who is slower than their own usual?")
        a = anomalies(df)
        if a.empty:
            st.info("No developer has enough merged PRs yet (needs 12+). "
                    "Lower min_prs in metrics.anomalies() to see more.")
        else:
            st.dataframe(a, hide_index=True)
        st.caption("Compares each developer's last 5 merged PRs with their own earlier history "
                   "(robust z-score on log cycle time; needs 12+ PRs). z of 2 or more is flagged. "
                   "size_ratio is how much bigger recent PRs were, since large PRs take longer.")

# ---------- AI assistant ----------
with tab5: render_chat(q)