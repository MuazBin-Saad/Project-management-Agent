"""Stage 2: Streamlit chat tab (src/dashboard/chat_tab.py)."""
import os
import sys
from pathlib import Path

import streamlit as st

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC / "agent"))
from agent import build_agent, ask  # noqa: E402

EXAMPLES = [
    "Is anyone slower than usual lately, and could it be explained by PR size?",
    "Is PR cycle time getting worse? What changed in review time?",
    "Who should take issue #2500? Show your reasoning.",
    "Which past issues look like #2400 and how long did they take to fix?",
]


@st.cache_resource
def _agent(_q):
    return build_agent(_q)


def render(q):
    st.subheader("Ask about your project")
    if not os.getenv("LLM_API_KEY", "").strip():
        st.info("Set LLM_API_KEY (and optionally LLM_PROVIDER = gemini | claude | openai) in .env, "
                "then restart, to enable the assistant.")
        return

    st.caption("Answers come from fixed tools over the project graph and are shown with the tools used. "
               "Metrics are prompts for a conversation, not rankings of people.")
    if "chat" not in st.session_state:
        st.session_state.chat = []

    with st.expander("Example questions"):
        for e in EXAMPLES:
            st.write("- " + e)

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("tools"):
                st.caption("Evidence from: " + ", ".join(m["tools"]))

    prompt = st.chat_input("Ask a question, e.g. 'Which open issues look stalled?'")
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Checking the project data..."):
                try:
                    answer, used = ask(_agent(q), st.session_state.chat)
                except Exception as e:  # show a readable error instead of a stack trace
                    answer, used = f"Sorry, the assistant failed: {e}", []
            st.markdown(answer)
            if used:
                st.caption("Evidence from: " + ", ".join(used))
        st.session_state.chat.append({"role": "assistant", "content": answer, "tools": used})