"""Stage 2: the LLM agent (src/agent/agent.py).

Provider is chosen with environment variables, so the same code supports the model comparison:
  LLM_PROVIDER = gemini | claude | openai     (default: gemini)
  LLM_API_KEY  = your key
  LLM_MODEL    = optional model name override
"""
import os

from tools import build_tools

SYSTEM = """You are a project-management assistant for a software team. You answer questions
about a GitHub project using ONLY the tools provided; never invent numbers, names or issues.

Rules:
- Call tools to get evidence before answering. Prefer several focused tool calls over guessing.
- Quote the specific figures and issue/PR numbers you used so the reader can check them.
- Treat metrics as prompts for a conversation, not judgements of people. When you report a
  developer being slower or faster than usual, mention possible non-performance reasons
  (larger PRs, review delays, holidays, onboarding) and suggest what to check.
- Link-source matters: FIXES links marked 'inferred' are weaker than 'github' links; say so
  when it affects the answer.
- If the data is insufficient, say so plainly and say what would be needed.
- Keep answers short and structured. Suggestions for assignees are suggestions for a human
  to accept or reject; show the reasons."""


def make_llm():
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "").strip() or None
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model or "gemini-2.5-flash", google_api_key=key, temperature=0)
    if provider in ("claude", "anthropic"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model or "claude-sonnet-5-5", api_key=key, temperature=0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-5", api_key=key)
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'")


def build_agent(q):
    llm, tools = make_llm(), build_tools(q)
    try:
        from langchain.agents import create_agent
        return create_agent(llm, tools, system_prompt=SYSTEM)
    except ImportError:
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(llm, tools, prompt=SYSTEM)


def _text(content):
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def ask(agent, history):
    """history: list of {"role": "user"|"assistant", "content": str}. Returns (answer, tools_used)."""
    msgs = [(m["role"], m["content"]) for m in history]
    result = agent.invoke({"messages": msgs}, config={"recursion_limit": 14})
    answer, used = "", []
    for m in result["messages"]:
        if getattr(m, "type", "") == "tool":
            used.append(m.name)
        elif getattr(m, "type", "") == "ai" and _text(m.content).strip():
            answer = _text(m.content)
    seen = []
    for t in used:
        if t not in seen:
            seen.append(t)
    return answer or "No answer was produced.", seen