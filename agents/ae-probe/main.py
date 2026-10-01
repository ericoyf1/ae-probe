"""Atlas Agent Engine four-admission probe agent (H1 research, benign).

Read-only probes; no LLM calls; results printed to stdout (logs) and returned
in the final message. Nothing is exfiltrated beyond the H1 researcher's own
invoke response.
"""
import json
import os
import urllib.request

from agent_engine_sdk_langgraph import App
from langgraph.graph import MessagesState, StateGraph

app = App(app_name="ae-probe")


def _mask(v):
    v = str(v)
    return (v[:3] + "***" + v[-2:]) if len(v) > 8 else "***"


def probe_env():
    """[C2] Process env: secret-ish vars, names + masked values only."""
    hits = {}
    for k in sorted(os.environ):
        ku = k.upper()
        if any(s in ku for s in ("SECRET", "KEY", "TOKEN", "PASSWORD", "URI", "ENDPOINT", "OAUTH", "MCP", "CREDENTIAL")):
            hits[k] = _mask(os.environ[k])
    return hits


def probe_oe():
    """[C1] Discover internal control-plane endpoints from env; try unauthenticated GETs."""
    out = {}
    bases = {}
    for var in ("AER_ENDPOINT", "TOOL_ENDPOINT", "OE_ENDPOINT", "AGENTENGINE_ENDPOINT", "AE_ORCHESTRATOR_URL"):
        v = os.environ.get(var)
        if v:
            bases[var] = v
            out["base:" + var] = v
    for name, base in bases.items():
        for path in ("/healthz", "/readyz", "/api/v1/agents", "/agents", "/api/v1/workspaces", "/workspaces", "/api/v1/deployments"):
            try:
                req = urllib.request.Request(base.rstrip("/") + path, method="GET")
                with urllib.request.urlopen(req, timeout=4) as r:
                    out[f"oe:{name}{path}"] = "HTTP %d: %s" % (r.status, r.read(150).decode(errors="replace"))
            except Exception as e:
                out[f"oe:{name}{path}"] = "err: %s" % str(e)[:90]
    return out


def probe_bare_http():
    """[C4] Direct outbound HTTP made OUTSIDE app.tool()/app.llm() wrappers."""
    results = {}
    for label, url in (
        ("public_httpbin", "https://httpbin.org/get"),
        ("linked_atlas_cluster", os.environ.get("MONGODB_URI", "").split("/")[2].split("?")[0] if os.environ.get("MONGODB_URI", "").count("/") >= 3 else ""),
    ):
        if not url:
            continue
        try:
            with urllib.request.urlopen("https://" + url if "://" not in url else url, timeout=6) as r:
                results[label] = "HTTP %d, %dB" % (r.status, len(r.read()))
        except Exception as e:
            results[label] = "err: %s" % str(e)[:110]
    return results


@app.tool()
def tool_env_read() -> str:
    """[C3] Registered tool reading the same process env (same-sandbox cohabitation)."""
    return json.dumps(probe_env())[:900]


@app.tool()
def tool_echo(x: str) -> str:
    """Plain echo tool."""
    return "echo:" + str(x)[:80]


@app.entrypoint
def build_agent():
    def call_model(state: MessagesState):
        report = {
            "c2_process_env": probe_env(),
            "c3_tool_env": json.loads(tool_env_read.fn() if hasattr(tool_env_read, "fn") else "{}"),
            "c1_oe": probe_oe(),
            "c4_bare_http": probe_bare_http(),
        }
        line = json.dumps(report, default=str)
        print("[AE-PROBE-REPORT] " + line[:2500], flush=True)
        return {"messages": [{"role": "assistant", "content": line[:3800]}]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.set_entry_point("agent")
    return graph.compile(checkpointer=app.checkpointer())


app.run()
