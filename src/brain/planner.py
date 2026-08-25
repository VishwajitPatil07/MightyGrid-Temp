import os
import json
import re
import requests

# The "planner" only READS INTENT from the request (which app, the objective, the
# exact values to type, a done sign). It does NOT plan UI steps -- the agent
# decides each step live from the screen.
#
# LATENCY: to keep the GPU free for UI-TARS, a fast in-process heuristic reads the
# app + values with no model at all. The qwen LLM is a fallback only for requests
# too vague to parse, and it UNLOADS itself right after so it never squats in VRAM
# during the grounding loop (two models on one GPU is what caused ~10x slowdown).

INTENT_API_URL = os.environ.get("MG_PLANNER_URL", "http://localhost:1234/v1/chat/completions")
INTENT_MODEL   = os.environ.get("MG_PLANNER_MODEL", "qwen2.5-7b-instruct")

SYSTEM_PROMPT = """You read a user's request for a phone task and extract its INTENT.
You do NOT plan screens or UI steps -- the agent decides every step live by looking
at the phone. Only capture what the user wants.

Output ONE JSON object and nothing else:
{"app":"<app to open, or empty>",
 "objective":"<one short sentence: the overall aim, not a list of taps>",
 "values":["<exact text the user wants entered, in order>"],
 "success_when":"<a visible sign the task is finished>"}

Rules:
- values = the literal things to type (emails, names, cities, dates, search text),
  in the order the user gave them. Empty list if nothing needs typing.
- Copy values EXACTLY from the request. Never invent or translate them.
- objective is the goal, not steps. Keep everything short.

Example:
Request: Open Uber and book a ride from Andheri to the airport
{"app":"Uber","objective":"book a ride from Andheri to the airport","values":["Andheri","airport"],"success_when":"a driver search or fare estimate screen is shown"}
"""

_EMAIL_RE = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')
_APP_RE = re.compile(
    r'\bopen\s+(?:the\s+)?([A-Za-z0-9][A-Za-z0-9._]*(?:\s[A-Za-z0-9][A-Za-z0-9._]*)?)'
    r'(?=\s+app\b|\s*,|\s+and\b|\s+then\b|\s+to\b|$)', re.I)
_VALUE_RE = re.compile(
    r'\b(?:type|enter|input|write)\s+["\']?(\S{2,})', re.I)
_SEARCH_VALUE_RE = re.compile(
    r'\bsearch\s+for\s+["\']?(.+?)(?:["\']?(?:\s*(?:then|after that|afterwards?)\b|[;\n]|$))', re.I)
# A "flight/train/bus/ride from X to Y" request: capture the ORIGIN and DESTINATION as two
# separate values (not the whole sentence). The date is deliberately NOT captured as a value
# -- it is chosen on the app's calendar, not typed into the city field. Origin stops at "to";
# destination stops at a date word ("on"/"for"), a connector, or end of string.
_FLIGHT_ROUTE_RE = re.compile(
    r'\bfrom\s+(?P<origin>.+?)\s+to\s+(?P<dest>.+?)'
    r'(?:\s+(?:on|for|departing|leaving|dated?)\b.*)?'
    r'(?:\s*(?:then|after that|afterwards?)\b|[;\n]|$)', re.I)
# Labeled credentials the way people actually write them:
#   "id --> Akashtest123", "password: testing123@", "username = akash", "otp 4821"
# Value is any run of non-space characters (so symbols in emails/passwords -- @ ! ( #
# $ etc. -- are kept, not truncated); trailing sentence punctuation is trimmed after.
_CRED_RE = re.compile(
    r'\b(?:id|user(?:name)?|e-?mail|pass(?:word|waord|wrd|wd)?|pwd|otp|code|mobile|phone)\b'
    r'\s*(?:=+>?|-+>?|:|is)?\s*["\']?(\S{3,})', re.I)
_VALUE_STOP = {"the", "a", "an", "into", "in", "on", "your", "my", "some", "text", "and",
               # field-name words, not values -- the real value follows them
               "email", "username", "password", "name", "number", "phone", "mobile",
               "address", "otp", "code", "id", "field", "box", "here", "details"}


def _clean_value(v):
    """Trim quotes and trailing sentence punctuation a capture may pick up, WITHOUT
    touching symbols that belong to the value. 'test@!(7est.com,' -> 'test@!(7est.com';
    'test@test.com.' -> 'test@test.com'. Keeps a trailing '@' (e.g. 'testing123@')."""
    v = v.strip()
    if len(v) >= 2 and v[0] in "\"'" and v[-1] in "\"'":
        v = v[1:-1]
    return v.rstrip("'\",;.").strip()


def _looks_like_value(v):
    """A real credential/value looks like one: has a digit, an @, or is long.
    Filters out connectors ('and', 'this') a loose pattern might grab."""
    if v.lower() in _VALUE_STOP or v.lower() in {"and", "this", "is", "are", "with", "for", "want"}:
        return False
    return any(c.isdigit() for c in v) or "@" in v or len(v) >= 5


def _heuristic_intent(query):
    """Read app + values straight from the request, no model. objective = the raw
    request (the grounder reads it every turn anyway); success_when left empty so
    the loop leans on a verified finished() instead of a vague keyword match."""
    q = query or ""
    app = ""
    m = _APP_RE.search(q)
    if m:
        app = re.sub(r'\s+app$', '', m.group(1).strip(), flags=re.I).strip()
    values = list(_EMAIL_RE.findall(q))
    for m in _VALUE_RE.finditer(q):
        v = _clean_value(m.group(1))
        if v and v.lower() not in _VALUE_STOP and v not in values:
            values.append(v)
    # A "from X to Y" route -> two values (origin, destination). Take this BEFORE the greedy
    # "search for ..." rule, which would otherwise swallow the whole sentence as one value
    # (the bug that filled the 'From' field with "a flight from Pune to Nagpur on 21 Aug 2026").
    route = _FLIGHT_ROUTE_RE.search(q)
    if route:
        for part in (route.group("origin"), route.group("dest")):
            v = _clean_value(part)
            # keep the CITY only: cut anything after a comma and drop a leading article
            v = re.sub(r"^(?:the|a|an)\s+", "", v, flags=re.I).split(",")[0].strip()
            if v and v.lower() not in _VALUE_STOP and v not in values:
                values.append(v)
    else:
        for m in _SEARCH_VALUE_RE.finditer(q):
            v = _clean_value(m.group(1))
            if v and v.lower() not in _VALUE_STOP and v not in values:
                values.append(v)
    for m in _CRED_RE.finditer(q):           # labeled credentials, in order
        v = _clean_value(m.group(1))
        if _looks_like_value(v) and v not in values:
            values.append(v)
    return {"app": app, "objective": q.strip(), "values": values, "success_when": ""}


def _first_json(text):
    text = re.sub(r"```(?:json)?", "", text).strip()
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        try:
            return json.loads(text[i:j + 1])
        except Exception:
            return None
    return None


def _post(payload):
    return requests.post(INTENT_API_URL, json=payload)


def _unload_model():
    """Best-effort: tell Ollama to drop the intent model from memory NOW, so it
    isn't holding VRAM while UI-TARS runs. Harmless if the endpoint isn't Ollama."""
    try:
        base = INTENT_API_URL.split("/v1/")[0]
        requests.post(base + "/api/generate",
                      json={"model": INTENT_MODEL, "keep_alive": 0}, timeout=5)
    except Exception:
        pass


# Split a client request into ordered sub-steps on STRONG sequence markers only
# ("then", "after that", ";", newlines, numbered "1." / "2)"). Deliberately NOT on
# bare "and" or "," -- those appear inside a single step ("id X and password Y").
_STEP_SPLIT = re.compile(
    r'(?:\bthen\b|\band then\b|\bafter that\b|\bafterwards?\b|[;\n]|(?<!\d)\b\d+\s*[.)]\s+)', re.I)


def plan_steps(query):
    """Turn a request into a list of step-intents (each an _heuristic_intent dict:
    app / objective / values / success_when), in order. One step for a simple
    request; several for a multi-step client workflow. The app is named once
    (usually step 1) and every step runs the app that step-1 opened."""
    q = (query or "").strip()
    parts = [p.strip(" ,.\t-") for p in _STEP_SPLIT.split(q)]
    parts = [p for p in parts if p and len(p) > 1]
    if len(parts) <= 1:
        return [_heuristic_intent(q)]
    steps = [_heuristic_intent(p) for p in parts]
    app = next((s["app"] for s in steps if s["app"]), "")
    if app and not steps[0]["app"]:
        steps[0]["app"] = app
    return steps


def parse_intent(query, persona=None, verbose=True):
    heur = _heuristic_intent(query)
    # DEFAULT: heuristic only -- NO second model. Running qwen next to UI-TARS on
    # a small (8 GB) GPU pushes UI-TARS onto the CPU (~10x slower per decision).
    # Keeping a SINGLE model on the GPU is what restores the fast ~5s decisions
    # the original had. The LLM intent parser is opt-in via MG_FORCE_LLM_INTENT=1,
    # meant for a big-GPU host (e.g. RunPod) where VRAM isn't the constraint.
    if not os.environ.get("MG_FORCE_LLM_INTENT"):
        if verbose:
            print("Intent (fast, no LLM): app=%r values=%r" % (heur["app"], heur["values"]))
        return heur

    base = {
        "model": INTENT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Request: %s" % query},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
        "keep_alive": 0,
    }
    if verbose:
        print("Intent model (requested): %s" % INTENT_MODEL)

    raw = None
    try:
        payload = dict(base)
        payload["response_format"] = {"type": "json_object"}
        r = _post(payload)
        if r.status_code >= 400:
            r = _post(base)
        r.raise_for_status()
        resp = r.json()
        if verbose:
            print("Served by (actual): %s" % resp.get("model"))
        raw = resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("Intent LLM unavailable (%s); using the fast heuristic" % e)
    finally:
        _unload_model()

    if raw is None:
        return heur
    obj = _first_json(raw)
    if not isinstance(obj, dict) or not obj.get("objective"):
        print("Could not parse intent; using the fast heuristic. Raw:\n%s" % raw)
        return heur

    vals = obj.get("values") or []
    if isinstance(vals, str):
        vals = [vals]
    return {
        "app": str(obj.get("app", "")).strip() or heur["app"],
        "objective": str(obj.get("objective", "")).strip() or query,
        "values": [str(v).strip() for v in vals if str(v).strip()] or heur["values"],
        "success_when": str(obj.get("success_when", "")).strip(),
    }


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else (
        "Open the Discord app, navigate to Add Friend, type Akas1306 into the username field, and press Send Friend Request"
    )
    intent = parse_intent(q)
    print("\n--- INTENT ---")
    for k, v in intent.items():
        print("%-12s %s" % (k + ":", v))