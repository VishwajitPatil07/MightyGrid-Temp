import base64
import os
import requests
import json
import re
from PIL import Image
from io import BytesIO
from .prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_DEBUG

# Model server endpoint + model name. Defaults point at LM Studio (local dev).
# To use vLLM instead (incl. RunPod Serverless), set these env vars -- the request
# format is identical (OpenAI-compatible /chat/completions), so nothing else in the
# pipeline changes:
#   MG_LLM_URL    e.g. https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1/chat/completions
#   MG_LLM_MODEL  must match the server's model id (RunPod: the MODEL_NAME you set,
#                 e.g. ByteDance-Seed/UI-TARS-7B-DPO)
#   MG_LLM_KEY    Bearer token for a hosted endpoint (RunPod API key). Empty for LM Studio.
API_URL = os.environ.get("MG_LLM_URL", "http://localhost:1234/v1/chat/completions")
MODEL_NAME = os.environ.get("MG_LLM_MODEL", "lmstudio-community/UI-TARS-7B-DPO-GGUF")
API_KEY = os.environ.get("MG_LLM_KEY", "")
# Cold-start on serverless (spin up a worker, load the 7B) can take a while; give the
# request room. Tunable via MG_LLM_TIMEOUT (seconds).
REQUEST_TIMEOUT = float(os.environ.get("MG_LLM_TIMEOUT", "180"))

# OPT-IN debug: when on, the model writes a one-line "Thought:" explaining its choice
# and we print it, so you can see WHY it tapped where it did. OFF by default because a
# Thought adds tokens => slower decisions; leave it off for normal/fast runs, turn it
# on ($env:MG_SHOW_THOUGHT="1") only when debugging a run. It does NOT change what the
# model decides (same crisp, persona-blind prompt) -- it only asks it to say why.
SHOW_THOUGHT = os.environ.get("MG_SHOW_THOUGHT", "0") not in ("0", "", "false", "off", "no")


def _headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = "Bearer %s" % API_KEY
    return h


def process_and_encode_image(image_path: str, max_width: int = None) -> str:
    # Image RESOLUTION is the dominant grounding-accuracy lever for UI-TARS (its own
    # paper: accuracy rises significantly with input resolution; small targets in a
    # shrunk image are the main failure mode). 336px was chosen only because the local
    # AMD card made big images take ~60s -- but on the cloud GPU the compute is tiny
    # (network dominates), so a bigger image is affordable and buys real accuracy.
    # Default 768px (a large step up from 336). Tunable via MG_IMG_WIDTH: 512 = faster
    # / a bit less accurate, 1024 = most accurate / a bit more upload+prefill. Changing
    # this does NOT affect coordinate math -- the model outputs 0-1000 normalized and
    # we map that to device pixels regardless of the sent image size.
    if max_width is None:
        max_width = int(os.environ.get("MG_IMG_WIDTH", "768"))
    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        w_percent = max_width / float(img.size[0])
        h_size = int(float(img.size[1]) * float(w_percent))
        img = img.resize((max_width, h_size), Image.Resampling.LANCZOS)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def _extract_thought(text):
    """Pull the one-line 'Thought:' out of the model's reply, if present (debug only)."""
    m = re.search(r'Thought:\s*(.+?)(?=\n\s*Action:|$)', text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_ui_tars_action(text: str) -> dict:
    action_part = text
    m = re.search(r'Action:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
    if m:
        action_part = m.group(1).strip()

    open_app = re.search(r'open\s*(?:=|\()\s*(?:[a-zA-Z0-9_]+\s*=\s*)?[\'"]?([^)\'"]+)[\'"]?\s*\)?', action_part, re.IGNORECASE)
    if open_app: return {"action": "open", "text": open_app.group(1).strip()}

    click = re.search(r'click\s*\(.*?(\d+)\s*,\s*(\d+)', action_part, re.IGNORECASE)
    if click: return {"action": "click", "target": [int(click.group(1)), int(click.group(2))]}

    scroll = re.search(r'scroll\s*\(.*?direction\s*=\s*[\'"]?(\w+)', action_part, re.IGNORECASE)
    if scroll: return {"action": "scroll", "direction": scroll.group(1).lower()}

    # Drag interceptor
    if re.search(r'drag', action_part, re.IGNORECASE): return {"action": "scroll", "direction": "down"}

    typ = re.search(r'type\s*\(\s*content\s*=\s*[\'"](.*?)[\'"]\s*\)', action_part, re.IGNORECASE | re.DOTALL)
    if typ: return {"action": "type", "text": typ.group(1)}

    if re.search(r'\babort\b', action_part, re.IGNORECASE): return {"action": "abort"}
    if re.search(r'press_back', action_part, re.IGNORECASE): return {"action": "back"}
    if re.search(r'press_home', action_part, re.IGNORECASE): return {"action": "home"}
    if re.search(r'press_enter', action_part, re.IGNORECASE): return {"action": "enter"}
    if re.search(r'wait', action_part, re.IGNORECASE): return {"action": "wait"}
    if re.search(r'finished', action_part, re.IGNORECASE): return {"action": "finished"}

    coord = re.search(r'[\(\[]\s*(\d+)\s*,\s*(\d+)\s*[\)\]]', action_part)
    if coord: return {"action": "click", "target": [int(coord.group(1)), int(coord.group(2))]}

    return {"action": "none", "target": "error"}


def ask_ui_tars_7b(goal: str, screen_data_json: str, image_path: str, action_history=None):
    base64_image = process_and_encode_image(image_path)

    history_text = "None"
    if action_history and len(action_history) > 0:
        # Bulleted, NOT numbered ("Step 1:", "Step 2:" ...). A numbered list invites the
        # model to CONTINUE the numbering with prose ("11. The user has searched for...")
        # instead of emitting an Action line -- which the parser then reads as 'none'.
        formatted_history = ["- %s" % act for act in action_history]
        history_text = "\n".join(formatted_history)
        if len(action_history) >= 2 and action_history[-1] == action_history[-2]:
            history_text += "\n\nSYSTEM WARNING: Your last action did not change the screen. You are stuck in a loop. DO NOT repeat the same action. Look at the screen and click a different button, or press_back()."

    elements_block = (screen_data_json or "").strip()
    if elements_block:
        elements_block = ("\n\nElements on screen (each with its (x,y) in 0-1000 coords):\n"
                          + elements_block
                          + "\nTo tap one, use ITS (x,y) from this list.")
    user_prompt = (f"Objective: {goal}\nDone so far:\n{history_text}{elements_block}\n\n"
                   f"Decide the SINGLE next action for the current screen.")

    # HARD CEILING on output length. A decision only needs one short "Action:"
    # line (~15-25 tokens). Normally capped at 64 no matter what MG_MAX_TOKENS is
    # set to -- a high env value was letting the model write a huge essay per call
    # (seconds -> minutes). The stop sequences also cut generation the moment the
    # action line ends. In debug (SHOW_THOUGHT) mode we allow a bit more room for the
    # one Thought line and do NOT stop on "\nThought".
    if SHOW_THOUGHT:
        system_prompt = SYSTEM_PROMPT_DEBUG
        max_tokens = min(int(os.environ.get("MG_MAX_TOKENS", "96")), 128)
        stop = ["\nObservation", "<|im_end|>", "<|endoftext|>"]
    else:
        system_prompt = SYSTEM_PROMPT
        max_tokens = min(int(os.environ.get("MG_MAX_TOKENS", "48")), 64)
        stop = ["\nThought", "\nObservation", "\n\n"]

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,%s" % base64_image}}]}
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stop": stop
    }

    try:
        response = requests.post(API_URL, json=payload, headers=_headers(), timeout=REQUEST_TIMEOUT)
    except requests.exceptions.ConnectionError:
        return {"action": "none", "target": "error"}, "Server Not Running"
    except requests.exceptions.Timeout:
        return {"action": "none", "target": "error"}, "Server timed out (cold start?)"

    response.raise_for_status()
    data = response.json()
    result_text = data["choices"][0]["message"]["content"].strip()
    gen = (data.get("usage") or {}).get("completion_tokens", "?")
    if SHOW_THOUGHT:
        thought = _extract_thought(result_text)
        if thought:
            print("      [thought] %s" % thought)
    print("      [grounder generated %s tokens, cap %d]" % (gen, max_tokens))
    return parse_ui_tars_action(result_text), result_text