"""
dream_ui.py
===========
Dreamscape Mapper — Interactive Terminal Interface
Run with:  python dream_ui.py

Wraps dream_pipeline.run_pipeline() with a user-friendly REPL that:
  • Accepts free-form dream text (multi-line supported)
  • Displays the structured output in a readable, colour-highlighted format
  • Offers a simple emotion bar chart in the terminal
  • Supports commands: help, demo, clear, quit
"""

import json
import sys
import os
import textwrap
from collections import OrderedDict

# ---------- colour helpers (ANSI, gracefully disabled on Windows) ----------
_USE_COLOR = sys.stdout.isatty() and os.name != "nt"

def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def bold(t):     return _c("1", t)
def cyan(t):     return _c("96", t)
def yellow(t):   return _c("93", t)
def green(t):    return _c("92", t)
def magenta(t):  return _c("95", t)
def red(t):      return _c("91", t)
def dim(t):      return _c("2", t)
def blue(t):     return _c("94", t)

# ---------- pipeline import ----------
try:
    from dream_pipeline_p import DreamPipelineModel, run_production_pipeline
    _prod_model = DreamPipelineModel()

    def run_pipeline(dream_text: str) -> dict:
        """Thin shim: runs production pipeline and normalises keys for the UI."""
        raw = run_production_pipeline(dream_text, _prod_model)
        if "error" in raw:
            raise ValueError(raw["error"])
        # Map semantic_relations (list of {agent,action,target}) to {Agent,Action,Target}
        sem_rels = [
            {
                "Agent":  s.get("agent",  "Unknown"),
                "Action": s.get("action", "Unknown"),
                "Target": s.get("target", "Unknown"),
            }
            for s in raw.get("semantic_relations", [])
        ] or [{"Agent": "Unknown", "Action": "Unknown", "Target": "Unknown"}]

        return {
            "Topic_Cluster":    raw["topic_cluster"],
            "Dominant_Emotion": raw["dominant_emotion"],
            "Key_Entities":     raw["key_entities"],
            "Semantic_Relation": sem_rels,
            "Emotion_Vector":   raw["emotion_vector"],
            "Cluster_Keywords": raw.get("cluster_keywords", []),
            "Coreference_Map":  {},  # not produced by production pipeline
            "Global_Stat":      raw.get("summary", ""),
        }

except ImportError:
    print(red("✗  dream_pipeline_p.py not found in the current directory."))
    print(dim("   Place dream_pipeline_p.py alongside dream_ui.py and retry."))
    sys.exit(1)
except Exception as _load_err:
    print(red(f"✗  Failed to load production model: {_load_err}"))
    sys.exit(1)

# ---------- constants ----------
BANNER = f"""
{cyan('╔══════════════════════════════════════════════════════════════╗')}
{cyan('║')}   {bold('🌙  Dreamscape Mapper  —  Thematic & Affective Analyser')}   {cyan('║')}
{cyan('║')}       {dim('Weakly Supervised NLP Pipeline  |  NRC Emotions')}          {cyan('║')}
{cyan('╚══════════════════════════════════════════════════════════════╝')}
"""

HELP_TEXT = f"""
{bold('Commands')}
  {yellow('demo')}     — run a built-in example dream
  {yellow('help')}     — show this message
  {yellow('clear')}    — clear the screen
  {yellow('quit')}     — exit

{bold('How to enter a dream')}
  • Type (or paste) your dream narrative and press {yellow('Enter twice')} (blank line)
    to submit, or end a single-line dream with {yellow('Enter')} directly.
  • Multi-sentence dreams give richer results.
"""

DEMO_DREAM = (
    "I looked in the mirror and suddenly my teeth started falling out. "
    "My mother was standing nearby laughing at me and I felt extremely embarrassed."
)

EMOTION_BAR_WIDTH = 30
EMOTION_ORDER = [
    "fear", "sadness", "anger", "disgust",
    "surprise", "anticipation", "joy", "trust",
    "negative", "positive",
]
EMOTION_COLORS = {
    "fear":        red,
    "sadness":     blue,
    "anger":       red,
    "disgust":     magenta,
    "surprise":    yellow,
    "anticipation":cyan,
    "joy":         green,
    "trust":       green,
    "negative":    dim,
    "positive":    green,
}


# ---------- rendering helpers ----------

def _separator(char="─", width=64):
    print(dim(char * width))


def _header(text: str):
    _separator()
    print(f"  {bold(text)}")
    _separator()


def _render_emotion_bars(emotion_vector: dict):
    """Print a simple ASCII bar chart for the NRC emotion vector."""
    print(f"\n  {bold('Emotion Profile  (NRC)')}")
    _separator("·")

    ordered = [(e, emotion_vector.get(e, 0.0)) for e in EMOTION_ORDER]
    max_val = max(v for _, v in ordered) if ordered else 1.0
    if max_val == 0:
        max_val = 1.0

    for emo, val in ordered:
        bar_len = int((val / max_val) * EMOTION_BAR_WIDTH)
        bar = "█" * bar_len + "░" * (EMOTION_BAR_WIDTH - bar_len)
        color = EMOTION_COLORS.get(emo, lambda x: x)
        label = f"{emo:<14}"
        score = f"{val:.3f}"
        print(f"  {dim(label)}  {color(bar)}  {score}")


def _render_result(result: dict):
    """Pretty-print the full pipeline output."""
    print()
    _header("📋  Analysis Result")

    # Topic cluster
    print(f"\n  {bold('Topic Cluster')}     {cyan(result['Topic_Cluster'])}")

    # Dominant emotion
    dom = result["Dominant_Emotion"]
    dom_color = EMOTION_COLORS.get(dom.lower(), yellow)
    print(f"  {bold('Dominant Emotion')}  {dom_color(dom)}")

    # Key entities
    entities_str = "  " + dim("·") + "  "
    entities_str += f"  {dim('·')}  ".join(green(e) for e in result["Key_Entities"])
    print(f"\n  {bold('Key Entities')}")
    print(f"    {', '.join(green(e) for e in result['Key_Entities'])}")

    # Semantic relation — pick the first relation from the list
    sr_list = result.get("Semantic_Relation", [])
    sr = sr_list[0] if sr_list else {"Agent": "Unknown", "Action": "Unknown", "Target": "Unknown"}
    print(f"\n  {bold('Semantic Relation')}")
    print(
        f"    {yellow(sr['Agent'])}  {dim('─[')}  {cyan(sr['Action'])}  {dim(']─▶')}  {magenta(sr['Target'])}"
    )

    # Cluster keywords
    if result.get("Cluster_Keywords"):
        kw = "  ".join(dim(k) for k in result["Cluster_Keywords"])
        print(f"\n  {bold('Cluster Keywords')}  {kw}")

    # Coreference
    if result.get("Coreference_Map"):
        cr = ", ".join(f"{dim(k)} → {green(v)}" for k, v in result["Coreference_Map"].items())
        print(f"  {bold('Coreference')}       {cr}")

    # Emotion bars
    _render_emotion_bars(result["Emotion_Vector"])

    # Global stat
    print(f"\n  {bold('Global Stat')}")
    wrapped = textwrap.fill(result["Global_Stat"], width=60, initial_indent="    ", subsequent_indent="    ")
    print(dim(wrapped))

    # Raw JSON toggle
    print(f"\n  {dim('─── raw JSON ───────────────────────────────────────────────')}")
    print(dim(json.dumps(result, indent=4)))
    _separator()
    print()


def _read_multiline_dream() -> str:
    """Read dream text; blank line or single Enter signals end of input."""
    print(
        f"\n{bold('Enter your dream')} "
        f"{dim('(press Enter twice or Ctrl-D to submit, Ctrl-C to cancel)')}\n"
    )
    lines = []
    try:
        while True:
            line = input()
            if line == "" and lines:       # blank line after content → submit
                break
            lines.append(line)
    except EOFError:
        pass  # Ctrl-D
    return " ".join(lines).strip()


# ---------- main REPL ----------

def main():
    print(BANNER)
    print(f"  Type {yellow('help')} for commands, {yellow('demo')} to run an example, {yellow('quit')} to exit.\n")

    while True:
        try:
            cmd = input(f"{cyan('dreamscape')} {dim('▶')} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{dim('Goodbye.')}")
            break

        if cmd in ("quit", "exit", "q"):
            print(f"\n{dim('Goodbye. Sweet dreams.')}")
            break

        elif cmd == "help":
            print(HELP_TEXT)

        elif cmd == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            print(BANNER)

        elif cmd == "demo":
            print(f"\n{bold('Demo dream:')}")
            print(f"  {dim(DEMO_DREAM)}\n")
            _run_and_display(DEMO_DREAM)

        elif cmd == "":
            # User pressed Enter at the prompt — go to dream entry
            dream = _read_multiline_dream()
            if dream:
                _run_and_display(dream)
            else:
                print(dim("  (no input received)"))

        else:
            # Treat any other input as the dream itself (single-line convenience)
            _run_and_display(cmd)


def _run_and_display(dream_text: str):
    """Run pipeline on dream_text and display the result."""
    if not dream_text.strip():
        print(red("  ✗  Empty input — please enter a dream narrative."))
        return

    print(f"\n{dim('  ⟳  Analysing...')}")
    try:
        result = run_pipeline(dream_text)
        _render_result(result)
    except Exception as exc:
        print(red(f"  ✗  Pipeline error: {exc}"))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
