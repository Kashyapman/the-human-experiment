"""
THE HUMAN EXPERIMENT — Automated Psychology Documentary Pipeline
================================================================
Channel : @TheHumanExperiment
Niche   : Human Psychology & Behaviour — told as Research Documentary

Core Architecture: Single-Task LLM Engine
─────────────────────────────────────────
Every LLM call does exactly ONE thing.
Every output is validated before the pipeline moves forward.
Every failure triggers a self-correction attempt before falling back.
No prompt ever asks for more than one decision simultaneously.

This eliminates the #1 cause of AI content pipeline failure: hallucination
caused by multi-task prompts that overload the model's attention window.

Pipeline Tasks (18 total):
  Phase 0  : Research    → 4 LLM tasks  + 2 web scrapes
  Phase 1  : Writing     → 5 LLM tasks  (chained, each builds on last)
  Phase 1b : Voice Dir.  → 2 LLM tasks  (SSML per line + speaker assignment)
  Phase 1c : Extraction  → 1 LLM task   (key statistic for thumbnail)
  Phase 3  : Visuals     → 1 LLM task   (one prompt per image slot, looped)
  Phase 5  : Metadata    → 5 LLM tasks  (title / description / tags / IG / FB)

GitHub Actions: Runs daily → Full pipeline → Upload → Zero human intervention.
"""

# ═══════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════════════════
import os, re, sys, time, json, math, glob, base64, random, urllib.parse
import wave, xml.etree.ElementTree as ET
from typing import Callable

import cv2
import numpy as np
import requests

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFilter
import PIL.ImageFont

from transformers import pipeline as hf_pipeline
from google import genai
from google.genai import types

from moviepy.editor import (
    ImageClip, VideoClip, VideoFileClip, ColorClip, TextClip,
    AudioFileClip, CompositeVideoClip, CompositeAudioClip,
    concatenate_videoclips, concatenate_audioclips,
)
from moviepy.video.fx.all import colorx, fadein, fadeout
from moviepy.audio.fx.all import audio_loop

from faster_whisper import WhisperModel
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Local modules (same repo)
from neural_voice import VoiceEngine, VOICE_MAP
import meta_upload

# ── Pillow >= 10 removed ANTIALIAS ──────────────────────────────────────────
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS


# ═══════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT CREDENTIALS
# ═══════════════════════════════════════════════════════════════════════════
GEMINI_KEY        = os.environ.get("GEMINI_API_KEY",        "")
OPENROUTER_KEY    = os.environ.get("OPENROUTER_API_KEY",    "")
YOUTUBE_TOKEN_VAL = os.environ.get("YOUTUBE_TOKEN_JSON",    "")
CF_ACCOUNT_ID     = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CF_API_TOKEN      = os.environ.get("CLOUDFLARE_API_TOKEN",  "")
PEXELS_KEY        = os.environ.get("PEXELS_API_KEY",        "")
PIXABAY_KEY       = os.environ.get("PIXABAY_API_KEY",       "")
SEARCH_API_KEY    = os.environ.get("SEARCH_API_KEY",        "")
GOOGLE_CSE_ID     = os.environ.get("GOOGLE_CSE_ID",         "")


# ═══════════════════════════════════════════════════════════════════════════
#  CHANNEL IDENTITY — The Human Experiment
# ═══════════════════════════════════════════════════════════════════════════
CHANNEL_HANDLE  = "@TheHumanExperiment"
CHANNEL_NAME    = "The Human Experiment"
TOPICS_FILE     = "topics.txt"

# Video dimensions (portrait for Shorts)
VIDEO_WIDTH         = 720
VIDEO_HEIGHT        = 1280
IMAGE_TRANSITION_T  = 3.2       # seconds per visual slot
CROSSFADE_DUR       = 0.45      # cross-dissolve overlap

# ── Colour palette (clinical cold aesthetic) ─────────────────────────────
COLOUR_BG           = (8,  10, 14)         # near-black
COLOUR_ACCENT       = (255, 230,  0)       # signal yellow
COLOUR_WHITE        = (245, 245, 248)
COLOUR_RED_BADGE    = (185,  20,  20)
COLOUR_HANDLE_GREY  = (120, 124, 130)

# ── Typography choices ───────────────────────────────────────────────────
# All loaded dynamically via get_subtitle_font(); these are fall-through paths
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]

# ── Banned phrases — never allowed in any script line ────────────────────
BANNED_PHRASES = [
    "shocking", "incredible", "unbelievable", "you won't believe",
    "mind-blowing", "mind-boggling", "fascinating", "amazing", "astonishing",
    "dive into", "buckle up", "chilling reminder", "terrifying",
    "will we ever know", "some say", "in the annals", "it's no secret",
    "today we're going to", "welcome back", "don't forget to subscribe",
    "let that sink in", "plot twist",
]


# ═══════════════════════════════════════════════════════════════════════════
#  ERA-MATCHED VISUAL TEXTURES
#  Appended to every FLUX.1 ai_prompt automatically
# ═══════════════════════════════════════════════════════════════════════════
ERA_STYLES: dict[str, str] = {
    "1900s-1930s": (
        "aged sepia institutional photograph, silver gelatin print, "
        "heavy grain, torn edges, water staining, faded tones"
    ),
    "1940s-1960s": (
        "black and white university laboratory photograph, high contrast, "
        "film grain, harsh fluorescent lighting, clinical vignette, "
        "government research facility aesthetic"
    ),
    "1970s-1980s": (
        "Kodachrome film photograph, slightly oversaturated, light leak on edge, "
        "dust on lens, academic conference room aesthetic, 16mm documentary look"
    ),
    "1990s-2000s": (
        "early digital camera photo, JPEG compression artifacts, "
        "fluorescent office lighting, slightly desaturated, grainy low-light"
    ),
    "modern": (
        "clean clinical white environment, cold blue-white lighting, "
        "brain scan fMRI aesthetic, modern research facility, "
        "4K documentary cinematography, neural pathway visualization"
    ),
    "unknown": (
        "aged archival research photograph, muted documentary tones, "
        "institutional aesthetic, film grain, slightly overexposed"
    ),
}

# ── Psychology-specific scene textures (added to relevant visual slots) ──
PSY_SCENE_TEXTURES = {
    "experiment":   "university laboratory, clinical equipment, observation window, one-way mirror",
    "data":         "printed research tables, typewritten results, overhead projector slide",
    "person":       "anonymous silhouette, blurred identity, documentary portrait lighting",
    "brain":        "anatomical diagram, neural pathway illustration, cross-section scan",
    "institution":  "hallway of academic building, institutional corridor, lecture hall",
}


# ═══════════════════════════════════════════════════════════════════════════
#  SFX & STINGER MAPS
# ═══════════════════════════════════════════════════════════════════════════
SFX_KEYWORD_MAP: dict[str, str] = {
    "scream":   "scream.mp3",
    "knock":    "knock.mp3",
    "breath":   "whisper.mp3",
    "static":   "static.mp3",
    "thud":     "thud.mp3",
    "tick":     "clock_tick.mp3",
    "silence":  "deep_silence.mp3",
}

STINGER_MAP: dict[str, str] = {
    "refused to publish":    "reverb_hit.mp3",
    "stopped the study":     "deep_impact.mp3",
    "never replicated":      "low_drone_sting.mp3",
    "opposite of":           "reverb_hit.mp3",
    "no explanation":        "deep_impact.mp3",
    "without their consent": "low_drone_sting.mp3",
    "classified":            "radio_static_burst.mp3",
    "percent":               "deep_impact.mp3",
    "indistinguishable":     "reverb_hit.mp3",
}


# ═══════════════════════════════════════════════════════════════════════════
#  CONTENT POOL — 40 psychology research angles
#  Each entry is ONE narrow research domain for task_propose_study()
# ═══════════════════════════════════════════════════════════════════════════
CONTENT_POOL: list[str] = [
    # OBEDIENCE & AUTHORITY
    "Studies where ordinary people followed authority instructions that violated their own ethics",
    "Experiments measuring how quickly participants abandoned their own perceptions under group pressure",
    "Research on obedience to authority in professional medical and military settings",
    "Studies where the presence of an authority figure changed moral decision-making measurably",

    # CONFORMITY & SOCIAL INFLUENCE
    "Documented cases where entire groups reached conclusions no individual would reach alone",
    "Experiments showing how quickly people adopt the beliefs of a group they just joined",
    "Studies on how group size affects individual willingness to report obvious errors",
    "Research on social contagion — how emotions and behaviours spread through groups invisibly",

    # DECISION MAKING & COGNITIVE BIAS
    "Studies where framing identical information differently produced completely opposite decisions",
    "Experiments where adding more information caused worse decisions than less information",
    "Research on how physical state — hunger, temperature, pain — changes ethical judgments measurably",
    "Studies proving expert decision-makers make predictably poor choices in their own speciality",
    "Experiments where financial incentives caused measurably worse performance than no incentive",

    # MEMORY & PERCEPTION
    "Research showing human memory actively reconstructs rather than retrieves stored events",
    "Studies where confident eyewitnesses described events that physically could not have occurred",
    "Experiments that implanted detailed false memories in normal healthy adults",
    "Research on how leading questions change what witnesses are certain they saw",

    # IDENTITY & PERSONALITY
    "Studies showing personality traits are far less stable across contexts than people report",
    "Experiments where normal individuals adopted harmful behaviours within hours of being given a role",
    "Research proving that self-reported behaviour and actual measured behaviour rarely match",
    "Studies on how anonymity changes moral behaviour in otherwise ethical individuals",

    # MOTIVATION & REWARD
    "Studies where telling children their work was impressive caused measurably worse future performance",
    "Experiments where externally rewarding an intrinsic behaviour destroyed the intrinsic motivation",
    "Research on why deadlines improve some people's performance and severely damage others",
    "Studies showing the documented gap between what people say motivates them and what actually does",

    # BYSTANDER & HELPING BEHAVIOUR
    "Studies where increasing the number of bystanders reduced the probability of help being given",
    "Experiments on the conditions required for people to intervene when witnessing harm",
    "Research on why people who help strangers give explanations that do not match their measured behaviour",

    # TRAUMA & RESILIENCE
    "Studies on what specifically predicts who recovers from acute stress and who does not",
    "Research measuring post-traumatic growth as a documented, reproducible psychological phenomenon",
    "Experiments on the specific conditions that accelerate or block recovery from adversity",

    # SOCIAL PERCEPTION & BIAS
    "Studies where identical CVs with different names received measurably different treatment",
    "Research on how attractiveness changes what competence and intelligence people are rated as having",
    "Experiments proving people consistently overestimate their own ability to detect deception",
    "Studies where changing one word in a description changed every downstream judgment made",

    # HAPPINESS & WELLBEING
    "Studies proving that what people predict will make them happy is consistently inaccurate",
    "Research on why winning a lottery and becoming paralysed produce similar long-term happiness levels",
    "Experiments showing that hedonic adaptation eliminates most anticipated positive emotions",
    "Studies on the one documented variable that predicts long-term life satisfaction more than any other",

    # REPLICATION & SCIENTIFIC PROCESS
    "Famous psychology studies that failed to replicate and what that revealed about the original findings",
    "Research papers retracted after the raw data was independently reviewed",
    "Studies stopped mid-experiment because of what was happening to participants",
    "Experiments that produced the exact opposite result when run by an independent team",
]


# ═══════════════════════════════════════════════════════════════════════════
#  VIDEO FORMAT SELECTION
# ═══════════════════════════════════════════════════════════════════════════
VIDEO_FORMATS: list[dict] = [
    {"label": "short",  "max_lines": 8,  "description": "Quick Hit  (~45 s)"},
    {"label": "medium", "max_lines": 12, "description": "Standard   (~65 s)"},
    {"label": "deep",   "max_lines": 16, "description": "Deep Dive  (~85 s)"},
]


# ═══════════════════════════════════════════════════════════════════════════
#  ANTI-BAN & MEMORY
# ═══════════════════════════════════════════════════════════════════════════
def anti_ban_sleep() -> None:
    """Randomises start time on GitHub Actions to avoid detectable bot cadence."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        secs = random.randint(180, 540)
        print(f"🕵️  Anti-ban pause: {secs // 60}m {secs % 60}s")
        time.sleep(secs)


def get_past_topics() -> str:
    """Returns the last 100 case names from memory, as a newline-separated string."""
    if not os.path.exists(TOPICS_FILE):
        return ""
    with open(TOPICS_FILE, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return "\n".join(lines[-100:])


def save_new_topic(topic: str) -> None:
    """Appends a completed topic to the memory file."""
    try:
        with open(TOPICS_FILE, "a", encoding="utf-8") as f:
            f.write(f"{topic}\n")
        print(f"💾 Saved to memory: '{topic}'")
    except Exception as e:
        print(f"⚠️  Memory save failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  SOTA MODEL SELECTOR
#  Hits OpenRouter live, scores against a reward matrix, returns best 3
# ═══════════════════════════════════════════════════════════════════════════
SOTA_REWARD_MATRIX: dict[str, int] = {
    "meta-llama/llama-3.3-70b-instruct:free": 99,
    "qwen/qwen3-235b-a22b:free":              98,
    "mistralai/mistral-large:free":           97,
    "deepseek/deepseek-r1:free":              95,
    "nvidia/nemotron-4-super:free":           94,
    "google/gemma-3-27b-it:free":             88,
}

DEFAULT_SOTA_MODELS: list[str] = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-235b-a22b:free",
    "mistralai/mistral-large:free",
]


def get_top_free_openrouter_models(limit: int = 3) -> list[str]:
    """Queries OpenRouter for free models and returns the top N by reward score."""
    print("🔍 Scoring OpenRouter SOTA models...")
    if not OPENROUTER_KEY:
        return DEFAULT_SOTA_MODELS
    try:
        r = requests.get("https://openrouter.ai/api/v1/models", timeout=15)
        if r.status_code != 200:
            return DEFAULT_SOTA_MODELS
        all_models = r.json().get("data", [])
        # Add this constant near the top with the other config
        BLOCKED_MODELS = [
            "llama-3.2-3b",     # Too small — fails format instructions
            "llama-3.2-1b",     # Too small
            "gemma-2-9b",       # Inconsistent JSON output
            "phi-3-mini",       # Too small
            "phi-3.5-mini",
            "qwen3-next-80b-a3b",
            "lfm-2.5-1.2b",      # add this — 1.2B model, too small
            "lfm-2.5",            # block the whole lfm-2.5 family
            "-1.2b",              # block anything under 3B by size suffix
            "-0.5b",
            "-1b:"
        ]
        free_ids = [
            m["id"] for m in all_models
            if ((m.get("pricing", {}).get("prompt") == "0"
                and m.get("pricing", {}).get("completion") == "0")
                or ":free" in m["id"])
            and not any(blocked in m["id"].lower() for blocked in BLOCKED_MODELS)
        ]
        if not free_ids:
            return DEFAULT_SOTA_MODELS

        def _score(mid: str) -> int:
            ml = mid.lower()
            for k, v in SOTA_REWARD_MATRIX.items():
                if k in ml:
                    return v
            s = 50
            if "instruct" in ml: s += 20
            if "llama-3"  in ml: s += 15
            elif "qwen"   in ml: s += 14
            elif "mistral" in ml: s += 10
            return s

        ranked = sorted(free_ids, key=_score, reverse=True)[:limit]
        print(f"🌟 SOTA cascade: {ranked}")
        return ranked
    except Exception as e:
        print(f"⚠️  Model scout failed ({e}), using defaults.")
        return DEFAULT_SOTA_MODELS


# ═══════════════════════════════════════════════════════════════════════════
#  ██████████████████████████████████████████████████████████████████████
#  SINGLE-TASK LLM ENGINE  ← the heart of this pipeline
#  ██████████████████████████████████████████████████████████████████████
#
#  Design rules:
#    1. Every call has ONE system prompt and ONE user prompt.
#    2. A validator function checks the raw string output.
#    3. On first failure, a self-correction prompt is constructed from
#       the model's own output + the error message → one more attempt.
#    4. If self-correction fails, the next model in the cascade is tried.
#    5. If all models fail, Gemini Flash is the absolute last resort.
#    6. Returns None only if everything fails.
# ═══════════════════════════════════════════════════════════════════════════

def single_task_llm(
    task_name:   str,
    sys_prompt:  str,
    user_prompt: str,
    validator:   Callable[[str], tuple[bool, str]],
    sota_models: list[str],
    temperature: float = 0.7,
) -> str | None:
    """
    The core LLM caller. One task. One validation. One self-correction.

    Parameters
    ----------
    task_name   : Human-readable name shown in logs.
    sys_prompt  : Narrow, single-role system instruction.
    user_prompt : Single, specific question / instruction.
    validator   : fn(output: str) → (is_valid: bool, error_msg: str).
    sota_models : Ordered list of model IDs to try.
    temperature : Creativity dial — lower for facts, higher for writing.

    Returns
    -------
    Validated string output, or None if all attempts exhausted.
    """
    print(f"  🔧 Task: {task_name}")
    STRICT_SUFFIX = (
        "\n\nCRITICAL: Return ONLY the exact output requested. "
        "No preamble. No explanation. No markdown. No apologies."
    )

    def _call_openrouter(model: str, prompt: str) -> str | None:
        if not OPENROUTER_KEY:
            return None
        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": prompt + STRICT_SUFFIX},
            ],
        }
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload, timeout=50,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
        return None

    def _call_gemini_flash(prompt: str) -> str | None:
        if not GEMINI_KEY:
            return None
        try:
            client = genai.Client(api_key=GEMINI_KEY)
            cfg = types.GenerateContentConfig(
                system_instruction=sys_prompt, temperature=temperature
            )
            rsp = client.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=prompt + STRICT_SUFFIX,
                config=cfg,
            )
            return rsp.text.strip()
        except Exception:
            return None

    # ── Try each OpenRouter model ────────────────────────────────────────
    for model in sota_models:
        raw = _call_openrouter(model, user_prompt)
        if raw:
            valid, err = validator(raw)
            if valid:
                print(f"    ✅ {task_name} → {model}")
                return raw
            # Self-correction: give the model its mistake and one more chance
            correction_prompt = (
                f"Your previous response was rejected.\n"
                f"PREVIOUS RESPONSE:\n{raw}\n\n"
                f"REJECTION REASON: {err}\n\n"
                f"Please fix this and return only the corrected version."
            )
            raw2 = _call_openrouter(model, correction_prompt)
            if raw2:
                valid2, _ = validator(raw2)
                if valid2:
                    print(f"    ✅ {task_name} (self-corrected) → {model}")
                    return raw2
        time.sleep(3)

    # ── Gemini Flash absolute fallback ───────────────────────────────────
    raw = _call_gemini_flash(user_prompt)
    if raw:
        valid, _ = validator(raw)
        if valid:
            print(f"    ✅ {task_name} → Gemini Flash fallback")
            return raw

    print(f"    ❌ {task_name} — all attempts exhausted.")
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  VALIDATORS
#  Each validator checks one specific contract.
#  Returns (True, "") on success or (False, "reason") on failure.
# ═══════════════════════════════════════════════════════════════════════════

def _check_banned(text: str) -> tuple[bool, str]:
    """Returns (False, msg) if any banned phrase found, else (True, '')."""
    tl = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase in tl:
            return False, f"Contains banned phrase: '{phrase}'"
    return True, ""


def validate_study_name(text: str) -> tuple[bool, str]:
    """Must be a non-empty string under 100 chars with at least one year or name."""
    t = text.strip()
    if not t:
        return False, "Empty response."
    if len(t) > 120:
        return False, f"Too long ({len(t)} chars). Must be a study name only."
    if len(t.split()) < 3:
        return False, "Too short — must include researcher name or study context."
    return True, ""


def validate_key_facts(text: str) -> tuple[bool, str]:
    """Must be 4-6 numbered lines, each containing at least one number or percentage."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if len(lines) < 4 or len(lines) > 6:
        return False, f"Expected 4-6 facts, got {len(lines)}."
    for i, line in enumerate(lines):
        if not re.search(r"\d", line):
            return False, f"Fact {i+1} contains no number or percentage: '{line}'"
    return True, ""


def validate_contradiction(text: str) -> tuple[bool, str]:
    """Must have exactly 2 sentences on 2 lines."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if len(lines) != 2:
        return False, f"Expected exactly 2 lines (expected / actual), got {len(lines)}."
    if len(lines[0].split()) < 5 or len(lines[1].split()) < 5:
        return False, "Both lines must be complete sentences."
    return True, ""


def validate_era(text: str) -> tuple[bool, str]:
    """Must be exactly one of the allowed era strings."""
    allowed = set(ERA_STYLES.keys())
    t = text.strip()
    if t not in allowed:
        return False, f"'{t}' is not a valid era. Must be one of: {sorted(allowed)}"
    return True, ""


def validate_single_line(min_w: int = 6, max_w: int = 24) -> Callable:
    """Returns a validator for a single sentence within a word range."""
    def _v(text: str) -> tuple[bool, str]:
        t = text.strip().strip('"').strip("'")
        if "\n" in t:
            return False, "Must be a single line (no newlines)."
        wc = len(t.split())
        if wc < min_w:
            return False, f"Too short ({wc} words). Minimum: {min_w}."
        if wc > max_w:
            return False, f"Too long ({wc} words). Maximum: {max_w}."
        ok, err = _check_banned(t)
        if not ok:
            return False, err
        return True, ""
    return _v


def validate_n_lines(n: int, min_w: int = 5, max_w: int = 22) -> Callable:
    """Returns a validator for exactly N lines, each within a word range."""
    def _v(text: str) -> tuple[bool, str]:
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        # Strip leading numbering (1. / 1) / - )
        lines = [re.sub(r"^[\d\-\.\)]+\s*", "", l) for l in lines]
        if len(lines) != n:
            return False, f"Expected exactly {n} lines, got {len(lines)}."
        for i, line in enumerate(lines):
            wc = len(line.split())
            if wc < min_w:
                return False, f"Line {i+1} too short ({wc} words)."
            if wc > max_w:
                return False, f"Line {i+1} too long ({wc} words)."
            ok, err = _check_banned(line)
            if not ok:
                return False, f"Line {i+1}: {err}"
        return True, ""
    return _v


def validate_ssml_line(clean_text: str) -> Callable:
    """
    Validates one SSML acting_text line.
    - Must contain at least 1 SSML tag.
    - Must not contain escaped tags (&lt;).
    - All words from clean_text must appear in the output.
    """
    def _v(text: str) -> tuple[bool, str]:
        t = text.strip()
        if not re.search(r"<(break|emphasis|prosody)[^>]*>", t):
            return False, "Missing SSML tag. Must include at least one <break>, <emphasis>, or <prosody>."
        if "&lt;" in t or "&gt;" in t:
            return False, "SSML tags are escaped. Use raw angle brackets < >."
        # Check key words present (ignoring punctuation)
        clean_words = set(re.sub(r"[^\w\s]", "", clean_text.lower()).split())
        tagged_words = set(re.sub(r"<[^>]+>", " ", t).lower().split())
        missing = clean_words - tagged_words - {"a", "an", "the", "of", "in", "to"}
        if len(missing) > 2:
            return False, f"Words from original text missing: {missing}"
        return True, ""
    return _v


def validate_speaker_json(text: str) -> tuple[bool, str]:
    """Must parse as JSON list of objects, each with 'speaker' field."""
    try:
        raw = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        if not isinstance(data, list):
            return False, "Must be a JSON array."
        allowed = {"narrator", "document", "witness", "researcher"}
        for i, item in enumerate(data):
            if "speaker" not in item:
                return False, f"Item {i} missing 'speaker' field."
            if item["speaker"] not in allowed:
                return False, f"Item {i} has invalid speaker: '{item['speaker']}'."
        return True, ""
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"


def validate_key_stat_json(text: str) -> tuple[bool, str]:
    """Must parse as JSON with 'stat' (contains digit) and 'context' (≤8 words)."""
    try:
        raw = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        if "stat" not in data or "context" not in data:
            return False, "Must have 'stat' and 'context' fields."
        if not re.search(r"\d", data["stat"]):
            return False, "'stat' must contain a number or percentage."
        if len(data["context"].split()) > 10:
            return False, "'context' must be 10 words or fewer."
        return True, ""
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"


def validate_visual_prompt_json(text: str) -> tuple[bool, str]:
    """
    Must parse as JSON with 'search_query' (string or list) and 'ai_prompt' (string).
    Handles the case where a model returns search_query as a JSON array.
    """
    try:
        raw = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        if "search_query" not in data or "ai_prompt" not in data:
            return False, "Must have 'search_query' and 'ai_prompt' fields."

        # Handle search_query returned as a list ["word1", "word2"]
        sq = data["search_query"]
        if isinstance(sq, list):
            sq = " ".join(str(item) for item in sq)
            data["search_query"] = sq  # normalise in place

        if not isinstance(sq, str):
            return False, f"'search_query' must be a string, got {type(sq).__name__}."

        if len(sq.split()) > 8:
            return False, f"'search_query' too long ({len(sq.split())} words). Max 8."

        if len(str(data["ai_prompt"])) < 20:
            return False, "'ai_prompt' too short (min 20 chars)."

        return True, ""
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"

def validate_youtube_title(text: str) -> tuple[bool, str]:
    """Must be ≤ 55 chars and not start with a banned phrase."""
    t = text.strip().strip('"')
    if len(t) > 60:
        return False, f"Title too long ({len(t)} chars). Max 60."
    if len(t) < 10:
        return False, "Title too short."
    ok, err = _check_banned(t)
    return ok, err


def validate_description(text: str) -> tuple[bool, str]:
    """Must be 2-4 sentences."""
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    if len(sentences) < 2:
        return False, "Too short — need at least 2 sentences."
    if len(sentences) > 5:
        return False, "Too long — max 4 sentences."
    return True, ""


def validate_tags(text: str) -> tuple[bool, str]:
    """Must be 8-12 comma-separated tags."""
    tags = [t.strip() for t in text.split(",") if t.strip()]
    if len(tags) < 8:
        return False, f"Too few tags ({len(tags)}). Need 8-12."
    if len(tags) > 14:
        return False, f"Too many tags ({len(tags)}). Need 8-12."
    return True, ""


def validate_caption(min_words: int = 20) -> Callable:
    """Returns a validator checking minimum word count for social captions."""
    def _v(text: str) -> tuple[bool, str]:
        wc = len(text.split())
        if wc < min_words:
            return False, f"Caption too short ({wc} words). Min {min_words}."
        return True, ""
    return _v


def validate_music_keywords(text: str) -> tuple[bool, str]:
    """2-4 words for Pixabay music search."""
    words = text.strip().split()
    if len(words) < 2 or len(words) > 6:
        return False, f"Expected 2-4 keyword words, got {len(words)}."
    return True, ""


# ═══════════════════════════════════════════════════════════════════════════
#  WEB SCRAPERS  (no LLM — deterministic, fast, free)
# ═══════════════════════════════════════════════════════════════════════════
_WEB_UA = {"User-Agent": "TheHumanExperimentBot/1.0 (Educational Documentary)"}


def scrape_wikipedia(query: str) -> str:
    """
    Fetches up to 4 500 chars of plain-text article extract from Wikipedia.
    Returns empty string on any failure.
    """
    print(f"    📚 Wikipedia → '{query}'")
    try:
        sr = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "format": "json",
                    "list": "search", "srsearch": query, "srlimit": 1},
            headers=_WEB_UA, timeout=12,
        )
        results = sr.json().get("query", {}).get("search", [])
        if not results:
            return ""
        title = results[0]["title"]
        er = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "format": "json", "prop": "extracts",
                    "titles": title, "exintro": False,
                    "explaintext": True, "exsectionformat": "plain"},
            headers=_WEB_UA, timeout=15,
        )
        pages = er.json().get("query", {}).get("pages", {})
        for _, page in pages.items():
            return page.get("extract", "")[:4500]
    except Exception as e:
        print(f"    ⚠️  Wikipedia scrape error: {e}")
    return ""


def scrape_google_news_rss(query: str) -> str:
    """
    Fetches up to 10 headlines + snippets from Google News RSS (no API key).
    Returns empty string on any failure.
    """
    print(f"    📰 Google News RSS → '{query}'")
    try:
        q   = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        r   = requests.get(url, timeout=12, headers=_WEB_UA)
        root  = ET.fromstring(r.content)
        items = root.findall(".//item")[:6]
        lines = []
        for item in items:
            for tag in ("title", "description"):
                el = item.find(tag)
                if el is not None and el.text:
                    clean = (el.text.replace("<b>", "").replace("</b>", "")
                             .replace("&nbsp;", " ").replace("&amp;", "&"))
                    lines.append(clean[:250])
        return "\n".join(lines)
    except Exception as e:
        print(f"    ⚠️  Google News RSS error: {e}")
    return ""


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 0 — RESEARCH TASKS
#  4 LLM tasks, each with a single, narrow job
# ═══════════════════════════════════════════════════════════════════════════

# ── Shared system prompt for all research tasks ──────────────────────────
_RESEARCH_SYS = (
    "You are a research archivist specialising in psychology and behavioural science. "
    "You are precise, factual, and economical. You never add commentary. "
    "You return only the exact output format requested."
)


def task_propose_study(domain: str, past_topics: str, sota_models: list[str]) -> str | None:
    """
    Task 0.1 — ONE job: propose one real, researchable psychology study.

    Returns the study name (e.g. "Milgram Obedience Experiment, 1963").
    """
    avoid = f"\nDo NOT suggest any study from:\n{past_topics}\n" if past_topics else ""
    prompt = (
        f"Domain: '{domain}'\n{avoid}\n"
        "Name ONE specific, real, documented psychology study, experiment, or research paper "
        "in this domain that has a Wikipedia article.\n"
        "Include the lead researcher's surname and the year.\n"
        "Return ONLY the name, e.g.: 'Milgram Obedience Experiment, 1963'"
    )
    return single_task_llm(
        task_name   = "Propose Study",
        sys_prompt  = _RESEARCH_SYS,
        user_prompt = prompt,
        validator   = validate_study_name,
        sota_models = sota_models,
        temperature = 0.75,
    )


def task_extract_key_facts(research_brief: str, sota_models: list[str]) -> str | None:
    """
    Task 0.2 — ONE job: extract 5 specific, numbered facts from the research.

    Each fact must contain at least one number, date, or percentage.
    Returns a newline-separated numbered list.
    """
    prompt = (
        "From the research text below, extract EXACTLY 5 specific, documented facts.\n"
        "Rules:\n"
        "  - Each fact must contain at least one number, percentage, or date.\n"
        "  - Use only information present in the text — do not add knowledge.\n"
        "  - Each fact must be one sentence.\n"
        "  - Number them 1-5.\n\n"
        f"RESEARCH TEXT:\n{research_brief[:3000]}"
    )
    return single_task_llm(
        task_name   = "Extract Key Facts",
        sys_prompt  = _RESEARCH_SYS,
        user_prompt = prompt,
        validator   = validate_key_facts,
        sota_models = sota_models,
        temperature = 0.15,
    )


def task_identify_contradiction(key_facts: str, study_name: str, sota_models: list[str]) -> str | None:
    """
    Task 0.3 — ONE job: identify the gap between expected and actual results.

    Returns exactly 2 lines:
      Line 1: What researchers expected to find.
      Line 2: What they actually found.
    """
    prompt = (
        f"Study: {study_name}\n\n"
        f"Facts:\n{key_facts}\n\n"
        "From these facts, identify the most striking gap between what was expected and what was found.\n"
        "Return EXACTLY 2 lines:\n"
        "Line 1: What the researchers expected (one sentence).\n"
        "Line 2: What they actually found (one sentence, must include a number or fact).\n"
        "Do not include labels like 'Expected:' or 'Actual:'."
    )
    return single_task_llm(
        task_name   = "Identify Contradiction",
        sys_prompt  = _RESEARCH_SYS,
        user_prompt = prompt,
        validator   = validate_contradiction,
        sota_models = sota_models,
        temperature = 0.2,
    )


def task_detect_era(research_brief: str, sota_models: list[str]) -> str | None:
    """
    Task 0.4 — ONE job: detect the decade of the study.
    Returns exactly one of the ERA_STYLES keys.
    Robust to models wrapping the answer in extra text.
    """
    allowed = list(ERA_STYLES.keys())

    def _fuzzy_era_validator(text: str) -> tuple[bool, str]:
        """
        Scans the response for any valid era string anywhere in the text.
        This handles models that return 'The era is: 1970s-1980s' instead
        of just '1970s-1980s'.
        """
        t = text.strip()
        # First try exact match
        if t in allowed:
            return True, ""
        # Then scan for any allowed value inside the text
        for era in allowed:
            if era in t:
                return True, ""
        return False, (
            f"Could not find a valid era in response: '{t[:80]}'. "
            f"Must contain one of: {allowed}"
        )

    prompt = (
        f"Read this research text and identify which era the study took place in.\n"
        f"Return ONLY one of these exact strings, nothing else: {allowed}\n\n"
        f"RESEARCH TEXT:\n{research_brief[:1500]}"
    )

    raw = single_task_llm(
        task_name   = "Detect Era",
        sys_prompt  = _RESEARCH_SYS,
        user_prompt = prompt,
        validator   = _fuzzy_era_validator,
        sota_models = sota_models,
        temperature = 0.0,
    )

    if not raw:
        # Regex fallback — extract from raw text using year detection
        years = re.findall(r"\b(1[89]\d{2}|20[012]\d)\b", research_brief)
        if years:
            earliest = min(int(y) for y in years)
            if   earliest < 1935: return "1900s-1930s"
            elif earliest < 1965: return "1940s-1960s"
            elif earliest < 1990: return "1970s-1980s"
            elif earliest < 2010: return "1990s-2000s"
            else:                 return "modern"
        return "unknown"

    # Extract the matched era from wherever it appears in the response
    for era in allowed:
        if era in raw:
            return era
    return raw.strip() if raw.strip() in allowed else "unknown"

# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1 — WRITING TASKS
#  5 LLM tasks — chained sequentially, each builds on the previous output
#
#  Shared channel persona injected into every writing system prompt:
# ═══════════════════════════════════════════════════════════════════════════

_WRITER_SYS = """You are the head writer for "The Human Experiment" — a YouTube channel \
that documents what happens when scientists study human behaviour and find something \
nobody expected.

Your voice is that of a research analyst: precise, controlled, never sensational. \
You state findings and let the data speak. You never editorialize.

You make the viewer feel three things, in this order:
  1. Certainty — they know something about human nature.
  2. Doubt — the study suggests they may be wrong about themselves.
  3. Unease — the implications extend far beyond the lab.

ABSOLUTE RULES (never broken, ever):
  - Never use these phrases: {banned}
  - Every sentence must be completable in one breath when spoken aloud.
  - Numbers and percentages from the research must be stated precisely.
  - The researcher's name and year must appear at least once in the full script.
  - Do not use hyperbole. State what happened. Nothing more.
""".format(banned=", ".join(f"'{p}'" for p in BANNED_PHRASES))


def task_write_hook(
    key_facts: str,
    contradiction: str,
    study_name: str,
    sota_models: list[str],
) -> str | None:
    """
    Task 1.1 — ONE job: write the single opening line.

    The hook must state the most impossible or counterintuitive result
    as a plain declarative sentence. No scene-setting. No names yet.
    8-22 words.
    """
    prompt = (
        f"Study: {study_name}\n\n"
        f"Key facts:\n{key_facts}\n\n"
        f"The central contradiction:\n{contradiction}\n\n"
        "Write ONLY the opening line of a short documentary script about this study.\n"
        "Requirements:\n"
        "  - State the most counterintuitive result as a plain fact.\n"
        "  - No names, no dates, no scene-setting — just the finding.\n"
        "  - 8 to 22 words.\n"
        "  - Must make a person stop scrolling.\n"
        "  - One sentence only.\n"
        "Example style (do not copy): "
        "'Sixty-five percent of ordinary people will cross a line they believe is monstrous, "
        "given one specific condition.'"
    )
    return single_task_llm(
        task_name   = "Write Hook",
        sys_prompt  = _WRITER_SYS,
        user_prompt = prompt,
        validator   = validate_single_line(min_w=8, max_w=22),
        sota_models = sota_models,
        temperature = 0.92,
    )


def task_write_escalation(
    hook_line: str,
    key_facts: str,
    study_name: str,
    sota_models: list[str],
) -> str | None:
    """
    Task 1.2 — ONE job: write exactly 4 escalation lines (lines 2-5).

    These reveal the study setup detail by detail — like tightening a vice.
    Each line must be shorter than 20 words and build tension.
    Return as 4 plain lines (no numbering, no bullets).
    """
    prompt = (
        f"Study: {study_name}\n\n"
        f"Opening line already written:\n'{hook_line}'\n\n"
        f"Key facts to draw from:\n{key_facts}\n\n"
        "Write EXACTLY 4 lines that follow the opening line.\n"
        "These lines reveal the study context step by step — who ran it, "
        "what participants were told, what they believed was happening.\n"
        "Requirements:\n"
        "  - Each line: 6-18 words, one complete sentence.\n"
        "  - Increasing tension — each line should feel heavier than the last.\n"
        "  - Use at least one specific number from the facts.\n"
        "  - No line can start with the word 'The' more than once.\n"
        "  - Return as 4 plain lines, no numbering."
    )
    result = single_task_llm(
        task_name   = "Write Escalation",
        sys_prompt  = _WRITER_SYS,
        user_prompt = prompt,
        validator   = validate_n_lines(n=4, min_w=6, max_w=18),
        sota_models = sota_models,
        temperature = 0.88,
    )
    if result:
        return [l.strip() for l in result.strip().split("\n") if l.strip()]
    return None


def task_write_contradiction_lines(
    hook_line:         str,
    escalation_lines:  list[str],
    contradiction:     str,
    study_name:        str,
    sota_models:       list[str],
) -> str | None:
    """
    Task 1.3 — ONE job: write exactly 3 lines showing the expected vs actual gap.

    These are the emotional pivot of the entire script.
    Line 1: What was predicted / expected.
    Line 2: What actually happened (must include the key number).
    Line 3: The implication — one sentence on what this means for people outside the lab.
    Return as 3 plain lines.
    """
    built_so_far = "\n".join([hook_line] + escalation_lines)
    prompt = (
        f"Study: {study_name}\n\n"
        f"Script written so far:\n{built_so_far}\n\n"
        f"The documented contradiction:\n{contradiction}\n\n"
        "Write EXACTLY 3 lines that form the emotional pivot of this script.\n"
        "Line 1: What the researchers predicted would happen (state it confidently).\n"
        "Line 2: What actually happened — must include the specific documented result.\n"
        "Line 3: What this means for anyone outside that lab — one sentence.\n"
        "Requirements:\n"
        "  - 8-20 words each.\n"
        "  - Line 2 must include at least one specific number or percentage.\n"
        "  - Line 3 must not mention the researchers — speak directly to the viewer's reality.\n"
        "  - Return as 3 plain lines, no numbering, no labels."
    )
    result = single_task_llm(
        task_name   = "Write Contradiction",
        sys_prompt  = _WRITER_SYS,
        user_prompt = prompt,
        validator   = validate_n_lines(n=3, min_w=8, max_w=20),
        sota_models = sota_models,
        temperature = 0.85,
    )
    if result:
        return [l.strip() for l in result.strip().split("\n") if l.strip()]
    return None


def task_write_impossible_detail(
    key_facts:   str,
    study_name:  str,
    built_script: str,
    sota_models: list[str],
) -> str | None:
    """
    Task 1.4 — ONE job: write 1-2 lines containing the most disturbing implication.

    This is the detail that makes the viewer reconsider everything.
    It must come from the documented facts — not speculation.
    """
    prompt = (
        f"Study: {study_name}\n\n"
        f"Script written so far:\n{built_script}\n\n"
        f"Documented facts:\n{key_facts}\n\n"
        "Write 1 or 2 lines containing the single most disturbing or counterintuitive "
        "implication revealed by this study.\n"
        "Requirements:\n"
        "  - Must be grounded in the documented facts — no speculation.\n"
        "  - Must create a moment of personal relevance: the viewer should feel this applies to them.\n"
        "  - 8-20 words per line.\n"
        "  - If 2 lines, they must flow as one continuous thought.\n"
        "  - No horror language, no exaggeration — state the implication clinically.\n"
        "  - Return as 1 or 2 plain lines, no numbering."
    )
    result = single_task_llm(
        task_name   = "Write Impossible Detail",
        sys_prompt  = _WRITER_SYS,
        user_prompt = prompt,
        validator   = _validate_one_or_two_lines,
        sota_models = sota_models,
        temperature = 0.88,
    )
    if result:
        return [l.strip() for l in result.strip().split("\n") if l.strip()]
    return None

def _validate_one_or_two_lines(text: str) -> tuple[bool, str]:
    """Accepts 1 or 2 lines, each 6-22 words. Used for impossible detail."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    lines = [re.sub(r"^[\d\-\.\)]+\s*", "", l) for l in lines]
    if len(lines) < 1 or len(lines) > 2:
        return False, f"Expected 1 or 2 lines, got {len(lines)}."
    for i, line in enumerate(lines):
        wc = len(line.split())
        if wc < 5:
            return False, f"Line {i+1} too short ({wc} words)."
        if wc > 24:
            return False, f"Line {i+1} too long ({wc} words)."
        ok, err = _check_banned(line)
        if not ok:
            return False, f"Line {i+1}: {err}"
    return True, ""

def task_write_loop_ending(
    hook_line:     str,
    built_script:  str,
    study_name:    str,
    sota_models:   list[str],
) -> str | None:
    """
    Task 1.5 — ONE job: write the single closing line that loops back to the hook.

    The last sentence must echo the first sentence — either the same word,
    the same concept, or a direct inversion that completes the circle.
    8-18 words.
    """
    prompt = (
        f"Study: {study_name}\n\n"
        f"Opening line:\n'{hook_line}'\n\n"
        f"Full script so far:\n{built_script}\n\n"
        "Write ONLY the final closing line of this script.\n"
        "Requirements:\n"
        "  - Must loop back to the opening line — echo its first word, its central image, "
        "or invert its statement in a way that reframes everything.\n"
        "  - 8-18 words.\n"
        "  - Must feel like the script has come full circle — the viewer is back at the start "
        "but now they understand something different.\n"
        "  - One sentence only.\n"
        "  - Do not use the word 'remember' or 'think about'."
    )
    return single_task_llm(
        task_name   = "Write Loop Ending",
        sys_prompt  = _WRITER_SYS,
        user_prompt = prompt,
        validator   = validate_single_line(min_w=8, max_w=18),
        sota_models = sota_models,
        temperature = 0.80,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1b — VOICE DIRECTION TASKS
#  Processed per-line to prevent any ambiguity about which tag goes where
# ═══════════════════════════════════════════════════════════════════════════

_SSML_SYS = (
    "You are a voice director for documentary narration. "
    "Your ONLY job is to add SSML pacing tags to a single spoken line. "
    "The tags are stage directions — they must never be spoken aloud. "
    "You never rewrite the content. You never add words. "
    "You only add: <break time='Xs'/>, <emphasis level='strong'>WORD</emphasis>, "
    "or <prosody rate='slow' pitch='-12%'>PHRASE</prosody>."
)


def task_add_ssml_to_line(
    clean_text:       str,
    style_instruction: str,
    line_index:        int,
    sota_models:       list[str],
) -> str:
    """
    Task 1.6 — ONE job: add SSML to ONE line only.

    Processes lines one at a time to prevent cross-line confusion.
    Falls back to the clean_text if SSML generation fails.
    """
    prompt = (
        f"LINE TO DIRECT (line {line_index}):\n'{clean_text}'\n\n"
        f"VOCAL STYLE: {style_instruction}\n\n"
        "Add SSML pacing tags to this single line.\n"
        "Rules:\n"
        "  - Add 1-3 SSML tags maximum. Do not over-tag.\n"
        "  - <break time='Xs'/> goes BEFORE the most important word or reveal.\n"
        "  - <emphasis level='strong'> wraps ONE key word — the one that carries the most weight.\n"
        "  - <prosody rate='slow' pitch='-12%'> wraps a phrase only when the style calls for dread.\n"
        "  - Do NOT change any words. Do NOT add words.\n"
        "  - Return ONLY the SSML-tagged version of the line. Nothing else."
    )
    result = single_task_llm(
        task_name   = f"SSML Line {line_index}",
        sys_prompt  = _SSML_SYS,
        user_prompt = prompt,
        validator   = validate_ssml_line(clean_text),
        sota_models = sota_models,
        temperature = 0.4,
    )
    return result if result else clean_text  # safe fallback


_SPEAKER_SYS = (
    "You are a documentary casting director. "
    "You assign one of four voice roles to each line of a script: "
    "narrator, document, witness, or researcher. "
    "You return ONLY valid JSON. Nothing else."
)

_SPEAKER_ROLES_GUIDE = """
narrator  : The main analytical voice. World-weary, precise, controlled.
            Used for: scene-setting, implications, connecting ideas, the hook, the loop ending.
document  : Reads official study methodology, published findings, or institutional statements.
            Cold, flat, bureaucratic. Used for: the experimental procedure, official results.
witness   : Quotes from participants, observers, or secondary researchers. 
            Personal, sometimes shaken. Used for: direct quotes, participant reactions.
researcher: The lead researcher's voice before the result was known — initially confident.
            Used for: what they expected, their hypothesis, their public statements afterward.
"""


def task_assign_speakers(
    script_lines: list[str],
    sota_models:  list[str],
) -> list[dict] | None:
    """
    Task 1.7 — ONE job: assign speaker roles to each line.

    Takes plain clean_text lines, returns JSON list with speaker field added.
    Rules enforced: narrator ≥ 40% of lines, no speaker used 3× in a row.
    """
    lines_json = json.dumps([{"index": i, "text": l} for i, l in enumerate(script_lines)], indent=2)
    prompt = (
        f"SPEAKER ROLES:\n{_SPEAKER_ROLES_GUIDE}\n\n"
        f"SCRIPT LINES:\n{lines_json}\n\n"
        "Assign one speaker role to each line.\n"
        "Rules:\n"
        "  - 'narrator' must be used for at least 40% of lines.\n"
        "  - No speaker may appear more than 2 times consecutively.\n"
        "  - 'document' and 'researcher' each appear at least once.\n"
        "  - Return a JSON array where each object has 'index' and 'speaker' fields only.\n"
        "Example: [{\"index\": 0, \"speaker\": \"narrator\"}, {\"index\": 1, \"speaker\": \"document\"}]"
    )
    result = single_task_llm(
        task_name   = "Assign Speakers",
        sys_prompt  = _SPEAKER_SYS,
        user_prompt = prompt,
        validator   = validate_speaker_json,
        sota_models = sota_models,
        temperature = 0.1,
    )
    if result:
        try:
            raw = result.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception:
            pass
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1c — EXTRACTION TASK
# ═══════════════════════════════════════════════════════════════════════════

_EXTRACT_SYS = (
    "You are a data extraction specialist. "
    "You identify the single most striking quantitative finding in a script. "
    "You return only valid JSON."
)


def task_extract_key_statistic(
    full_script_text: str,
    sota_models:      list[str],
) -> dict:
    """
    Task 1.8 — ONE job: extract the thumbnail statistic.

    Returns dict with 'stat' (the number, e.g. "65%") and
    'context' (≤8 words, e.g. "of normal people crossed the line").
    Falls back to a safe default if extraction fails.
    """
    prompt = (
        f"SCRIPT:\n{full_script_text}\n\n"
        "Find the single most striking number, percentage, or statistic in this script.\n"
        "Return a JSON object with exactly two fields:\n"
        "  'stat': the number only (e.g. '65%' or '26 minutes' or '1 in 3')\n"
        "  'context': a phrase of 6-8 words explaining what it measured "
        "(e.g. 'of participants continued past their objections')\n"
        "Return ONLY the JSON object."
    )
    result = single_task_llm(
        task_name   = "Extract Key Statistic",
        sys_prompt  = _EXTRACT_SYS,
        user_prompt = prompt,
        validator   = validate_key_stat_json,
        sota_models = sota_models,
        temperature = 0.1,
    )
    if result:
        try:
            raw = result.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception:
            pass
    return {"stat": "", "context": "of participants in the original study"}


# ═══════════════════════════════════════════════════════════════════════════
#  SCRIPT ASSEMBLY
#  Combines all writing task outputs into the final structured script
# ═══════════════════════════════════════════════════════════════════════════

def assemble_script(
    study_name:         str,
    era:                str,
    hook:               str,
    escalation:         list[str],
    contradiction:      list[str],
    impossible_detail:  list[str],
    loop_ending:        str,
    sota_models:        list[str],
) -> dict | None:
    """
    Assembles all writing task outputs into the final script structure.
    Adds SSML (one line at a time) and speaker assignments.
    """
    # Build ordered clean_text list
    raw_lines: list[str] = (
        [hook]
        + (escalation or [])
        + (contradiction or [])
        + (impossible_detail or [])
        + ([loop_ending] if loop_ending else [])
    )

    if not raw_lines:
        print("❌ No script lines assembled.")
        return None

    # ── Infer style instructions per line position ───────────────────────
    style_map: dict[int, str] = {}
    total = len(raw_lines)
    style_map[0] = "Measured, controlled. State the finding as cold fact. No emotion."
    for i in range(1, min(5, total)):
        style_map[i] = "Slightly faster. Building. Each sentence heavier than the last."
    for i in range(5, min(8, total)):
        style_map[i] = "Slow down. This is the pivot. Precise. Weighted."
    for i in range(8, total - 1):
        style_map[i] = "Low pitch. Clinical dread. The implication, not the drama."
    if total > 1:
        style_map[total - 1] = "Hushed. Deliberate. Let the final word hang."

    # ── Add SSML to each line individually (Task 1.6, looped) ───────────
    print("  🎙️ Adding SSML voice direction (one line at a time)...")
    acting_lines: list[str] = []
    for i, clean in enumerate(raw_lines):
        style = style_map.get(i, "Measured, precise narration.")
        acting = task_add_ssml_to_line(clean, style, i, sota_models)
        acting_lines.append(acting)
        time.sleep(1.5)  # Rate-limit buffer between per-line calls

    # ── Assign speakers (Task 1.7) ───────────────────────────────────────
    print("  🎭 Assigning speaker roles...")
    speaker_assignments = task_assign_speakers(raw_lines, sota_models)

    speaker_map: dict[int, str] = {}
    if speaker_assignments:
        for item in speaker_assignments:
            speaker_map[item["index"]] = item.get("speaker", "narrator")

    # ── Build final structured line objects ──────────────────────────────
    structured_lines: list[dict] = []
    for i, clean in enumerate(raw_lines):
        structured_lines.append({
            "speaker":           speaker_map.get(i, "narrator"),
            "style_instruction": style_map.get(i, "Measured, precise narration."),
            "clean_text":        clean,
            "acting_text":       acting_lines[i],
        })

    return {
        "case_name": study_name,
        "era":       era,
        "lines":     structured_lines,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 3 — VISUAL DIRECTION TASK
#  Generates one image prompt per visual slot (called in a loop)
# ═══════════════════════════════════════════════════════════════════════════

_VIS_SYS = (
    "You are a Documentary Cinematographer and Archival Researcher "
    "specialising in psychology and institutional settings. "
    "You create precise, diegetic visual prompts for FLUX.1 image generation "
    "and Wikipedia/archive image searches. "
    "You return only valid JSON."
)


def task_write_one_visual_prompt(
    slot_index:    int,
    total_slots:   int,
    script_text:   str,
    era:           str,
    era_texture:   str,
    sota_models:   list[str],
) -> dict:
    """
    Task 3.1 — ONE job: write ONE image prompt for ONE visual slot.

    Called in a loop — generates one prompt per slot independently.
    This eliminates the multi-output confusion of generating all prompts at once.
    """
    # Rotate shot types to ensure visual variety
    shot_types = [
        "Extreme Close-Up of a physical object",
        "Wide establishing shot of a room or space",
        "Over-the-shoulder angle suggesting observation",
        "Dutch angle — slightly tilted, creating unease",
        "Bird's-eye view looking directly down",
        "Low angle looking up at a figure or structure",
    ]
    shot = shot_types[slot_index % len(shot_types)]

    prompt = (
        f"Visual slot {slot_index + 1} of {total_slots}.\n\n"
        f"Script excerpt:\n'{script_text[:800]}'\n\n"
        f"Era of study: {era}\n"
        f"Mandatory visual texture: {era_texture}\n\n"
        f"Shot type for this slot: {shot}\n\n"
        "Return a JSON object with exactly two fields:\n"
        "  'search_query': 2-5 keywords for Wikipedia/archive image search "
        "(specific nouns: place names, equipment, researcher names, years).\n"
        "  'ai_prompt': A FLUX.1 image prompt. Must:\n"
        "    - Describe a PHYSICAL OBJECT in a PHYSICAL SPACE (no abstract concepts).\n"
        f"    - Use the shot type: {shot}.\n"
        "    - Include the era texture at the end.\n"
        "    - End with: 'vertical composition, cinematic documentary'\n"
        "    - NEVER include legible text, signs, or numbers in the scene.\n"
        "Return ONLY the JSON object."
    )
    result = single_task_llm(
        task_name   = f"Visual Prompt Slot {slot_index + 1}",
        sys_prompt  = _VIS_SYS,
        user_prompt = prompt,
        validator   = validate_visual_prompt_json,
        sota_models = sota_models,
        temperature = 0.80,
    )
    if result:
        try:
            raw = result.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            # Normalise search_query if it came back as a list
            if isinstance(data.get("search_query"), list):
                data["search_query"] = " ".join(str(x) for x in data["search_query"])
            return data
        except Exception:
            pass

    # Safe fallback
    return {
        "search_query": f"psychology research {era}",
        "ai_prompt": (
            f"Empty institutional corridor, observation window in wall, "
            f"{shot.lower()}, {era_texture}, vertical composition, cinematic documentary"
        ),
    }


def generate_all_visual_prompts(
    full_script_text: str,
    required_images:  int,
    era:              str,
    sota_models:      list[str],
) -> list[dict]:
    """Calls task_write_one_visual_prompt in a loop — one slot at a time."""
    print(f"🎬 Generating {required_images} visual prompts (one at a time)...")
    era_texture = ERA_STYLES.get(era, ERA_STYLES["unknown"])
    prompts = []
    for i in range(required_images):
        p = task_write_one_visual_prompt(
            slot_index=i,
            total_slots=required_images,
            script_text=full_script_text,
            era=era,
            era_texture=era_texture,
            sota_models=sota_models,
        )
        prompts.append(p)
        time.sleep(1.5)  # Rate-limit buffer
    return prompts


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 5 — METADATA TASKS  (5 separate single-task LLM calls)
# ═══════════════════════════════════════════════════════════════════════════

_META_SYS = (
    "You are an elite YouTube SEO Strategist for a psychology documentary channel. "
    "You write metadata that drives clicks without clickbait. "
    "You rely on curiosity gaps, not exaggeration. "
    "You return ONLY the exact output requested."
)


def task_write_youtube_title(
    full_script_text: str,
    key_stat:         dict,
    study_name:       str,
    sota_models:      list[str],
) -> str:
    """Task 5.1 — ONE job: write one YouTube title."""
    stat_line = f"Key statistic: {key_stat.get('stat','')} {key_stat.get('context','')}"
    prompt = (
        f"Study: {study_name}\n"
        f"{stat_line}\n\n"
        f"Script:\n{full_script_text[:600]}\n\n"
        "Write ONE YouTube Shorts title.\n"
        "Requirements:\n"
        "  - Under 55 characters.\n"
        "  - Creates a curiosity gap — does NOT reveal the result.\n"
        "  - Does not use the study name directly.\n"
        "  - Does not use clickbait words like 'shocking' or 'insane'.\n"
        "  - Speaks to what the viewer will understand about themselves.\n"
        "  - No hashtags. No quotes.\n"
        "Example style: 'Your brain makes this decision before you do'"
    )
    result = single_task_llm(
        task_name   = "YouTube Title",
        sys_prompt  = _META_SYS,
        user_prompt = prompt,
        validator   = validate_youtube_title,
        sota_models = sota_models,
        temperature = 0.88,
    )
    title = (result or "The study that changed everything").strip('"')
    return f"{title} #shorts"


def task_write_youtube_description(
    title:       str,
    study_name:  str,
    key_stat:    dict,
    sota_models: list[str],
) -> str:
    """Task 5.2 — ONE job: write a 3-sentence YouTube description."""
    prompt = (
        f"Video title: {title}\n"
        f"Study: {study_name}\n"
        f"Key finding: {key_stat.get('stat','')} {key_stat.get('context','')}\n\n"
        "Write a YouTube description for this psychology documentary Short.\n"
        "Requirements:\n"
        "  - Exactly 3 sentences.\n"
        "  - Sentence 1: What this study revealed (state the finding, not the question).\n"
        "  - Sentence 2: Why this matters to anyone watching — personal relevance.\n"
        "  - Sentence 3: A question that provokes comment debate.\n"
        "  - No hashtags in the description body.\n"
        "  - No promotional language."
    )
    result = single_task_llm(
        task_name   = "YouTube Description",
        sys_prompt  = _META_SYS,
        user_prompt = prompt,
        validator   = validate_description,
        sota_models = sota_models,
        temperature = 0.75,
    )
    return result or f"Research on {study_name} revealed findings that challenge assumptions about human behaviour."


def task_write_seo_tags(
    title:       str,
    study_name:  str,
    era:         str,
    sota_models: list[str],
) -> list[str]:
    """Task 5.3 — ONE job: write 10 SEO tags."""
    prompt = (
        f"Video title: {title}\n"
        f"Study: {study_name}\n"
        f"Era: {era}\n\n"
        "Write exactly 10 YouTube SEO tags for this psychology documentary Short.\n"
        "Mix 3 tiers:\n"
        "  Tier 1 (broad, 3 tags): psychology, human behavior, brain science\n"
        "  Tier 2 (mid, 4 tags): related to the specific study type\n"
        "  Tier 3 (specific, 3 tags): highly specific to this study and era\n"
        "Return as comma-separated values only. No hashtags. No quotes."
    )
    result = single_task_llm(
        task_name   = "SEO Tags",
        sys_prompt  = _META_SYS,
        user_prompt = prompt,
        validator   = validate_tags,
        sota_models = sota_models,
        temperature = 0.55,
    )
    if result:
        return [t.strip().replace("#", "") for t in result.split(",") if t.strip()]
    return ["psychology", "human behavior", "brain science", "experiment",
            "documentary", "mindset", "cognition", "social psychology", "science", "research"]


def task_write_ig_caption(
    title:       str,
    description: str,
    key_stat:    dict,
    sota_models: list[str],
) -> str:
    """Task 5.4 — ONE job: write one Instagram caption."""
    prompt = (
        f"Video title: {title}\n"
        f"Description: {description}\n"
        f"Key stat: {key_stat.get('stat','')} {key_stat.get('context','')}\n\n"
        "Write one Instagram Reels caption for a psychology documentary Short.\n"
        "Requirements:\n"
        "  - First line: a scroll-stopping statement (not a question). Under 10 words.\n"
        "  - 2-3 body sentences: tease the finding without revealing it fully.\n"
        "  - Final line: a debate-driving call-to-action.\n"
        "  - Exactly 6 hashtags on the last line.\n"
        "  - Hashtags must include: #psychology #shorts plus 4 specific ones.\n"
        "  - Total caption: 40-80 words."
    )
    result = single_task_llm(
        task_name   = "Instagram Caption",
        sys_prompt  = _META_SYS,
        user_prompt = prompt,
        validator   = validate_caption(min_words=30),
        sota_models = sota_models,
        temperature = 0.85,
    )
    return result or f"{title}\n\n{description}\n\n#psychology #shorts #humanbehavior #science #mind #experiment"


def task_write_fb_caption(
    title:       str,
    description: str,
    sota_models: list[str],
) -> str:
    """Task 5.5 — ONE job: write one Facebook caption."""
    prompt = (
        f"Video title: {title}\n"
        f"Description: {description}\n\n"
        "Write one Facebook Reels caption for a psychology documentary Short.\n"
        "Requirements:\n"
        "  - Open with a 'What would you have done in this situation?' style question.\n"
        "  - 2-3 conversational sentences about the study's findings.\n"
        "  - Close with one sentence encouraging comment debate.\n"
        "  - Exactly 3 hashtags.\n"
        "  - Warm but slightly unnerving tone — like a trusted friend sharing disturbing news.\n"
        "  - Total: 35-70 words."
    )
    result = single_task_llm(
        task_name   = "Facebook Caption",
        sys_prompt  = _META_SYS,
        user_prompt = prompt,
        validator   = validate_caption(min_words=25),
        sota_models = sota_models,
        temperature = 0.82,
    )
    return result or f"{title}\n\n{description}\n\n#psychology #humanbehavior #science"


def task_write_music_keywords(
    full_script_text: str,
    sota_models:      list[str],
) -> str:
    """Task (Music) — ONE job: generate 2-3 Pixabay music search keywords."""
    prompt = (
        f"Script:\n{full_script_text[:400]}\n\n"
        "Write 2-3 words for searching a stock music library for background music "
        "that fits this psychology documentary.\n"
        "Rules:\n"
        "  - Use ONLY musical/atmospheric adjectives and nouns.\n"
        "  - BANNED: psychology, murder, ghost, death, blood, experiment.\n"
        "  - GOOD examples: 'dark ambient drone', 'tension strings', 'minimal piano'.\n"
        "  - Return only the keywords. No explanation."
    )
    result = single_task_llm(
        task_name   = "Music Keywords",
        sys_prompt  = "You are a film music supervisor. Return only keyword data.",
        user_prompt = prompt,
        validator   = validate_music_keywords,
        sota_models = sota_models,
        temperature = 0.60,
    )
    return result or "dark ambient tension"


# ═══════════════════════════════════════════════════════════════════════════
#  4-LAYER TITANIUM IMAGE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def fetch_archive_image(query: str, filename: str) -> bool:
    """Layer 1: Wikipedia pageimages → Google CSE → Internet Archive."""
    print(f"  🏛️  [1/4] Archives: '{query[:40]}'")
    clean = " ".join(query.split()[:5])

    # Wikipedia pageimages
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "format": "json", "prop": "pageimages",
                    "generator": "search", "gsrsearch": clean,
                    "gsrlimit": 3, "pithumbsize": 1080},
            headers=_WEB_UA, timeout=12,
        )
        for _, page in r.json().get("query", {}).get("pages", {}).items():
            if "thumbnail" in page:
                data = requests.get(page["thumbnail"]["source"],
                                    headers=_WEB_UA, timeout=15).content
                with open(filename, "wb") as f: f.write(data)
                if os.path.getsize(filename) > 1000: return True
    except Exception: pass

    # Google Custom Search Engine
    if SEARCH_API_KEY and GOOGLE_CSE_ID:
        try:
            params = {"q": f"{clean} psychology research",
                      "cx": GOOGLE_CSE_ID, "key": SEARCH_API_KEY,
                      "searchType": "image", "num": 1, "safe": "active"}
            items = requests.get(
                "https://www.googleapis.com/customsearch/v1", params=params
            ).json().get("items", [])
            if items:
                data = requests.get(items[0]["link"],
                                    headers=_WEB_UA, timeout=15).content
                with open(filename, "wb") as f: f.write(data)
                if os.path.getsize(filename) > 1000: return True
        except Exception: pass

    # Internet Archive
    try:
        docs = requests.get(
            "https://archive.org/advancedsearch.php",
            params={"q": f'"{clean}" AND mediatype:image',
                    "fl": "identifier", "rows": 3, "output": "json"},
            headers=_WEB_UA, timeout=12,
        ).json().get("response", {}).get("docs", [])
        for doc in docs:
            iid = doc.get("identifier")
            if iid:
                data = requests.get(
                    f"https://archive.org/download/{iid}/{iid}.jpg",
                    headers=_WEB_UA, timeout=15).content
                if len(data) > 1000:
                    with open(filename, "wb") as f: f.write(data)
                    return True
    except Exception: pass
    return False


def fetch_cloudflare_image(prompt: str, filename: str) -> bool:
    """Layer 2: Cloudflare Workers AI — FLUX.1 Schnell."""
    print(f"  ☁️  [2/4] FLUX.1: '{prompt[:45]}...'")
    if not CF_ACCOUNT_ID or not CF_API_TOKEN: return False
    url = (f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}"
           f"/ai/run/@cf/black-forest-labs/flux-1-schnell")
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}",
               "Content-Type": "application/json"}
    try:
        r = requests.post(url, headers=headers,
                          json={"prompt": prompt}, timeout=55)
        if r.status_code == 200:
            ct = r.headers.get("Content-Type", "")
            if "application/json" in ct:
                b64 = r.json().get("result", {}).get("image")
                if b64:
                    with open(filename, "wb") as f:
                        f.write(base64.b64decode(b64))
                    return True
            else:
                with open(filename, "wb") as f: f.write(r.content)
                if os.path.getsize(filename) > 1000: return True
    except Exception: pass
    return False


def fetch_pexels_image(prompt: str, filename: str) -> bool:
    """Layer 3: Pexels stock photography."""
    print(f"  📷 [3/4] Pexels: '{prompt[:40]}...'")
    if not PEXELS_KEY: return False
    query = " ".join(prompt.split()[:5])
    try:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": PEXELS_KEY},
            params={"query": query, "per_page": 1, "orientation": "portrait"},
            timeout=30,
        )
        if r.status_code == 200:
            photos = r.json().get("photos", [])
            if photos:
                data = requests.get(photos[0]["src"]["large2x"], timeout=20).content
                with open(filename, "wb") as f: f.write(data)
                if os.path.getsize(filename) > 1000: return True
    except Exception: pass
    return False


def fetch_placeholder_image(filename: str, label: str = "") -> bool:
    """Layer 4: Pillow dark RGB fallback — pipeline never crashes."""
    try:
        img  = PIL.Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), COLOUR_BG)
        draw = PIL.ImageDraw.Draw(img)
        if label:
            fn = get_subtitle_font(36)
            draw.text((40, VIDEO_HEIGHT // 2), label[:40],
                      font=fn, fill=COLOUR_HANDLE_GREY)
        img.save(filename, "JPEG")
        return True
    except Exception: return False


def verify_and_convert_image(filename: str) -> bool:
    """Normalises any image to RGB JPEG and verifies it is readable."""
    try:
        with PIL.Image.open(filename) as img:
            img.load()
            img = img.convert("RGB")
            img.save(filename, format="JPEG", quality=95)
        return True
    except Exception: return False


def apply_diegetic_matting(filename: str) -> bool:
    """
    Wraps every image in one of three diegetic frames:
    polaroid / cinematic shadow / crt monitor.
    This makes every visual feel like it exists in a physical world.
    """
    try:
        with PIL.Image.open(filename) as img:
            img = img.convert("RGBA")
            tw, th = VIDEO_WIDTH, VIDEO_HEIGHT
            bg  = PIL.Image.new("RGBA", (tw, th), (*COLOUR_BG, 255))
            style = random.choice(["polaroid", "cinematic_shadow", "crt_monitor"])

            if style == "polaroid":
                img.thumbnail((440, 440), PIL.Image.Resampling.LANCZOS)
                fw, fh = img.width + 36, img.height + 110
                frame  = PIL.Image.new("RGBA", (fw, fh), (240, 240, 235, 255))
                frame.paste(img, (18, 18))
                frame  = frame.rotate(random.uniform(-4.5, 4.5),
                                      expand=True, fillcolor=(0, 0, 0, 0))
                ox = (tw - frame.width)  // 2
                oy = (th - frame.height) // 2
                bg.paste(frame, (ox, oy), frame)

            elif style == "cinematic_shadow":
                img.thumbnail((600, 820), PIL.Image.Resampling.LANCZOS)
                shadow = PIL.Image.new("RGBA", img.size, (0, 0, 0, 210))
                shadow = shadow.filter(PIL.ImageFilter.GaussianBlur(18))
                ox = (tw - img.width)  // 2
                oy = (th - img.height) // 2
                bg.paste(shadow, (ox + 18, oy + 18), shadow)
                bg.paste(img,    (ox, oy),            img)

            elif style == "crt_monitor":
                img.thumbnail((700, 1040), PIL.Image.Resampling.LANCZOS)
                d = PIL.ImageDraw.Draw(img)
                for y in range(0, img.height, 4):
                    d.line([(0, y), (img.width, y)], fill=(0, 0, 0, 65), width=1)
                ox = (tw - img.width)  // 2
                oy = (th - img.height) // 2
                bg.paste(img, (ox, oy), img)

            bg.convert("RGB").save(filename, format="JPEG", quality=95)
            return True
    except Exception as e:
        print(f"  ⚠️  Matting error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  PARALLAX ENGINE — eased, organic camera movement
# ═══════════════════════════════════════════════════════════════════════════

def _ease_inout(t: float) -> float:
    """Cosine S-curve. Maps linear 0→1 to organic ease-in/ease-out 0→1."""
    return (1 - math.cos(t * math.pi)) / 2


def generate_depth_map(image_path: str) -> str | None:
    """Runs Depth-Anything-V2-Small on CPU to produce a depth map."""
    print(f"  🧠 Depth map: {os.path.basename(image_path)}")
    try:
        estimator  = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device="cpu",
        )
        img        = PIL.Image.open(image_path).convert("RGB")
        depth_path = image_path.replace(".jpg", "_depth.jpg")
        estimator(img)["depth"].save(depth_path)
        return depth_path
    except Exception as e:
        print(f"  ⚠️  Depth map failed: {e}")
        return None


def apply_parallax_frame(
    t:           float,
    duration:    float,
    img_arr:     np.ndarray,
    depth_arr:   np.ndarray,
    direction:   str,
) -> np.ndarray:
    """Applies eased depth-mapped parallax shift to one frame."""
    max_shift = 30.0
    eased     = _ease_inout(t / duration)
    shift_px  = max_shift * (1.0 - eased) if direction == "left" else max_shift * eased

    norm_depth = depth_arr / 255.0
    shift_map  = (norm_depth * shift_px).astype(np.float32)

    h, w = img_arr.shape[:2]
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)
    for y in range(h):
        map_y[y, :] = y
        map_x[y, :] = np.arange(w) + shift_map[y, :]

    return cv2.remap(img_arr, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def get_image_clip(search_query: str, ai_prompt: str, duration: float, index: int):
    """
    Full Titanium Pipeline + diegetic matting + eased parallax + cross-dissolve.
    Tries all 4 layers in order, applies motion and transition, returns a MoviePy clip.
    """
    fname = f"temp_img_{index}.jpg"

    # Titanium pipeline
    ok = fetch_archive_image(search_query, fname)
    if not ok: ok = fetch_cloudflare_image(ai_prompt, fname)
    if not ok: ok = fetch_pexels_image(ai_prompt, fname)
    if not ok: ok = fetch_placeholder_image(fname, search_query[:30])

    if not verify_and_convert_image(fname):
        fetch_placeholder_image(fname)
    apply_diegetic_matting(fname)

    try:
        base = ImageClip(fname).resize(height=VIDEO_HEIGHT)
        if base.w < VIDEO_WIDTH:
            base = base.resize(width=VIDEO_WIDTH)
        base = base.crop(
            x_center=base.w / 2, y_center=base.h / 2,
            width=VIDEO_WIDTH, height=VIDEO_HEIGHT,
        )

        # Depth parallax attempt
        base = base.set_duration(duration)
        cropped_path = f"temp_cropped_{index}.jpg"
        base.save_frame(cropped_path, t=0)
        depth_path = generate_depth_map(cropped_path)

        if depth_path and os.path.exists(depth_path):
            img_arr   = cv2.cvtColor(cv2.imread(cropped_path), cv2.COLOR_BGR2RGB)
            depth_arr = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            cam_dir   = "left" if index % 2 == 0 else "right"
            clip = VideoClip(
                make_frame=lambda t: apply_parallax_frame(
                    t, duration, img_arr, depth_arr, cam_dir
                ),
                duration=duration,
            )
        else:
            # Ken Burns eased zoom fallback
            def _zoom(t):
                e = _ease_inout(t / duration)
                return (1 + 0.06 * e) if index % 2 == 0 else (1.06 - 0.06 * e)
            clip = base.resize(_zoom).crop(
                x_center=VIDEO_WIDTH / 2, y_center=VIDEO_HEIGHT / 2,
                width=VIDEO_WIDTH, height=VIDEO_HEIGHT,
            )

        # Cross-dissolve
        clip = clip.fx(fadein, CROSSFADE_DUR).fx(fadeout, CROSSFADE_DUR)
        return clip

    except Exception as e:
        print(f"  ⚠️  Clip {index} error: {e}")
        return ColorClip(size=(VIDEO_WIDTH, VIDEO_HEIGHT),
                         color=COLOUR_BG, duration=duration)


# ═══════════════════════════════════════════════════════════════════════════
#  ATMOSPHERICS & MUSIC
# ═══════════════════════════════════════════════════════════════════════════

def fetch_atmospheric_b_roll(duration: float, filename: str = "temp_atmosphere.mp4") -> bool:
    """Fetches a Pexels portrait video for subtle atmospheric overlay."""
    print("  🌫️  Fetching atmospheric B-roll...")
    if not PEXELS_KEY: return False
    queries = [
        "dust particles dark background", "film grain overlay",
        "laboratory glass reflection", "smoke dark background",
        "rain drops window dark", "fog corridor",
    ]
    try:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": PEXELS_KEY},
            params={"query": random.choice(queries),
                    "per_page": 3, "orientation": "portrait"},
            timeout=30,
        )
        if r.status_code == 200:
            videos = r.json().get("videos", [])
            if videos:
                video = random.choice(videos)
                files = ([f for f in video.get("video_files", [])
                          if f.get("quality") == "hd"]
                         or video.get("video_files", []))
                if files:
                    with open(filename, "wb") as f:
                        f.write(requests.get(files[0]["link"], timeout=60).content)
                    return True
    except Exception: pass
    return False


def fetch_pixabay_audio(
    music_keywords: str,
    filename: str = "temp_bg_music.mp3",
) -> bool:
    """Fetches a background music track from Pixabay using pre-generated keywords."""
    print(f"  🎵 Fetching music: '{music_keywords}'")
    if not PIXABAY_KEY: return False
    try:
        r = requests.get(
            "https://pixabay.com/api/audio/",
            params={"key": PIXABAY_KEY, "q": music_keywords, "per_page": 5},
            timeout=15,
        )
        if r.status_code == 200:
            hits = r.json().get("hits", [])
            if hits:
                track = random.choice(hits[:3])
                if track.get("audio"):
                    with open(filename, "wb") as f:
                        f.write(requests.get(track["audio"], timeout=45).content)
                    return True
    except Exception: pass
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  SFX & STINGERS
# ═══════════════════════════════════════════════════════════════════════════

def add_sfx(audio_clip, text: str):
    """Layers a keyword-triggered ambient SFX at 18% volume."""
    for kw, sfx_file in SFX_KEYWORD_MAP.items():
        if kw in text.lower():
            path = os.path.join("sfx", sfx_file)
            if os.path.exists(path):
                try:
                    sfx = AudioFileClip(path).volumex(0.18)
                    return CompositeAudioClip(
                        [audio_clip,
                         sfx.subclip(0, min(sfx.duration, audio_clip.duration))]
                    )
                except Exception: pass
    return audio_clip


def add_stinger_sfx(audio_clip, text: str):
    """Layers a cinematic impact stinger at 0.35s offset on narrative beat words."""
    for kw, sfx_file in STINGER_MAP.items():
        if kw in text.lower():
            path = os.path.join("sfx", sfx_file)
            if os.path.exists(path):
                try:
                    stinger = (AudioFileClip(path)
                               .volumex(0.38)
                               .set_start(min(0.35, max(0.0, audio_clip.duration - 0.8))))
                    return CompositeAudioClip([audio_clip, stinger])
                except Exception: pass
    return audio_clip


# ═══════════════════════════════════════════════════════════════════════════
#  NETFLIX KARAOKE SUBTITLE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

def get_subtitle_font(size: int = 60) -> PIL.ImageFont.ImageFont:
    """Loads a bold system font. Tries multiple paths, falls back to default."""
    for path in FONT_PATHS:
        if os.path.exists(path):
            try:
                return PIL.ImageFont.truetype(path, size)
            except Exception:
                continue
    return PIL.ImageFont.load_default()


def make_karaoke_frame(
    words:      list[dict],
    active_idx: int,
    vid_width:  int,
) -> PIL.Image.Image:
    """
    Renders one transparent RGBA frame for the karaoke subtitle system.
    Active word → signal yellow, size 66, full opacity.
    Inactive words → white, size 54, 62% opacity.
    8-directional black stroke on all words for readability over any image.
    """
    frame_h  = 145
    img      = PIL.Image.new("RGBA", (vid_width, frame_h), (0, 0, 0, 0))
    draw     = PIL.ImageDraw.Draw(img)
    fn_norm  = get_subtitle_font(54)
    fn_actv  = get_subtitle_font(66)

    # Pre-measure all word widths
    widths, total_w = [], 0
    for i, w in enumerate(words):
        fn    = fn_actv if i == active_idx else fn_norm
        bbox  = draw.textbbox((0, 0), w["word"] + " ", font=fn)
        ww    = bbox[2] - bbox[0]
        widths.append(ww)
        total_w += ww

    # Clamp to screen width
    max_w = vid_width - 44
    if total_w > max_w:
        scale  = max_w / total_w
        widths = [int(w * scale) for w in widths]
        total_w = sum(widths)

    x = (vid_width - total_w) // 2

    for i, w in enumerate(words):
        is_active = (i == active_idx)
        fn        = fn_actv if is_active else fn_norm
        fill      = (*COLOUR_ACCENT, 255) if is_active else (*COLOUR_WHITE, 158)
        y         = (frame_h - (66 if is_active else 54)) // 2

        # Stroke: 8-directional black outline
        for dx, dy in [(-2,2),(2,2),(-2,-2),(2,-2),(0,3),(0,-3),(3,0),(-3,0)]:
            draw.text((x+dx, y+dy), w["word"], font=fn, fill=(0, 0, 0, 215))

        draw.text((x, y), w["word"], font=fn, fill=fill)
        x += widths[i]

    return img


def _pil_rgba_to_clip(pil_img: PIL.Image.Image, duration: float):
    """Converts a PIL RGBA image to a MoviePy clip with proper alpha mask."""
    rgb_arr   = np.array(pil_img.convert("RGB"))
    alpha_arr = np.array(pil_img.split()[3]).astype(float) / 255.0
    clip = ImageClip(rgb_arr, duration=duration)
    mask = ImageClip(alpha_arr, ismask=True, duration=duration)
    return clip.set_mask(mask)


def add_dynamic_subtitles(video_clip, audio_path: str):
    """
    Adds Netflix-style karaoke subtitles using faster-whisper word timestamps.
    Groups into 5-word phrases, highlights active word in signal yellow.
    Falls back to plain word clips on any PIL failure.
    """
    print("  📝 Generating karaoke subtitles...")
    PHRASE_SIZE = 5
    try:
        model       = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_path, word_timestamps=True)

        all_words = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    clean = w.word.strip().upper()
                    if clean:
                        all_words.append({"word": clean,
                                          "start": w.start, "end": w.end})

        if not all_words:
            return video_clip

        phrases   = [all_words[i:i+PHRASE_SIZE]
                     for i in range(0, len(all_words), PHRASE_SIZE)]
        sub_clips = []
        sub_y     = int(video_clip.h * 0.72)

        for phrase in phrases:
            for active_idx, word_info in enumerate(phrase):
                dur = max(word_info["end"] - word_info["start"], 0.06)
                try:
                    frame = make_karaoke_frame(phrase, active_idx, VIDEO_WIDTH)
                    wc = (_pil_rgba_to_clip(frame, dur)
                          .set_start(word_info["start"])
                          .set_position(("center", sub_y)))
                    sub_clips.append(wc)
                except Exception:
                    # Per-word fallback
                    try:
                        tc = (TextClip(word_info["word"], fontsize=68,
                                       color="yellow", stroke_color="black",
                                       stroke_width=2, font="Impact",
                                       method="caption",
                                       size=(int(video_clip.w * 0.9), None))
                              .set_start(word_info["start"])
                              .set_end(word_info["end"])
                              .set_position(("center", video_clip.h * 0.72)))
                        sub_clips.append(tc)
                    except Exception:
                        pass

        return (CompositeVideoClip([video_clip] + sub_clips)
                if sub_clips else video_clip)

    except Exception as e:
        print(f"  ⚠️  Subtitle error: {e}")
        return video_clip


# ═══════════════════════════════════════════════════════════════════════════
#  THUMBNAIL GENERATOR — Statistic-first formula
# ═══════════════════════════════════════════════════════════════════════════

def generate_thumbnail(
    source_image_path: str,
    key_stat:          dict,
    output_path:       str = "thumbnail.jpg",
) -> str | None:
    """
    Composes a high-CTR thumbnail using the statistic-first formula:
      Large stat number in signal yellow (hero element)
      Context line in white below
      Red badge + channel handle at bottom
      Dark-graded base image
    """
    print("  🖼️  Composing thumbnail...")
    stat    = key_stat.get("stat", "")
    context = key_stat.get("context", "")
    if not stat:
        stat, context = "?", "of participants crossed the line"

    try:
        with PIL.Image.open(source_image_path) as img:
            img = img.convert("RGB").resize((1280, 720), PIL.Image.Resampling.LANCZOS)
            arr = np.array(img, dtype=np.float32)
            # Dark cinematic grade + cold blue tint
            arr = np.clip(arr * 0.48 + 8, 0, 255).astype(np.uint8)
            arr[:, :, 2] = np.clip(arr[:, :, 2].astype(int) + 22, 0, 255)  # blue push
            base = PIL.Image.fromarray(arr)

        draw     = PIL.ImageDraw.Draw(base)
        fn_stat  = get_subtitle_font(210)
        fn_ctx   = get_subtitle_font(52)
        fn_badge = get_subtitle_font(38)
        fn_hdl   = get_subtitle_font(34)

        # Bottom banner
        draw.rectangle([(0, 590), (1280, 720)], fill=(*COLOUR_BG, 255))
        draw.text((44, 610), CHANNEL_HANDLE, font=fn_hdl, fill=COLOUR_HANDLE_GREY)

        # Stat number — signal yellow with black stroke
        stat_display = stat[:6]  # prevent overflow
        for dx, dy in [(-4,4),(4,4),(-4,-4),(4,-4),(0,5),(0,-5),(5,0),(-5,0)]:
            draw.text((40+dx, 50+dy), stat_display, font=fn_stat, fill=(0, 0, 0))
        draw.text((40, 50), stat_display, font=fn_stat, fill=COLOUR_ACCENT)

        # Context line — white, below stat
        ctx_upper = context.upper()[:55]
        for dx, dy in [(-2,2),(2,2),(-2,-2),(2,-2)]:
            draw.text((44+dx, 350+dy), ctx_upper, font=fn_ctx, fill=(0, 0, 0))
        draw.text((44, 350), ctx_upper, font=fn_ctx, fill=COLOUR_WHITE)

        # Red badge
        draw.rectangle([(44, 405), (240, 450)], fill=COLOUR_RED_BADGE)
        draw.text((58, 410), "DOCUMENTED", font=fn_badge, fill=COLOUR_WHITE)

        base.save(output_path, "JPEG", quality=96)
        print(f"  ✅ Thumbnail saved: {output_path}")
        return output_path
    except Exception as e:
        print(f"  ⚠️  Thumbnail failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  END SCREEN — 2.5s black card drives replay
# ═══════════════════════════════════════════════════════════════════════════

def add_end_screen(video_clip, last_clean_line: str = ""):
    """
    Appends a 2.5-second black card with a haunting closing question.
    The question is extracted from the final script line if possible.
    """
    print("  🎬 Adding end screen...")
    END_DUR = 2.5
    try:
        card = ColorClip(size=(VIDEO_WIDTH, VIDEO_HEIGHT),
                         color=COLOUR_BG, duration=END_DUR)

        closing = last_clean_line if "?" in last_clean_line else "What would you have done?"

        q = (TextClip(closing, fontsize=50, color="white", font="DejaVu-Sans-Bold",
              stroke_color="#BB1414", stroke_width=2,
              method="caption", size=(int(VIDEO_WIDTH * 0.84), None))
     .set_position("center").set_duration(END_DUR).fx(fadein, 0.75))

        h = (TextClip(CHANNEL_HANDLE, fontsize=28,
                      color="#6E7278", font="DejaVu-Sans-Bold")
             .set_position(("center", int(VIDEO_HEIGHT * 0.76)))
             .set_duration(END_DUR).fx(fadein, 1.3))
        
        # In the watermark block (Step 20):
        wm = (TextClip(CHANNEL_HANDLE, fontsize=26, color="white",
                       font="DejaVu-Sans-Bold", stroke_color="black", stroke_width=1)

        end = CompositeVideoClip([card, q, h])
        return concatenate_videoclips([video_clip, end], method="compose")
    except Exception as e:
        print(f"  ⚠️  End screen error: {e}")
        return video_clip


# ═══════════════════════════════════════════════════════════════════════════
#  YOUTUBE UPLOAD (with custom thumbnail)
# ═══════════════════════════════════════════════════════════════════════════

def upload_to_youtube(
    file_path:       str,
    yt_metadata:     dict,
    thumbnail_path:  str | None = None,
) -> tuple[bool, str | None]:
    """Uploads video + thumbnail to YouTube via the Data API v3."""
    if not file_path or not YOUTUBE_TOKEN_VAL:
        return False, None
    print("  🚀 Uploading to YouTube...")
    try:
        creds   = Credentials.from_authorized_user_info(json.loads(YOUTUBE_TOKEN_VAL))
        youtube = build("youtube", "v3", credentials=creds)

        body = {
            "snippet": {
                "title":       yt_metadata["title"],
                "description": yt_metadata["description"],
                "tags":        yt_metadata["tags"],
                "categoryId":  "27",   # 27 = Education (correct for this channel)
            },
            "status": {
                "privacyStatus":          "public",
                "selfDeclaredMadeForKids": False,
            },
        }

        insert_rsp = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=MediaFileUpload(file_path, chunksize=-1, resumable=True),
        ).execute()

        video_id = insert_rsp.get("id")
        print(f"  ✅ YouTube upload success! ID: {video_id}")

        if thumbnail_path and video_id and os.path.exists(thumbnail_path):
            try:
                youtube.thumbnails().set(
                    videoId=video_id,
                    media_body=MediaFileUpload(thumbnail_path),
                ).execute()
                print("  ✅ Thumbnail uploaded.")
            except Exception as e:
                print(f"  ⚠️  Thumbnail upload: {e}")

        return True, video_id
    except Exception as e:
        print(f"  ❌ YouTube upload failed: {e}")
        return False, None


# ═══════════════════════════════════════════════════════════════════════════
#  MASTER ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════

def main_pipeline() -> bool:
    """
    Executes the full pipeline from research to upload.
    Returns True on successful upload, False on any critical failure.

    Step sequence (each step logged, any critical failure returns False):
      0. Boot: anti-ban sleep, VoiceEngine init, SOTA model selection
      1. Format selection
      2-6. Phase 0: Research (4 LLM tasks + 2 web scrapes)
      7-11. Phase 1: Writing (5 LLM tasks, chained)
      12-13. Phase 1b: Voice direction (SSML per line + speaker assignment)
      14. Phase 1c: Key statistic extraction
      15. Script assembly
      16. Phase 2: Audio build (VoiceEngine + SFX + stingers)
      17. Phase 3: Visual direction (N prompt tasks, one per slot)
      18. Phase 3: Clip generation (Titanium pipeline × N)
      19. Video assembly (concatenate + atmospheric overlay + music)
      20. Karaoke subtitles + watermark + end screen
      21. Render
      22. Thumbnail generation
      23. YouTube upload
      24. Save topic to memory
      25. Phase 5: Metadata (5 tasks) + social upload
    """
    print("\n" + "═"*60)
    print(f"  {CHANNEL_NAME} — Pipeline Starting")
    print("═"*60)

    # ── Step 0: Boot ─────────────────────────────────────────────────────
    anti_ban_sleep()

    try:
        voice_engine = VoiceEngine()
    except Exception as e:
        print(f"❌ VoiceEngine init failed: {e}")
        return False

    sota_models = get_top_free_openrouter_models()
    past_topics = get_past_topics()

    # ── Step 1: Format selection ─────────────────────────────────────────
    fmt = random.choices(VIDEO_FORMATS, weights=[20, 60, 20], k=1)[0]
    print(f"\n📐 Format: {fmt['description']}")

    # ── Steps 2–6: Phase 0 — Research ────────────────────────────────────
    print("\n── Phase 0: Research ─────────────────────────────────────────")
    domain = random.choice(CONTENT_POOL)
    print(f"  📂 Domain: {domain[:60]}")

    study_name = task_propose_study(domain, past_topics, sota_models)
    if not study_name:
        print("❌ Could not propose a study. Aborting.")
        return False

    print(f"  🔬 Study: {study_name}")

    wiki_text  = scrape_wikipedia(study_name)
    news_text  = scrape_google_news_rss(study_name)
    research   = (
        f"STUDY: {study_name}\n\n"
        + (f"=== WIKIPEDIA ===\n{wiki_text}\n\n" if wiki_text else "")
        + (f"=== RECENT NEWS ===\n{news_text}\n\n" if news_text else "")
    )

    key_facts = task_extract_key_facts(research, sota_models)
    if not key_facts:
        print("❌ Could not extract key facts. Aborting.")
        return False

    contradiction = task_identify_contradiction(key_facts, study_name, sota_models)
    if not contradiction:
        print("❌ Could not identify contradiction. Aborting.")
        return False

    era = task_detect_era(research, sota_models) or "unknown"

    # ── Steps 7–11: Phase 1 — Writing (chained) ──────────────────────────
    print("\n── Phase 1: Writing ──────────────────────────────────────────")

    hook = task_write_hook(key_facts, contradiction, study_name, sota_models)
    if not hook:
        print("❌ Hook generation failed. Aborting.")
        return False

    escalation = task_write_escalation(hook, key_facts, study_name, sota_models)
    if not escalation:
        print("⚠️  Escalation failed — using minimal fallback.")
        escalation = []

    built_so_far = "\n".join([hook] + escalation)

    contra_lines = task_write_contradiction_lines(
        hook, escalation, contradiction, study_name, sota_models
    )
    if not contra_lines:
        print("⚠️  Contradiction lines failed — using minimal fallback.")
        contra_lines = []

    built_so_far += "\n" + "\n".join(contra_lines)

    imp_detail = task_write_impossible_detail(
        key_facts, study_name, built_so_far, sota_models
    )
    if not imp_detail:
        print("⚠️  Impossible detail failed — using minimal fallback.")
        imp_detail = []

    built_so_far += "\n" + "\n".join(imp_detail)

    loop_end = task_write_loop_ending(hook, built_so_far, study_name, sota_models)
    if not loop_end:
        print("⚠️  Loop ending failed — using hook as fallback.")
        loop_end = hook

    # ── Steps 12–15: Phase 1b + 1c — Voice direction + stat ──────────────
    print("\n── Phase 1b: Voice Direction + Script Assembly ───────────────")

    script = assemble_script(
        study_name, era, hook, escalation, contra_lines,
        imp_detail, loop_end, sota_models
    )
    if not script or not script.get("lines"):
        print("❌ Script assembly failed. Aborting.")
        return False

    # Enforce format line limit
    script["lines"] = script["lines"][:fmt["max_lines"]]

    full_script_text = " ".join(l["clean_text"] for l in script["lines"])
    print(f"\n  📄 Script: {len(script['lines'])} lines, "
          f"~{len(full_script_text.split())} words")

    print("\n── Phase 1c: Key Statistic Extraction ───────────────────────")
    key_stat = task_extract_key_statistic(full_script_text, sota_models)
    print(f"  📊 Stat: {key_stat.get('stat')} — {key_stat.get('context')}")

    # ── Step 16: Phase 2 — Audio Build ───────────────────────────────────
    print("\n── Phase 2: Audio Assembly ───────────────────────────────────")
    audio_clips = []

    for i, line in enumerate(script["lines"]):
        clean  = line["clean_text"]
        acting = line["acting_text"]
        style  = line["style_instruction"]
        voice  = VOICE_MAP.get(line["speaker"], "Charon")

        print(f"  🎙️  Line {i+1}/{len(script['lines'])} [{voice}]")
        wav = voice_engine.generate_acting_line(acting, clean, style, i, voice)
        if wav:
            clip = AudioFileClip(wav)
            clip = add_sfx(clip, clean)
            clip = add_stinger_sfx(clip, clean)
            audio_clips.append(clip)

    if not audio_clips:
        print("❌ No audio clips generated. Aborting.")
        return False

    master_voice = concatenate_audioclips(audio_clips)
    print(f"  ✅ Master audio: {master_voice.duration:.1f}s")

    # ── Step 17: Visual Direction ─────────────────────────────────────────
    print("\n── Phase 3: Visual Direction ─────────────────────────────────")
    required_images = min(max(2, int(master_voice.duration / IMAGE_TRANSITION_T)), 12)
    visual_prompts  = generate_all_visual_prompts(
        full_script_text, required_images, era, sota_models
    )
    dur_per_image   = master_voice.duration / len(visual_prompts)
    first_img_path  = "temp_img_0.jpg"

    # ── Step 18: Clip generation ──────────────────────────────────────────
    print("\n── Phase 3: Clip Generation ──────────────────────────────────")
    visual_clips = []
    for i, vp in enumerate(visual_prompts):
        print(f"  🖼️  Clip {i+1}/{len(visual_prompts)}")
        clip = get_image_clip(
            vp.get("search_query", "psychology research"),
            vp.get("ai_prompt", "dark institutional corridor, documentary"),
            dur_per_image, i,
        )
        visual_clips.append(clip)

    # ── Step 19: Video assembly ───────────────────────────────────────────
    print("\n── Phase 3: Video Assembly ───────────────────────────────────")
    try:
        final_video = (
            concatenate_videoclips(
                visual_clips, method="compose", padding=-CROSSFADE_DUR
            )
            .set_duration(master_voice.duration)
            .fx(colorx, 0.83)
        )

        # Atmospheric overlay (22% opacity)
        if fetch_atmospheric_b_roll(master_voice.duration):
            try:
                from moviepy.video.fx.all import loop as vfx_loop
                atm = VideoFileClip("temp_atmosphere.mp4").without_audio()
                atm = vfx_loop(atm, duration=master_voice.duration).resize(height=VIDEO_HEIGHT)
                if atm.w < VIDEO_WIDTH:
                    atm = atm.resize(width=VIDEO_WIDTH)
                atm = (atm.crop(x_center=atm.w/2, y_center=atm.h/2,
                                width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
                       .set_opacity(0.22))
                final_video = CompositeVideoClip([final_video, atm])
            except Exception as e:
                print(f"  ⚠️  Atmospheric overlay: {e}")

        final_video = final_video.set_audio(master_voice)

    except Exception as e:
        print(f"❌ Video assembly failed: {e}")
        return False

    # ── Step 20: Subtitles + watermark + music ────────────────────────────
    print("\n── Phase 1 Upgrades: Subtitles / Watermark / Music ──────────")
    temp_voice_wav = "temp_master_voice.wav"
    master_voice.write_audiofile(temp_voice_wav, fps=24000, logger=None)
    final_video = add_dynamic_subtitles(final_video, temp_voice_wav)

    try:
        wm = (TextClip(CHANNEL_HANDLE, fontsize=26, color="white",
                       font="Impact", stroke_color="black", stroke_width=1)
              .set_opacity(0.30)
              .set_position(("center", 135))
              .set_duration(final_video.duration))
        final_video = CompositeVideoClip([final_video, wm])
    except Exception: pass

    # Music keywords (single task)
    music_kw = task_write_music_keywords(full_script_text, sota_models)
    if fetch_pixabay_audio(music_kw):
        try:
            bg = audio_loop(
                AudioFileClip("temp_bg_music.mp3").volumex(0.052),
                duration=final_video.duration,
            )
            final_video = final_video.set_audio(
                CompositeAudioClip([final_video.audio, bg])
            )
        except Exception: pass

    # End screen
    last_line = script["lines"][-1]["clean_text"] if script["lines"] else ""
    final_video = add_end_screen(final_video, last_line)

    # ── Step 21: Render ───────────────────────────────────────────────────
    print("\n── Rendering Final Video ─────────────────────────────────────")
    output_file = "final_video.mp4"
    try:
        final_video.write_videofile(
            output_file, codec="libx264", audio_codec="aac",
            fps=24, preset="fast", threads=2, logger=None,
        )
        print(f"  ✅ Rendered: {output_file}")
    except Exception as e:
        print(f"❌ Render failed: {e}")
        return False

    # ── Step 22: Thumbnail ────────────────────────────────────────────────
    print("\n── Thumbnail Generation ──────────────────────────────────────")
    thumbnail_path = None
    if os.path.exists(first_img_path):
        thumbnail_path = generate_thumbnail(first_img_path, key_stat)

    # ── Step 23: YouTube upload ───────────────────────────────────────────
    print("\n── Phase 5: Metadata Tasks ───────────────────────────────────")
    print("  ⏳ Rate limit recovery pause (30s)...")
    time.sleep(30)   # ADD THIS LINE
    title       = task_write_youtube_title(full_script_text, key_stat, study_name, sota_models)
    description = task_write_youtube_description(title, study_name, key_stat, sota_models)
    tags        = task_write_seo_tags(title, study_name, era, sota_models)
    ig_caption  = task_write_ig_caption(title, description, key_stat, sota_models)
    fb_caption  = task_write_fb_caption(title, description, sota_models)

    yt_metadata = {
        "title":       title,
        "description": description + "\n\nWhat would you have done? 👇",
        "tags":        tags,
    }

    print(f"\n  📋 Title: {title}")
    print(f"  🏷️  Tags: {', '.join(tags[:5])}...")

    print("\n── Upload ────────────────────────────────────────────────────")
    success, video_id = upload_to_youtube(output_file, yt_metadata, thumbnail_path)

    if not success:
        print("❌ YouTube upload failed.")
        return False

    # ── Step 24: Save topic to memory ────────────────────────────────────
    save_new_topic(study_name)

    # ── Step 25: Social uploads ───────────────────────────────────────────
    print("\n── Social Broadcasting ───────────────────────────────────────")
    meta_upload.upload_to_facebook(output_file, fb_caption)
    temp_url = meta_upload.get_temp_public_url(output_file)
    if temp_url:
        meta_upload.upload_to_instagram(temp_url, ig_caption)

    # ── Cleanup ───────────────────────────────────────────────────────────
    for pattern in ("temp_*.wav", "temp_*.jpg", "temp_*.mp4", "temp_*.mp3"):
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except Exception:
                pass

    print("\n" + "═"*60)
    print(f"  ✅ Pipeline complete — {CHANNEL_NAME}")
    print(f"  🎬 YouTube: https://youtube.com/watch?v={video_id}")
    print("═"*60 + "\n")
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    success = main_pipeline()
    sys.exit(0 if success else 1)
