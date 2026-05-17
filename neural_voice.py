"""
THE HUMAN EXPERIMENT — Voice Engine
=====================================
Handles all TTS generation via Gemini's prebuilt voices.

4-Voice System:
  Charon   → narrator   : World-weary research analyst. Precise, controlled, slightly detached.
  Fenrir   → document   : Cold, flat, institutional. Reads official findings and methodology.
  Kore     → witness    : Personal, sometimes shaken. Participant quotes and observer reactions.
  Puck     → researcher : Initially confident academic voice, becomes unsettled by their own data.

Each voice passes through a 5-stage mastering chain tuned for documentary narration:
  1. High-pass filter  (80 Hz)   — removes mic rumble
  2. Low-pass filter   (12 kHz)  — removes harsh sibilance
  3. Dynamic compression         — podcast-grade loudness consistency
  4. Normalize                   — consistent peak level across all lines
  5. Dynamic trailing silence    — duration driven by SILENCE_MAP per emotional style

Single-task design:
  generate_acting_line() does exactly one thing: render one line to a mastered .wav file.
  It never batches. Never reuses state. Retry logic is isolated per line.
"""

import os
import wave
import time

from google import genai
from google.genai import types
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range, normalize


# ═══════════════════════════════════════════════════════════════════════════
#  VOICE MAP  (exported — imported by main.py)
# ═══════════════════════════════════════════════════════════════════════════
VOICE_MAP: dict[str, str] = {
    "narrator":   "Charon",   # Primary: measured, precise, authoritative
    "document":   "Fenrir",   # Clinical: cold, flat, reads data and reports
    "witness":    "Kore",     # Personal: shaken, human, quoting participants
    "researcher": "Puck",     # Academic: confident at start, unsettled by end
}


# ═══════════════════════════════════════════════════════════════════════════
#  SILENCE MAP
#  Controls the trailing silence appended after each line.
#  Silence duration is a narrative pacing decision — not a flat constant.
#  Longer pauses after heavy revelations. Short pauses after rapid facts.
# ═══════════════════════════════════════════════════════════════════════════
SILENCE_MAP: dict[str, int] = {
    # Heavy / slow lines — let them breathe
    "hushed":         950,
    "whisper":        900,
    "haunting":       1100,
    "dread":          1050,
    "deliberate":     980,
    "let the":        980,    # matches "let the final word hang"

    # Pivot / weighted lines
    "pivot":          700,
    "weighted":       750,
    "slow down":      800,
    "precise":        600,

    # Building / escalation lines
    "building":       300,
    "heavier":        350,
    "faster":         180,
    "rapid":          150,

    # Clinical / document lines
    "cold":           200,
    "flat":           160,
    "clinical":       220,
    "institutional":  180,
    "measured":       280,

    # Default fallback
    "default":        320,
}


def _get_silence_ms(style_instruction: str) -> int:
    """
    Matches the style instruction string against SILENCE_MAP keywords.
    Returns silence duration in milliseconds.
    First match wins — keys are ordered from most specific to least.
    """
    style_lower = style_instruction.lower()
    for key, duration in SILENCE_MAP.items():
        if key in style_lower:
            return duration
    return SILENCE_MAP["default"]


# ═══════════════════════════════════════════════════════════════════════════
#  VOICE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class VoiceEngine:
    """
    Gemini TTS wrapper with professional mastering chain.

    Usage:
        engine = VoiceEngine()
        wav_path = engine.generate_acting_line(
            acting_text       = "<break time='1s'/>They couldn't explain it.",
            clean_text        = "They couldn't explain it.",
            style_instruction = "Hushed, deliberate. Let the final word hang.",
            index             = 0,
            voice_name        = "Charon"
        )
    """

    # TTS model preference order — Flash is primary, Pro is fallback
    TTS_MODELS = [
        "gemini-2.5-flash-preview-tts",
        "gemini-2.5-pro",
    ]

    # Native output spec for all Gemini TTS models
    SAMPLE_RATE  = 24000
    CHANNELS     = 1      # Mono
    SAMPLE_WIDTH = 2      # 16-bit PCM

    def __init__(self) -> None:
        print("🎚️  Voice Engine initialising...")
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Add it to your GitHub Actions secrets."
            )
        self.client = genai.Client(api_key=self.api_key)
        print("✅ Voice Engine ready.")

    # ─────────────────────────────────────────────────────────────────────
    #  MASTERING CHAIN
    # ─────────────────────────────────────────────────────────────────────

    def _master(self, sound: AudioSegment, style_instruction: str) -> AudioSegment:
        """
        5-stage mastering chain tuned for documentary narration.

        Stage 1 — High-Pass (80 Hz):
            Removes low-frequency room rumble and handling noise below 80 Hz.
            Essential for clean, broadcast-quality audio.

        Stage 2 — Low-Pass (12 kHz):
            Removes harsh digital sibilance above 12 kHz.
            Keeps the voice warm and documentary-appropriate without muddiness.

        Stage 3 — Compression:
            threshold = -14 dBFS  → tight gate, catches peaks cleanly
            ratio     = 4.5 : 1   → firm but not over-squashed
            attack    = 4 ms      → fast enough to catch transients
            release   = 40 ms     → snappy, lets natural voice dynamics breathe
            Result: consistent loudness across all four voice types,
                    regardless of how aggressively Gemini renders each line.

        Stage 4 — Normalize to -0.2 dBFS:
            Maximises perceived loudness consistently.
            Headroom = 0.2 prevents any inter-sample clipping.

        Stage 5 — Dynamic trailing silence:
            Duration is read from SILENCE_MAP using the style instruction.
            A "whisper" line hangs for 950 ms.
            A "measured" narration line gets 280 ms.
            This is the most important pacing control in the entire pipeline.
        """
        # Stage 1
        sound = sound.high_pass_filter(80)

        # Stage 2
        sound = sound.low_pass_filter(12000)

        # Stage 3
        sound = compress_dynamic_range(
            sound,
            threshold = -14.0,
            ratio     = 4.5,
            attack    = 4.0,
            release   = 40.0,
        )

        # Stage 4
        sound = normalize(sound, headroom=0.2)

        # Stage 5
        silence_ms = _get_silence_ms(style_instruction)
        sound = sound + AudioSegment.silent(duration=silence_ms)

        return sound

    # ─────────────────────────────────────────────────────────────────────
    #  DIRECTOR SYSTEM PROMPT  (shared across all voice roles)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_director_prompt(
        acting_text:       str,
        style_instruction: str,
        voice_name:        str,
    ) -> str:
        """
        Builds the TTS director prompt for one line.

        Design decisions:
        - The model is given a ROLE, not just an instruction.
          Roles produce more consistent vocal performances than style adjectives alone.
        - SSML execution is described as STAGE DIRECTIONS, not as markup.
          This framing prevents the model from reading tag names aloud.
        - The studio environment description is included because Gemini TTS
          responds to spatial context — "late-night radio studio" consistently
          produces a slightly reverberant, intimate vocal quality vs. a dead booth.
        - The specific voice's character notes ensure the four voices
          stay differentiated even when Gemini's default rendering would flatten them.
        """
        voice_character = {
            "Charon": (
                "You are the narrator of 'The Human Experiment' — a research analyst "
                "who has spent years studying what humans do when observed. "
                "Your voice is controlled, precise, and carries the quiet weight of someone "
                "who has read the data and is no longer surprised by it — but still thinks "
                "the public should know."
            ),
            "Fenrir": (
                "You are reading from an official research document — methodology section, "
                "published results, or institutional statement. "
                "Your voice is completely flat and institutional. "
                "You have no opinion about what you are reading. "
                "You are a recording, not a person."
            ),
            "Kore": (
                "You are quoting a participant, observer, or secondary researcher. "
                "This is a personal account. You were there. "
                "Your voice carries the specific texture of someone describing something "
                "they still haven't fully processed. Not dramatic — quietly unsettled."
            ),
            "Puck": (
                "You are the lead researcher, speaking before the results were known. "
                "Early lines: confident academic, setting up your hypothesis. "
                "Later lines: that same voice, but now you have seen the data. "
                "The confidence is still there — but now it's working very hard to stay."
            ),
        }.get(voice_name, "You are a precise, controlled documentary narrator.")

        return f"""{voice_character}

YOUR VOCAL DELIVERY FOR THIS SPECIFIC LINE:
"{style_instruction}"

CRITICAL STAGE DIRECTION RULES:
The script below contains SSML tags. These are silent stage directions — never speak them.
Execute them exactly:

  <break time="Xs"/>               → Stop completely for X seconds. Silence. Not a hesitation.
  <emphasis level="strong">WORD</emphasis>  → Give that word full weight. Not louder — heavier.
  <prosody rate="slow" pitch="-12%">PHRASE</prosody>  → Slow your delivery. Drop your pitch. Maximum gravity.
  <prosody rate="fast">PHRASE</prosody>     → Accelerate. Rapid-fire factual delivery.

RECORDING ENVIRONMENT:
You are in a quiet, slightly reverberant late-night documentary studio.
There is natural room presence in your voice — not a dead anechoic recording.
This is a real studio. You are a real performer. The audience is real.

SCRIPT LINE:
{acting_text}"""

    # ─────────────────────────────────────────────────────────────────────
    #  CORE METHOD: generate_acting_line
    # ─────────────────────────────────────────────────────────────────────

    def generate_acting_line(
        self,
        acting_text:        str,
        clean_text:         str,
        style_instruction:  str,
        index:              int,
        voice_name:         str = "Charon",
    ) -> str | None:
        """
        Renders one script line to a mastered .wav file.

        Single-task design:
            This method does exactly one thing.
            It never batches multiple lines.
            It never reuses audio state from a previous call.
            Retry logic is fully self-contained within this call.

        Parameters
        ----------
        acting_text       : SSML-tagged script text for the TTS model to perform.
        clean_text        : Plain text (used for logging only — not sent to TTS).
        style_instruction : Pacing and emotional direction for this specific line.
        index             : Sequential index — used for unique temp filename only.
        voice_name        : Gemini prebuilt voice name from VOICE_MAP.

        Returns
        -------
        Path to the mastered .wav file, or None if all attempts failed.

        Retry strategy:
            - Tries each TTS model up to 3 times.
            - On 429 (rate limit) or 503 (server): exponential backoff (35s, 47s, 59s).
            - On any other exception: logs and moves to the next model.
            - Returns None only after all models × all retries are exhausted.
        """
        output_path = f"temp_voice_{index}.wav"
        temp_raw    = f"temp_raw_{index}.wav"

        print(f"    🎙️  [{voice_name}] Line {index}: {clean_text[:50]}...")

        prompt = self._build_director_prompt(acting_text, style_instruction, voice_name)

        tts_config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            ),
        )

        for model_name in self.TTS_MODELS:
            for attempt in range(3):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=tts_config,
                    )

                    # Extract raw PCM bytes from response
                    audio_bytes = None
                    if (response.candidates
                            and response.candidates[0].content.parts):
                        for part in response.candidates[0].content.parts:
                            if part.inline_data:
                                audio_bytes = part.inline_data.data
                                break

                    if not audio_bytes:
                        print(f"      ⚠️  No audio data (model={model_name}, "
                              f"attempt={attempt + 1}/3)")
                        continue

                    # Write raw PCM → temp wav
                    with wave.open(temp_raw, "wb") as wf:
                        wf.setnchannels(self.CHANNELS)
                        wf.setsampwidth(self.SAMPLE_WIDTH)
                        wf.setframerate(self.SAMPLE_RATE)
                        wf.writeframes(audio_bytes)

                    # Load → master → export
                    sound  = AudioSegment.from_file(temp_raw)
                    sound  = self._master(sound, style_instruction)
                    sound.export(output_path, format="wav")

                    # Clean up temp raw
                    if os.path.exists(temp_raw):
                        os.remove(temp_raw)

                    dur = len(sound) / 1000.0
                    print(f"      ✅ Line {index} [{voice_name}] → {dur:.2f}s")
                    return output_path

                except Exception as e:
                    err = str(e)
                    if "429" in err or "503" in err or "RESOURCE_EXHAUSTED" in err:
                        wait = 35 + (attempt * 12)
                        print(f"      ⏳ Rate limit — waiting {wait}s "
                              f"(attempt {attempt + 1}/3, model={model_name})")
                        time.sleep(wait)
                    else:
                        print(f"      ⚠️  TTS error [{model_name}] "
                              f"attempt {attempt + 1}: {e}")
                        break  # Non-retriable — try next model immediately

            time.sleep(4)  # Brief pause between model switches

        # Clean up any leftover temp file
        if os.path.exists(temp_raw):
            try:
                os.remove(temp_raw)
            except Exception:
                pass

        print(f"      ❌ Line {index} — all TTS attempts exhausted.")
        return None
