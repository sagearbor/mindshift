#!/usr/bin/env python3
"""
Audio Emotion Analyzer CLI using Google GenAI SDK
=================================================
Analyzes vocal tone, pitch, and rhythm from WAV audio files to detect emotions
using Gemini (defaulting to Gemini 3.8 Flash).

Features:
  - Multi-file batch processing: iterates through a list of WAV files, uploading
    and analyzing each while ensuring errors are isolated per file and processing continues.
  - File upload confirmation message: Displays 'File [filename] uploaded successfully.'
  - Audio playback: plays WAV files using pygame.mixer (with fallback and graceful error handling)
    via --play or --interactive.
  - Programmatic API support: Can be imported and called directly by AI agents
    (Claude Code, OpenAI Codex) via `analyze_audio_files()` or invoked via CLI subprocess.
  - Strict JSON array schema adhering to prompt requirements:
    'filename', 'primary_emotion', 'alternative_emotion', 'confidence', 'audible_evidence'.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# Suppress ALSA / sound driver noise on headless Linux systems if needed
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def load_env_file(filepath: str = ".env") -> None:
    """Built-in .env parser so no external python-dotenv package is strictly required."""
    env_path = Path(filepath)
    if not env_path.is_file():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass


load_env_file(str(Path(__file__).resolve().parents[2] / ".env"))
_ca_bundle = Path.home() / ".cache/macos-ca-bundle.pem"
if _ca_bundle.is_file():
    os.environ.setdefault("SSL_CERT_FILE", str(_ca_bundle))

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    from pydantic import BaseModel, Field
except ImportError as err:
    sys.stderr.write(
        f"Error: Missing required Python dependencies ({err}).\n"
        "Please install the Google GenAI SDK and Pydantic:\n"
        "  pip install google-genai pydantic\n"
    )
    sys.exit(1)


# Allowed primary emotions per specification
PrimaryEmotionType = Literal[
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "other",
    "uncertain",
]


class EmotionAnalysis(BaseModel):
    filename: str = Field(
        description="The filename or path of the analyzed WAV audio file"
    )
    primary_emotion: PrimaryEmotionType = Field(
        description="The dominant perceived emotion in the audio: anger, disgust, fear, happiness, neutral, sadness, other, or uncertain."
    )
    alternative_emotion: Optional[str] = Field(
        default=None,
        description="A secondary or alternative emotion or emotional nuance observed in the audio."
    )
    confidence: int = Field(
        ge=0,
        le=100,
        description="Confidence score for the primary emotion from 0 to 100.",
    )
    audible_evidence: str = Field(
        description="A short sentence describing the specific vocal tone, pitch, pace, volume, or rhythm that indicates the emotion."
    )


def parse_emotion_response(response: Any, filename: str) -> EmotionAnalysis:
    """Reject incomplete generation before admitting a schema-validated label."""
    candidates = response.candidates or []
    if len(candidates) != 1:
        raise ValueError('Expected one completed response candidate')
    reason = candidates[0].finish_reason
    reason = getattr(reason, 'value', reason)
    if reason != 'STOP':
        raise ValueError(f'Incomplete or blocked response: {reason}')
    raw_text = response.text
    if not raw_text or not raw_text.strip():
        raise ValueError('Gemini returned an empty response')
    result = json.loads(raw_text)
    if not isinstance(result, dict):
        raise ValueError('Expected one JSON object')
    result['filename'] = filename
    return EmotionAnalysis.model_validate(result)


def validate_wav_file(filepath: str) -> None:
    """
    Validates file accessibility, permissions, non-emptiness, and WAV structure.
    Raises specific, user-friendly exceptions for any file access issue.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: '{filepath}'. Please verify the path and try again."
        )

    if path.is_dir():
        raise IsADirectoryError(
            f"Path is a directory, not a file: '{filepath}'. Please specify a WAV audio file."
        )

    if not os.access(path, os.R_OK):
        raise PermissionError(
            f"Permission denied: Cannot read file '{filepath}'. Check file permissions."
        )

    try:
        size = path.stat().st_size
    except OSError as e:
        raise OSError(f"Could not access file statistics for '{filepath}': {e}")

    if size == 0:
        raise ValueError(
            f"File is empty (0 bytes): '{filepath}'. Cannot analyze an empty audio file."
        )

    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()

            if channels <= 0 or sample_rate <= 0:
                raise ValueError(
                    f"Invalid WAV parameters in '{filepath}': channels={channels}, sample_rate={sample_rate}."
                )

            duration = n_frames / float(sample_rate)
            if duration <= 0:
                raise ValueError(f"WAV audio in '{filepath}' has zero duration.")
    except wave.Error as wave_err:
        raise ValueError(
            f"Invalid or corrupted WAV file '{filepath}': {wave_err}. "
            "Make sure the file is a standard RIFF/WAVE audio file."
        )
    except Exception as e:
        if not isinstance(e, ValueError):
            raise ValueError(f"Could not read audio header for '{filepath}': {e}")
        raise


def play_audio_file(filepath: str, block: bool = True) -> bool:
    """
    Plays the given WAV file using pygame.mixer (with graceful fallback and error handling).
    Handles potential playback errors (e.g., missing audio device, codec failure).
    """
    base_name = os.path.basename(filepath)
    try:
        import pygame

        # Try to initialize pygame.mixer if not already active
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init()
            except Exception as init_err:
                # On headless systems or missing hardware, report cleanly
                sys.stderr.write(
                    f"[Playback Notice] Could not initialize audio mixer for '{base_name}': {init_err}\n"
                )
                return False

        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()
        sys.stderr.write(f"[Playback] Playing audio file '{base_name}'...\n")

        if block:
            import time
            while pygame.mixer.music.get_busy():
                time.sleep(0.08)

        return True
    except Exception as playback_err:
        sys.stderr.write(
            f"[Playback Error] Failed to play audio file '{base_name}': {playback_err}\n"
        )
        return False


def upload_audio_to_gemini(
    client: genai.Client,
    filepath: str,
    verbose: bool = False,
) -> Any:
    """
    Uploads a WAV file to the Gemini API and waits for it to become ACTIVE.
    Prints upload confirmation message: 'File [filename] uploaded successfully.'
    """
    abs_path = os.path.abspath(filepath)
    base_name = os.path.basename(filepath)

    if verbose:
        sys.stderr.write(f"[*] Uploading '{base_name}' to Gemini Files API...\n")

    uploaded_file = client.files.upload(
        file=abs_path,
        config=types.UploadFileConfig(mime_type="audio/wav", display_name="audio-clip.wav"),
    )

    # Wait for file processing if needed
    max_wait_seconds = 20
    waited = 0
    while (
        uploaded_file.state
        and uploaded_file.state.name == "PROCESSING"
        and waited < max_wait_seconds
    ):
        if verbose:
            sys.stderr.write(f"[*] Processing audio in Gemini API ({waited}s)...\n")
        import time
        time.sleep(1)
        waited += 1
        uploaded_file = client.files.get(name=uploaded_file.name)

    if not uploaded_file.state or uploaded_file.state.name != "ACTIVE":
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            sys.stderr.write('[Cleanup notice] Could not delete incomplete upload.\n')
        raise RuntimeError(
            f"Uploaded audio did not become ACTIVE for '{base_name}'."
        )

    # Explicit confirmation message as requested:
    # 'File [filename] uploaded successfully.'
    sys.stderr.write(f"File [{base_name}] uploaded successfully.\n")
    sys.stderr.flush()
    return uploaded_file


def analyze_single_file(
    client: genai.Client,
    filepath: str,
    model_name: str = "gemini-3.8-flash",
    play_audio: bool = False,
    interactive_playback: bool = False,
    verbose: bool = False,
) -> EmotionAnalysis:
    """
    Validates, optionally plays, uploads, and analyzes a single WAV audio file.
    Cleans up the remote uploaded file in Gemini storage upon completion.
    """
    # 1. Validate file accessibility and WAV structure
    validate_wav_file(filepath)
    base_name = os.path.basename(filepath)

    # 2. Trigger audio playback if requested
    if interactive_playback and sys.stdin.isatty():
        try:
            choice = input(f"Play audio file [{base_name}]? [y/N]: ").strip().lower()
            if choice in ("y", "yes"):
                play_audio_file(filepath, block=True)
        except (EOFError, KeyboardInterrupt):
            pass
    elif play_audio:
        play_audio_file(filepath, block=False)

    # 3. Upload to Gemini API
    uploaded_file = None
    try:
        uploaded_file = upload_audio_to_gemini(client, filepath, verbose=verbose)

        # 4. Analyze using the explicitly selected model; never silently substitute another.
        if verbose:
            sys.stderr.write(f"[*] Analyzing vocal affect with {model_name}...\n")

        prompt = (
            "You are an expert acoustic analyst specializing in vocal affect and emotion recognition. "
            "Listen directly to the attached audio; use vocal expression, not merely sentence meaning. "
            "Ignore filenames, metadata and dataset knowledge. If audio cannot be heard, report uncertain and say audio unavailable. "
            "Identify the primary emotion conveyed by the speaker or vocal tone, choosing strictly "
            "from: anger, disgust, fear, happiness, neutral, sadness, other, or uncertain. "
            "Provide an alternative or secondary emotional nuance, a confidence score from 0 to 100, "
            "and a concise 'audible_evidence' sentence explicitly highlighting specific vocal characteristics "
            "such as pitch variations, vocal fry/tremor, cadence, volume changes, breathiness, or rhythm."
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EmotionAnalysis,
            max_output_tokens=40000,
        )

        response = client.models.generate_content(
            model=model_name, contents=[uploaded_file, prompt], config=config
        )
        return parse_emotion_response(response, base_name)

    finally:
        # Clean up remote file from Gemini Files API
        if uploaded_file and hasattr(uploaded_file, "name"):
            try:
                if verbose:
                    sys.stderr.write(f"[*] Cleaning up remote file {uploaded_file.name}...\n")
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass


def analyze_audio_files(
    files: List[str],
    api_key: Optional[str] = None,
    model: str = "gemini-3.8-flash",
    play_audio: bool = False,
    interactive_playback: bool = False,
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Programmatic entry point for Claude Code, Codex, or Python scripts.
    Iterates through a list of WAV files, uploading and analyzing each with Gemini.
    Errors are isolated per file so that one failed file does not abort the batch.

    Returns:
      (successful_results, error_dict)
      - successful_results: list of dictionaries conforming to the EmotionAnalysis schema.
      - error_dict: map of {filepath: error_message} for any failed files.
    """
    resolved_key = api_key or (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    if not resolved_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please set the environment variable or pass api_key."
        )

    client = genai.Client(
        api_key=resolved_key,
        http_options={"headers": {"User-Agent": "aistudio-build"}},
    )

    results: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}

    for filepath in files:
        try:
            analysis = analyze_single_file(
                client=client,
                filepath=filepath,
                model_name=model,
                play_audio=play_audio,
                interactive_playback=interactive_playback,
                verbose=verbose,
            )
            results.append(analysis.model_dump())
        except Exception as file_exc:
            err_msg = str(file_exc).replace(resolved_key, "[REDACTED]")
            errors[filepath] = err_msg
            sys.stderr.write(f"[ERROR] Failed to process '{filepath}': {err_msg}\n")
            if isinstance(file_exc, APIError):
                break  # Provider/auth/quota errors abort the batch; never become emotion labels.

    return results, errors


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="audio_emotion_analyzer.py",
        description="Analyze emotions in WAV audio files using Google GenAI SDK and Gemini 3.8 Flash.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples for Humans & Programmatic Agents (Claude Code / Codex):
  # Single WAV file
  python3 audio_emotion_analyzer.py voice.wav

  # List of multiple WAV files (continues processing if one fails)
  python3 audio_emotion_analyzer.py audio1.wav audio2.wav missing.wav audio3.wav

  # Play audio during analysis using pygame.mixer
  python3 audio_emotion_analyzer.py voice.wav --play

  # Interactive playback prompt
  python3 audio_emotion_analyzer.py voice.wav --interactive

  # Programmatic / piping mode (clean JSON array output for scripts & Codex)
  python3 audio_emotion_analyzer.py *.wav --compact

  # Save output JSON array directly to a file
  python3 audio_emotion_analyzer.py *.wav --output results.json
        """,
    )

    parser.add_argument(
        "files",
        nargs="+",
        metavar="FILE.wav",
        help="One or more WAV audio files to analyze in sequence.",
    )

    parser.add_argument(
        "--model",
        "-m",
        default="gemini-3.8-flash",
        help="Gemini model to use for analysis (default: 'gemini-3.8-flash').",
    )

    parser.add_argument(
        "--play",
        "-p",
        action="store_true",
        help="Trigger audio playback of the WAV file via pygame.mixer.",
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Prompt the user interactively whether to play each WAV file.",
    )

    parser.add_argument(
        "--api-key",
        "-k",
        default=None,
        help="Gemini API key (defaults to GEMINI_API_KEY environment variable).",
    )

    parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="OUT_FILE",
        help="Save the resulting JSON array to the specified file path.",
    )

    parser.add_argument(
        "--compact",
        action="store_true",
        help="Output compact single-line JSON array instead of indented JSON.",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a human-readable emotion summary table to stderr alongside JSON.",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed upload and diagnostic progress logs to stderr.",
    )

    return parser.parse_args()


def print_summary_table(results: List[Dict[str, Any]], errors: Dict[str, str]) -> None:
    """Print an eye-catching ASCII table summary of analysis results and file errors to stderr."""
    sys.stderr.write("\n" + "=" * 80 + "\n")
    sys.stderr.write("AUDIO EMOTION BATCH ANALYSIS RESULTS\n")
    sys.stderr.write("=" * 80 + "\n")

    for r in results:
        sys.stderr.write(f"File:       {r['filename']}\n")
        sys.stderr.write(
            f"Primary:    {r['primary_emotion'].upper()} ({r['confidence']}% confidence)\n"
        )
        sys.stderr.write(f"Secondary:  {r['alternative_emotion']}\n")
        sys.stderr.write(f"Evidence:   {r['audible_evidence']}\n")
        sys.stderr.write("-" * 80 + "\n")

    if errors:
        sys.stderr.write(f"FAILED FILES ({len(errors)}):\n")
        for f, err in errors.items():
            sys.stderr.write(f"  - {f}: {err}\n")
        sys.stderr.write("=" * 80 + "\n")


def main() -> None:
    args = parse_arguments()

    # Determine API key
    api_key = args.api_key or (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    if not api_key:
        sys.stderr.write(
            "Error: GEMINI_API_KEY is not set.\n\n"
            "Please provide a Gemini API key via one of:\n"
            "  1. Environment variable: export GEMINI_API_KEY='your_api_key'\n"
            "  2. CLI argument:         python3 audio_emotion_analyzer.py sample.wav --api-key 'your_api_key'\n"
            "  3. In a .env file:       GEMINI_API_KEY=your_api_key\n"
        )
        sys.exit(1)

    # Process all files with isolated per-file error handling
    results, errors = analyze_audio_files(
        files=args.files,
        api_key=api_key,
        model=args.model,
        play_audio=args.play,
        interactive_playback=args.interactive,
        verbose=args.verbose,
    )

    # Format output as pure JSON array to stdout (crucial for Claude Code, Codex, and pipes)
    indent = None if args.compact else 2
    json_output_str = json.dumps(results, indent=indent)
    print(json_output_str, flush=True)

    # Save to file if requested
    if args.output:
        try:
            out_path = Path(args.output)
            out_path.write_text(json_output_str, encoding="utf-8")
            if args.verbose:
                sys.stderr.write(f"[*] Results saved to '{args.output}'.\n")
        except Exception as e:
            sys.stderr.write(f"Warning: Could not save output to '{args.output}': {e}\n")

    if args.summary:
        print_summary_table(results, errors)

    # Exit code strategy:
    # 0 = All files succeeded
    # 2 = Some files succeeded and some failed (partial batch success)
    # 1 = All files failed
    if not results and errors:
        sys.exit(1)
    elif errors:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
