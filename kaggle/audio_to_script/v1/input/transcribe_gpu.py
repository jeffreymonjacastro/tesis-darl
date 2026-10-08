"""GPU transcription runner for Kaggle T4 — faster-whisper large-v3."""

import subprocess
import sys

# Install faster-whisper first before importing
print("Installing faster-whisper...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "faster-whisper"])
print("faster-whisper installed.", flush=True)

import json
import time
import traceback
from pathlib import Path
import datetime

from faster_whisper import WhisperModel

def format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def main() -> None:
    output_dir = Path("/kaggle/working")
    progress_path = output_dir / "run_summary.json"
    progress = {"status": "running", "stage": "start", "error": None}

    def save_progress():
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    try:
        save_progress()

        # --- CONFIGURATION ---
        WHISPER_MODEL = "large-v3"
        LANGUAGE = "es"
        BEAM_SIZE = 5
        TEMPERATURE = 0.0
        VAD_FILTER = True
        VAD_PARAMS = {"min_silence_duration_ms": 350, "speech_pad_ms": 180}
        INITIAL_PROMPT = (
            "Reunión de asesoría de tesis DARL: Drift-Aware Reinforcement Learning for Selective "
            "Updating of Two-Stage Tabular ML Pipelines. Participan Jeffrey, la asesora Ariana y "
            "Dayane. Se habla de data drift, concept drift, covariate shift, PPO, DQN, Q-learning, "
            "POMDP, AUC, PSI, KS, C2ST, XGBoost, TableShift, PhysioNet, PFC1 y PFC2."
        )
        # --- PATHS ---
        input_dir = Path("/kaggle/input")

        progress["stage"] = "find_audio"
        save_progress()

        audio_files = list(input_dir.rglob("*.m4a")) + list(input_dir.rglob("*.wav")) + list(input_dir.rglob("*.mp3"))
        if not audio_files:
            raise FileNotFoundError(f"No audio files found recursively in {input_dir}")
        audio_path = audio_files[0]
        audio_label = audio_path.name
        audio_stem = audio_path.stem

        if audio_stem.lower().startswith("asesoria"):
            PARTICIPANTS = "Jeffrey, Ariana y Dayane"
        elif audio_stem.lower().startswith("osman"):
            PARTICIPANTS = "Jeffrey, Dayane y Osman"
        else:
            PARTICIPANTS = "Desconocido"

        print(f"Found audio: {audio_path}, Participants: {PARTICIPANTS}", flush=True)

        progress["stage"] = "load_model"
        save_progress()

        local_model_path = "/kaggle/input/faster-whisper-large-v3"
        print(f"Loading from {local_model_path} on CUDA/float16...", flush=True)
        model = WhisperModel(local_model_path, device="cuda", compute_type="float16")

        progress["stage"] = "transcribe"
        save_progress()

        print(f"Starting transcription...", flush=True)
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=LANGUAGE,
            beam_size=BEAM_SIZE,
            best_of=BEAM_SIZE,
            temperature=TEMPERATURE,
            vad_filter=VAD_FILTER,
            vad_parameters=VAD_PARAMS,
            word_timestamps=True,
            condition_on_previous_text=True,
            initial_prompt=INITIAL_PROMPT,
        )

        duration_label = format_time(info.duration)

        started = time.time()
        segments = []
        for index, segment in enumerate(segments_iter, start=1):
            words = [
                {
                    "start": word.start,
                    "end": word.end,
                    "word": word.word,
                    "probability": word.probability,
                }
                for word in (segment.words or [])
            ]
            segments.append(
                {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                    "words": words,
                }
            )

            if index % 20 == 0:
                elapsed = time.time() - started
                print(f"segments={index} audio={segment.end:.1f}s elapsed={elapsed:.1f}s", flush=True)

            if index % 50 == 0:
                # Checkpoint
                checkpoint_data = {"segments": segments}
                (output_dir / "transcript_partial.json").write_text(json.dumps(checkpoint_data, ensure_ascii=False, indent=2), encoding="utf-8")

        progress["stage"] = "write_outputs"
        save_progress()

        # 1. JSON complete
        payload = {
            "audio": str(audio_path),
            "model": WHISPER_MODEL,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "segments": segments,
        }
        json_path = output_dir / "transcript.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # 2. Markdown
        md_lines = []
        md_lines.append(f"# Transcripción de la asesoría — Semana X\n")
        md_lines.append(f"**Audio:** `{audio_label}`  ")
        md_lines.append(f"**Participantes:** {PARTICIPANTS}  ")
        md_lines.append(f"**Duración:** {duration_label}\n")
        md_lines.append(f"> Los fragmentos que no pudieron distinguirse con suficiente claridad están marcados como `[inaudible]`.\n")

        for segment in segments:
            start_fmt = format_time(segment["start"])
            end_fmt = format_time(segment["end"])
            text = segment["text"]
            # Basic replacement for low probability words could be added here if needed,
            # but usually [inaudible] is manual or based on specific word logprobs.
            # We'll just write the text.
            md_lines.append(f"{start_fmt} - {end_fmt} [Hablante]: \"{text}\"\n")

        md_path = output_dir / f"{audio_stem}_transcription.md"
        md_path.write_text("\n".join(md_lines), encoding="utf-8")

        # Clean partial
        if (output_dir / "transcript_partial.json").exists():
            (output_dir / "transcript_partial.json").unlink()

        progress["status"] = "complete"
        progress["stage"] = "done"
        progress["model"] = WHISPER_MODEL
        progress["duration"] = info.duration
        progress["time_elapsed"] = time.time() - started
        progress["segments_count"] = len(segments)

        print(f"Finished successfully. Wrote {len(segments)} segments.", flush=True)

    except Exception as exc:
        progress["status"] = "failed"
        progress["error"] = repr(exc)
        progress["traceback"] = traceback.format_exc()
        print(f"Error: {repr(exc)}", flush=True)
    finally:
        save_progress()

if __name__ == "__main__":
    main()
