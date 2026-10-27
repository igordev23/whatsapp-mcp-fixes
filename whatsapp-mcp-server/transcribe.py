"""Transcrição local de áudio via whisper.cpp — offline, sem API, sem custo."""

import os
import shutil
import subprocess
import tempfile

# Modelo GGML do Whisper. Sobrescrevível por env var para trocar de tamanho
# sem mexer no código.
DEFAULT_MODEL = os.path.expanduser(
    os.environ.get(
        "WHISPER_MODEL",
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "models", "ggml-large-v3-turbo-q5_0.bin"),
    )
)

# whisper.cpp só aceita WAV PCM 16 kHz mono. Áudio de WhatsApp vem em Opus/OGG,
# então a conversão não é opcional.
WHISPER_SAMPLE_RATE = 16000


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise RuntimeError(
            f"'{binary}' não encontrado. Instale com: sudo apt install "
            f"{'whisper.cpp' if binary.startswith('whisper') else binary} "
            f"(macOS: brew install {'whisper-cpp' if binary.startswith('whisper') else binary})"
        )
    return path


def to_whisper_wav(input_file: str, output_file: str) -> str:
    """Converte qualquer áudio para o WAV que o whisper.cpp exige."""
    cmd = [
        _require("ffmpeg"),
        "-i", input_file,
        "-ar", str(WHISPER_SAMPLE_RATE),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-y",
        output_file,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg falhou ao converter o áudio: {proc.stderr[-400:]}")
    return output_file


def transcribe_file(
    audio_path: str,
    language: str = "pt",
    model_path: str | None = None,
) -> str:
    """Transcreve um arquivo de áudio local. Devolve o texto puro."""
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Áudio não encontrado: {audio_path}")

    model = model_path or DEFAULT_MODEL
    if not os.path.isfile(model):
        raise RuntimeError(
            f"Modelo Whisper não encontrado em {model}. Baixe um .bin de "
            "https://huggingface.co/ggerganov/whisper.cpp (gratuito) ou aponte "
            "a env var WHISPER_MODEL para outro caminho."
        )

    with tempfile.TemporaryDirectory() as tmp:
        wav = to_whisper_wav(audio_path, os.path.join(tmp, "audio.wav"))
        cmd = [
            _require("whisper-cli"),
            "-m", model,
            "-f", wav,
            "-l", language,
            "-nt",   # sem timestamps: queremos texto corrido
            "-np",   # sem barulho de progresso no stdout
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"whisper-cli falhou: {proc.stderr[-400:]}")

    return " ".join(proc.stdout.split()).strip()
