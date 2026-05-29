# -*- coding: utf-8 -*-
"""
TADAC 로컬 Whisper STT API 서버 (학교 GPU 서버 배포용)

faster-whisper (CTranslate2) 기반 — openai/whisper 대비 4~6배 빠르고 VRAM 50% 절감.
API 규격은 OpenAI Audio Transcriptions API를 모방하여 기존 클라이언트 호환.

실행:
    pip install faster-whisper
    uvicorn app_whisper:app --host 0.0.0.0 --port 8002
"""

import os
import tempfile
import time
from pathlib import Path

import torch
from faster_whisper import WhisperModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="TADAC Whisper STT Server", version="2.0.0")

# ── 모델 로드 (서버 기동 시 1회) ────────────────────────────────────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")

print(f"[Whisper] faster-whisper 모델 로딩: {WHISPER_MODEL_SIZE} on {DEVICE} ({COMPUTE_TYPE})")
whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
)
print(f"[Whisper] 모델 로딩 완료")

REPLACEMENT_CHAR = "�"


@app.post("/v1/audio/transcriptions")
async def transcribe(request: Request):
    """OpenAI Audio Transcriptions API 호환 엔드포인트.

    multipart/form-data 로 오디오 파일과 옵션을 받아
    faster-whisper 모델로 추론 후 결과를 JSON 으로 반환한다.
    """
    form = await request.form()

    file = form.get("file")
    if file is None:
        raise HTTPException(status_code=400, detail="file field is required")

    language = form.get("language") or None
    prompt = form.get("prompt") or None
    response_format = form.get("response_format", "json")

    suffix = Path(getattr(file, "filename", "audio.mp3")).suffix or ".mp3"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="whisper_")
    os.close(tmp_fd)

    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        print(
            f"[Whisper] 추론 시작: {getattr(file, 'filename', '?')} "
            f"({len(content) / 1024 / 1024:.1f} MB) lang={language}"
        )

        t0 = time.time()

        # faster-whisper transcribe
        segments_iter, info = whisper_model.transcribe(
            tmp_path,
            language=language,
            initial_prompt=prompt,
            word_timestamps=True,
            beam_size=5,
            vad_filter=True,          # VAD로 무음 구간 스킵 → 추가 속도 향상
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
        )

        # segments는 generator이므로 순회하며 수집
        text_parts = []
        words = []
        segments_out = []
        duration = 0.0

        for seg in segments_iter:
            seg_start = round(seg.start, 3)
            seg_end = round(seg.end, 3)
            seg_text = seg.text.strip()

            text_parts.append(seg_text)

            segments_out.append({
                "id": len(segments_out),
                "start": seg_start,
                "end": seg_end,
                "text": seg_text,
            })

            if seg_end > duration:
                duration = seg_end

            # 단어 타임스탬프 평탄화
            if seg.words:
                for w in seg.words:
                    word_text = w.word or ""
                    if REPLACEMENT_CHAR in word_text:
                        word_text = word_text.replace(REPLACEMENT_CHAR, "")
                    if not word_text:
                        continue
                    words.append({
                        "word": word_text,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                    })

        text = " ".join(text_parts).strip()
        elapsed = time.time() - t0
        detected_lang = info.language or language or "ko"

        print(
            f"[Whisper] 추론 완료: 단어 {len(words)}개, "
            f"세그먼트 {len(segments_out)}개, {duration:.1f}초 분량, "
            f"소요 {elapsed:.1f}초 (x{duration / elapsed:.1f} realtime)"
        )

        if response_format == "json":
            return JSONResponse({"text": text})

        # verbose_json (OpenAI SDK TranscriptionVerbose 호환)
        return JSONResponse({
            "task": "transcribe",
            "language": detected_lang,
            "duration": round(duration, 3),
            "text": text,
            "words": words,
            "segments": segments_out,
        })

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/health")
async def health():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "status": "ok",
        "model": WHISPER_MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "engine": "faster-whisper",
        "gpu": gpu_name,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_whisper:app", host="0.0.0.0", port=8002)
