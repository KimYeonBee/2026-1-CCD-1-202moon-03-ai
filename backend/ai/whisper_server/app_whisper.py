# -*- coding: utf-8 -*-
"""
TADAC 로컬 Whisper STT API 서버 (학교 GPU 서버 배포용)

OpenAI Audio Transcriptions API 규격을 모방하여
EC2 오케스트레이터가 OpenAI SDK(base_url 변경)로 호출할 수 있도록 한다.

실행:
    uvicorn app_whisper:app --host 0.0.0.0 --port 8002
"""

import os
import tempfile
from pathlib import Path

import torch
import whisper
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="TADAC Whisper STT Server", version="1.0.0")

# ── 모델 로드 (서버 기동 시 1회) ────────────────────────────────────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[Whisper] 모델 로딩: {WHISPER_MODEL_SIZE} on {DEVICE}")
whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=DEVICE)
print(f"[Whisper] 모델 로딩 완료")

REPLACEMENT_CHAR = "�"


@app.post("/v1/audio/transcriptions")
async def transcribe(request: Request):
    """OpenAI Audio Transcriptions API 호환 엔드포인트.

    multipart/form-data 로 오디오 파일과 옵션을 받아
    로컬 Whisper 모델로 추론 후 결과를 JSON 으로 반환한다.
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

        options = {
            "word_timestamps": True,
            "fp16": (DEVICE == "cuda"),
        }
        if language:
            options["language"] = language
        if prompt:
            options["initial_prompt"] = prompt

        result = whisper_model.transcribe(tmp_path, **options)

        text = (result.get("text") or "").strip()
        detected_lang = result.get("language", language or "ko")

        # 세그먼트에서 단어 타임스탬프를 평탄화
        words = []
        segments_out = []
        duration = 0.0

        for seg in result.get("segments", []):
            seg_start = round(seg.get("start", 0.0), 3)
            seg_end = round(seg.get("end", 0.0), 3)
            seg_text = seg.get("text", "")

            segments_out.append({
                "id": seg.get("id", len(segments_out)),
                "start": seg_start,
                "end": seg_end,
                "text": seg_text,
            })

            if seg_end > duration:
                duration = seg_end

            for w in seg.get("words", []):
                word_text = w.get("word", "")
                if REPLACEMENT_CHAR in word_text:
                    word_text = word_text.replace(REPLACEMENT_CHAR, "")
                if not word_text:
                    continue
                words.append({
                    "word": word_text,
                    "start": round(w.get("start", 0.0), 3),
                    "end": round(w.get("end", 0.0), 3),
                })

        print(
            f"[Whisper] 추론 완료: 단어 {len(words)}개, "
            f"세그먼트 {len(segments_out)}개, {duration:.1f}초"
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
        "gpu": gpu_name,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_whisper:app", host="0.0.0.0", port=8002)
