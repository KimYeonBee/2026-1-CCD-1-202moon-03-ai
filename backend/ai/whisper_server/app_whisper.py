# -*- coding: utf-8 -*-
"""
TADAC 로컬 Whisper STT API 서버 (멀티 GPU 워커 풀)

faster-whisper (CTranslate2) 기반 — openai/whisper 대비 4~6배 빠르고 VRAM 50% 절감.
API 규격은 OpenAI Audio Transcriptions API를 모방하여 기존 클라이언트 호환.

멀티 GPU 지원:
    - 서버 기동 시 사용 가능한 모든 GPU에 Whisper 모델을 로드
    - 요청이 들어오면 빈 GPU 워커에 자동 배정 (모두 바쁘면 대기)
    - GET /gpu-status 로 현재 GPU 상태 확인 가능

실행:
    pip install faster-whisper
    uvicorn app_whisper:app --host 0.0.0.0 --port 8002
"""

import asyncio
import os
import tempfile
import time
import uuid
from pathlib import Path

import torch
from faster_whisper import WhisperModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="TADAC Whisper STT Server", version="3.0.0")

# ── 설정 ─────────────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "/home/202moon/whisper_server/faster-whisper-large-v3"
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

# 특정 GPU만 사용하려면 환경변수로 지정 (예: "0,2" → GPU 0, 2만 사용)
# 미설정 시 사용 가능한 모든 GPU 사용
WHISPER_GPUS = os.getenv("WHISPER_GPUS", None)

REPLACEMENT_CHAR = "�"


def _is_retryable_cuda_error(exc):
    msg = str(exc).lower()
    return "cuda failed" in msg or "cuda error" in msg or "invalid argument" in msg


def _clear_cuda_cache(gpu_id):
    if gpu_id is None or gpu_id < 0 or not torch.cuda.is_available():
        return
    try:
        with torch.cuda.device(gpu_id):
            torch.cuda.empty_cache()
    except Exception as cache_error:
        print(f"[Whisper] GPU {gpu_id} cache clear 실패: {cache_error}")


# ── 멀티 GPU 워커 풀 ─────────────────────────────────────────────────────────

class GPUWorkerPool:
    """GPU별 Whisper 모델을 관리하는 워커 풀.

    - 각 GPU에 독립적인 WhisperModel 인스턴스를 로드
    - asyncio.Queue 로 빈 워커 관리 → 요청이 오면 빈 GPU에 배정
    - 모든 GPU가 바쁘면 하나가 끝날 때까지 대기 (backpressure)
    """

    def __init__(self):
        self.workers = {}      # gpu_id → {"model": WhisperModel, "gpu_name": str}
        self._queue = None     # asyncio.Queue — event loop 안에서 초기화

    def load_models(self, model_size, compute_type, gpu_ids=None):
        """서버 기동 시 호출 — 각 GPU에 모델 로드 (동기)."""
        if not torch.cuda.is_available():
            # CPU 폴백
            print(f"[Whisper] CUDA 사용 불가 → CPU 모드")
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            self.workers[-1] = {"model": model, "gpu_name": "CPU"}
            print(f"[Whisper] CPU 워커 로드 완료")
            return

        available_gpus = list(range(torch.cuda.device_count()))

        if gpu_ids is not None:
            # 환경변수로 지정된 GPU만 사용
            target_gpus = [g for g in gpu_ids if g in available_gpus]
        else:
            target_gpus = available_gpus

        if not target_gpus:
            raise RuntimeError(f"사용 가능한 GPU 없음. 전체: {available_gpus}, 요청: {gpu_ids}")

        print(f"[Whisper] 멀티 GPU 모델 로딩: {model_size} × {len(target_gpus)}대 ({compute_type})")

        for gpu_id in target_gpus:
            gpu_name = torch.cuda.get_device_name(gpu_id)
            try:
                print(f"[Whisper]   GPU {gpu_id} ({gpu_name}) 로딩 중...")
                model = WhisperModel(
                    model_size,
                    device="cuda",
                    device_index=gpu_id,
                    compute_type=compute_type,
                )
                self.workers[gpu_id] = {"model": model, "gpu_name": gpu_name}
                print(f"[Whisper]   GPU {gpu_id} ({gpu_name}) 로딩 완료 ✓")
            except Exception as e:
                print(f"[Whisper]   GPU {gpu_id} ({gpu_name}) 로딩 실패: {e}")

        if not self.workers:
            raise RuntimeError("모든 GPU 로딩 실패")

        print(f"[Whisper] 워커 풀 준비 완료: {len(self.workers)}대 GPU")

    def _ensure_queue(self):
        """asyncio.Queue는 이벤트 루프 안에서만 생성 가능 → 최초 호출 시 lazy init."""
        if self._queue is None:
            self._queue = asyncio.Queue()
            for gpu_id in self.workers:
                self._queue.put_nowait(gpu_id)

    async def acquire(self):
        """빈 GPU 워커를 가져온다. 모두 바쁘면 하나가 끝날 때까지 대기."""
        self._ensure_queue()
        gpu_id = await self._queue.get()
        return gpu_id, self.workers[gpu_id]["model"]

    def release(self, gpu_id):
        """사용 완료된 GPU를 풀에 반환."""
        self._ensure_queue()
        self._queue.put_nowait(gpu_id)

    def status(self):
        """현재 GPU 풀 상태 반환."""
        self._ensure_queue()
        free_count = self._queue.qsize()
        total = len(self.workers)

        # 어떤 GPU가 현재 큐에 있는지 확인 (peek)
        # asyncio.Queue는 내부 _queue 속성으로 접근 가능
        free_ids = set()
        try:
            free_ids = set(self._queue._queue)
        except Exception:
            pass

        worker_status = []
        for gpu_id, info in sorted(self.workers.items()):
            worker_status.append({
                "gpu_id":   gpu_id,
                "gpu_name": info["gpu_name"],
                "busy":     gpu_id not in free_ids,
            })

        return {
            "total_gpus": total,
            "free_gpus":  free_count,
            "busy_gpus":  total - free_count,
            "workers":    worker_status,
        }


# ── 워커 풀 초기화 ───────────────────────────────────────────────────────────
gpu_pool = GPUWorkerPool()

# 환경변수 파싱
_gpu_ids = None
if WHISPER_GPUS:
    try:
        _gpu_ids = [int(g.strip()) for g in WHISPER_GPUS.split(",")]
        print(f"[Whisper] 지정 GPU: {_gpu_ids}")
    except ValueError:
        print(f"[Whisper] WHISPER_GPUS 파싱 실패: '{WHISPER_GPUS}' → 전체 GPU 사용")

gpu_pool.load_models(WHISPER_MODEL_SIZE, COMPUTE_TYPE, gpu_ids=_gpu_ids)


# ── 추론 함수 (동기 — asyncio.to_thread로 호출) ─────────────────────────────

def _do_transcribe(model, tmp_path, language, prompt, filename, request_id=None, gpu_id=None):
    """동기 Whisper 추론 — GPU 워커 스레드에서 실행."""
    t0 = time.time()

    segments_iter, info = model.transcribe(
        tmp_path,
        language=language,
        initial_prompt=prompt,
        word_timestamps=False,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

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

        for w in getattr(seg, "words", None) or []:
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
        f"[Whisper] 추론 완료[{request_id}]: {filename} GPU {gpu_id} | 단어 {len(words)}개, "
        f"세그먼트 {len(segments_out)}개, {duration:.1f}초 분량, "
        f"소요 {elapsed:.1f}초 (x{duration / elapsed:.1f} realtime)"
    )

    return {
        "text": text,
        "words": words,
        "segments": segments_out,
        "duration": duration,
        "detected_lang": detected_lang,
    }


# ── API 엔드포인트 ───────────────────────────────────────────────────────────

@app.post("/v1/audio/transcriptions")
async def transcribe(request: Request):
    """OpenAI Audio Transcriptions API 호환 엔드포인트.

    multipart/form-data 로 오디오 파일과 옵션을 받아
    빈 GPU 워커에 배정하여 추론 후 결과를 JSON으로 반환.
    """
    form = await request.form()

    file = form.get("file")
    if file is None:
        raise HTTPException(status_code=400, detail="file field is required")

    language = form.get("language") or None
    prompt = form.get("prompt") or None
    response_format = form.get("response_format", "json")
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:8]

    suffix = Path(getattr(file, "filename", "audio.mp3")).suffix or ".mp3"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="whisper_")
    os.close(tmp_fd)

    gpu_id = None
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="empty audio file")

        with open(tmp_path, "wb") as f:
            f.write(content)

        filename = getattr(file, 'filename', '?')
        file_mb = len(content) / 1024 / 1024
        max_attempts = max(1, len(gpu_pool.workers))
        last_error = None
        result = None
        used_gpu_id = None

        for attempt in range(1, max_attempts + 1):
            # GPU 워커 획득 (빈 GPU가 없으면 대기)
            gpu_id, model = await gpu_pool.acquire()
            used_gpu_id = gpu_id
            print(
                f"[Whisper] 추론 시작[{request_id}]: {filename} "
                f"({file_mb:.1f} MB) lang={language} → GPU {gpu_id} "
                f"(attempt {attempt}/{max_attempts})"
            )

            try:
                # 동기 추론을 별도 스레드에서 실행 (이벤트 루프 블로킹 방지)
                result = await asyncio.to_thread(
                    _do_transcribe, model, tmp_path, language, prompt, filename, request_id, gpu_id
                )
                gpu_pool.release(gpu_id)
                gpu_id = None
                break
            except Exception as e:
                last_error = e
                print(f"[Whisper] 추론 실패[{request_id}]: GPU {gpu_id} | {type(e).__name__}: {e}")
                _clear_cuda_cache(gpu_id)
                gpu_pool.release(gpu_id)
                gpu_id = None

                if not _is_retryable_cuda_error(e) or attempt >= max_attempts:
                    raise

                await asyncio.sleep(0.2)

        if result is None:
            raise RuntimeError(f"Whisper 추론 실패: {last_error}")

        if response_format == "json":
            return JSONResponse({"text": result["text"]})

        # verbose_json (OpenAI SDK TranscriptionVerbose 호환)
        import json as _json
        payload = {
            "task": "transcribe",
            "language": result["detected_lang"],
            "duration": round(result["duration"], 3),
            "text": result["text"],
            "words": result["words"],
            "segments": result["segments"],
        }
        payload_size = len(_json.dumps(payload))
        print(f"[Whisper] 응답 전송[{request_id}]: {payload_size / 1024:.1f} KB (GPU {used_gpu_id})")
        return JSONResponse(payload)

    except HTTPException:
        raise
    except Exception as e:
        # 에러 시에도 GPU 반환
        if gpu_id is not None:
            gpu_pool.release(gpu_id)
        print(f"[Whisper] 요청 실패[{request_id}]: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Whisper inference failed: {e}") from e

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/gpu-status")
@app.get("/v1/gpu-status")
async def gpu_status():
    """GPU 워커 풀 상태 조회 — 클라이언트가 병렬 요청 수를 결정하는 데 사용."""
    return gpu_pool.status()


@app.get("/health")
async def health():
    status = gpu_pool.status()
    return {
        "status": "ok",
        "model": WHISPER_MODEL_SIZE,
        "compute_type": COMPUTE_TYPE,
        "engine": "faster-whisper",
        "gpus": status,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_whisper:app", host="0.0.0.0", port=8002)
