# -*- coding: utf-8 -*-
"""
TADAC FastAPI 서버 — AI 파이프라인 HTTP 엔드포인트

엔드포인트:
    POST /api/process      — 파일 업로드 → game_data JSON 반환
    POST /api/process-url  — YouTube URL → game_data JSON 반환
    GET  /api/health       — 서버 상태 확인

실행:
    python api.py   →   http://localhost:8000

난이도 파라미터 (blanks_per_sentence, fall_speed, lead_time)는
프론트엔드가 실시간으로 관리하므로 API에서 받지 않음.
AI는 항상 세그먼트당 최대 빈칸 2개로 생성, 프론트가 몇 개 보여줄지 결정.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 같은 폴더의 파이프라인 모듈 임포트
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pipeline_module

# FastAPI 앱 초기화
app = FastAPI(title="TADAC AI Pipeline API", version="1.0.0")

# CORS 설정 — React 개발 서버 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # CRA 기본 포트
        "http://localhost:5173",   # Vite 기본 포트
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 업로드 허용 확장자 — 백엔드 API는 일단 영상 파일만 받음
ALLOWED_EXTENSIONS = {".mp4", ".webm"}

# AI 내부 고정값 — 난이도는 프론트엔드가 관리
MAX_BLANKS_PER_SENTENCE = 2   # 세그먼트당 최대 빈칸 수
BASE_FALL_SPEED         = 1.0  # 기준값. 프론트가 target_time 기반으로 재계산
BASE_LEAD_TIME          = 3.0  # 기준값. 프론트가 target_time 기반으로 재계산

YOUTUBE_PREFIXES = ("https://www.youtube.com", "https://youtu.be", "http://www.youtube.com")


def _is_youtube_url(url: str) -> bool:
    return url.startswith(YOUTUBE_PREFIXES)


def _download_url_to_file(url: str, tmp_dir: str) -> str:
    """HTTP(S) URL에서 파일을 다운로드하여 로컬 임시 경로를 반환한다."""
    parsed = urlparse(url)
    filename = Path(parsed.path).name or "download.mp4"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        suffix = ".mp4"
    dest = os.path.join(tmp_dir, f"input{suffix}")

    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=8192):
                f.write(chunk)

    return dest


# ── 요청 바디 모델 ────────────────────────────────────────────────────────────
# POST /api/process-url 에서 사용
# 콘텐츠 처리 관련 파라미터만 받음 (난이도 파라미터 제외)

class UrlRequest(BaseModel):
    url:        str
    language:   str  = "ko"
    stt_prompt: str  = None
    refine:     bool = True
    shorts:     bool = False


# ── 서버 상태 확인 ────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    api_key_set = bool(os.getenv("OPENAI_API_KEY"))
    return {
        "status":            "ok",
        "api_key_configured": api_key_set,
    }


# ── S3 URL로 파일 처리 ────────────────────────────────────────────────────────
# S3 URL을 받아서 다운로드 후 game_data JSON 반환

@app.post("/api/process")
async def process_file(req: UrlRequest):
    if not req.url.startswith("http"):
        raise HTTPException(status_code=400, detail="유효한 URL이 아닙니다")

    print(f"[TADAC] API /process: {req.url}")

    tmp_dir = tempfile.mkdtemp(prefix="tadac_url_download_")
    try:
        source = _download_url_to_file(req.url, tmp_dir)

        game_data = pipeline_module.run_pipeline(
            source              = source,
            language            = req.language,
            blanks_per_sentence = MAX_BLANKS_PER_SENTENCE,
            fall_speed          = BASE_FALL_SPEED,
            lead_time           = BASE_LEAD_TIME,
            stt_prompt          = req.stt_prompt,
            refine              = req.refine,
            generate_shorts     = req.shorts,
        )
        return game_data

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=400, detail=f"URL 다운로드 실패: {e.response.status_code}")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── YouTube URL 처리 ──────────────────────────────────────────────────────────
# YouTube URL을 받아서 game_data JSON 반환

@app.post("/api/process-url")
async def process_url(req: UrlRequest):
    if not req.url.startswith("http"):
        raise HTTPException(status_code=400, detail="유효한 URL이 아닙니다")

    is_youtube = _is_youtube_url(req.url)
    print(f"[TADAC] API /process-url: {req.url} ({'YouTube' if is_youtube else 'Direct URL'})")

    tmp_dir = None
    try:
        if is_youtube:
            source = req.url
        else:
            tmp_dir = tempfile.mkdtemp(prefix="tadac_url_download_")
            source = _download_url_to_file(req.url, tmp_dir)

        game_data = pipeline_module.run_pipeline(
            source              = source,
            language            = req.language,
            blanks_per_sentence = MAX_BLANKS_PER_SENTENCE,
            fall_speed          = BASE_FALL_SPEED,
            lead_time           = BASE_LEAD_TIME,
            stt_prompt          = req.stt_prompt,
            refine              = req.refine,
            generate_shorts     = req.shorts,
        )
        return game_data

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=400, detail=f"URL 다운로드 실패: {e.response.status_code}")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ── YouTube URL 스트리밍 처리 (SSE) ───────────────────────────────────────────
# 챕터 단위로 처리 완료 즉시 SSE 이벤트로 전달

from fastapi.responses import StreamingResponse

@app.post("/api/process-url/stream")
async def process_url_stream(req: UrlRequest):
    """
    챕터별 스트리밍 처리 — SSE (Server-Sent Events)

    YouTube URL 또는 S3 등 일반 HTTPS URL 모두 처리.
    - YouTube URL → 파이프라인에 URL 그대로 전달
    - 일반 URL (S3 등) → 파일 다운로드 후 로컬 경로로 파이프라인 전달

    이벤트 흐름:
        1. init:          챕터 목록 + 전체 정보
        2. chapter_ready: 챕터별 게임 데이터 (빈칸 자막 + 퀴즈)
        3. complete:      처리 완료 + 통계
    """
    import json

    if not req.url.startswith("http"):
        raise HTTPException(status_code=400, detail="유효한 URL이 아닙니다")

    is_youtube = _is_youtube_url(req.url)
    print(f"[TADAC] API /process-url/stream: {req.url} ({'YouTube' if is_youtube else 'Direct URL'})")

    if is_youtube:
        source = req.url
        tmp_dir = None
    else:
        tmp_dir = tempfile.mkdtemp(prefix="tadac_url_download_")
        try:
            source = _download_url_to_file(req.url, tmp_dir)
        except httpx.HTTPStatusError as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=f"URL 다운로드 실패: {e.response.status_code}")
        except Exception as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"URL 다운로드 중 오류: {e}")

    def event_generator():
        try:
            for chunk in pipeline_module.run_pipeline_streaming(
                source              = source,
                language            = req.language,
                blanks_per_sentence = MAX_BLANKS_PER_SENTENCE,
                fall_speed          = BASE_FALL_SPEED,
                lead_time           = BASE_LEAD_TIME,
                stt_prompt          = req.stt_prompt,
                refine              = req.refine,
                generate_shorts     = req.shorts,
            ):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as e:
            error_event = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection":    "keep-alive",
            "X-Accel-Buffering": "no",  # nginx 버퍼링 방지
        },
    )


# ── S3 URL 스트리밍 처리 (SSE) ────────────────────────────────────────────────
# S3 URL을 받아서 다운로드 후 챕터 단위로 SSE 이벤트 전달

@app.post("/api/process/stream")
async def process_file_stream(req: UrlRequest):
    """
    S3 URL → 파일 다운로드 → 챕터별 스트리밍 처리 — SSE (Server-Sent Events)

    이벤트 흐름:
        1. init:          챕터 목록 + 전체 정보
        2. chapter_ready: 챕터별 게임 데이터 (빈칸 자막 + 퀴즈)
        3. complete:      처리 완료 + 통계
    """
    import json

    if not req.url.startswith("http"):
        raise HTTPException(status_code=400, detail="유효한 URL이 아닙니다")

    print(f"[TADAC] API /process/stream: {req.url}")

    tmp_dir = tempfile.mkdtemp(prefix="tadac_url_download_")
    try:
        source = _download_url_to_file(req.url, tmp_dir)
    except httpx.HTTPStatusError as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"URL 다운로드 실패: {e.response.status_code}")
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"URL 다운로드 중 오류: {e}")

    def event_generator():
        try:
            for chunk in pipeline_module.run_pipeline_streaming(
                source              = source,
                language            = req.language,
                blanks_per_sentence = MAX_BLANKS_PER_SENTENCE,
                fall_speed          = BASE_FALL_SPEED,
                lead_time           = BASE_LEAD_TIME,
                stt_prompt          = req.stt_prompt,
                refine              = req.refine,
                generate_shorts     = req.shorts,
            ):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as e:
            error_event = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection":    "keep-alive",
            "X-Accel-Buffering": "no",  # nginx 버퍼링 방지
        },
    )


# ── 직접 실행 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
