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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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


# ── 파일 업로드 처리 ──────────────────────────────────────────────────────────
# 비디오 파일을 받아서 game_data JSON 반환

@app.post("/api/process")
async def process_file(
    file:       UploadFile = File(...),
    language:   str        = Form("ko"),
    stt_prompt: str        = Form(None),
    refine:     bool       = Form(True),
    shorts:     bool       = Form(False),
):
    # 파일 확장자 검사
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식: '{suffix}'. 허용: {ALLOWED_EXTENSIONS}",
        )

    # 업로드 파일을 임시 폴더에 저장
    tmp_dir = tempfile.mkdtemp(prefix="tadac_upload_")
    try:
        file_path = os.path.join(tmp_dir, f"input{suffix}")
        content   = await file.read()

        with open(file_path, "wb") as f:
            f.write(content)

        print(f"[TADAC] API /process: {file.filename} ({len(content)/1024/1024:.1f} MB)")

        # 파이프라인 실행 — 난이도 파라미터는 내부 고정값 사용
        game_data = pipeline_module.run_pipeline(
            source              = file_path,
            language            = language,
            blanks_per_sentence = MAX_BLANKS_PER_SENTENCE,
            fall_speed          = BASE_FALL_SPEED,
            lead_time           = BASE_LEAD_TIME,
            stt_prompt          = stt_prompt,
            refine              = refine,
            generate_shorts     = shorts,
        )
        return game_data

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)  # 임시 파일 정리


# ── YouTube URL 처리 ──────────────────────────────────────────────────────────
# YouTube URL을 받아서 game_data JSON 반환

@app.post("/api/process-url")
async def process_url(req: UrlRequest):
    # YouTube URL 유효성 검사
    if not req.url.startswith(("https://www.youtube.com", "https://youtu.be", "http://www.youtube.com")):
        raise HTTPException(status_code=400, detail="YouTube URL만 지원합니다")

    print(f"[TADAC] API /process-url: {req.url}")

    try:
        # 파이프라인 실행 — 난이도 파라미터는 내부 고정값 사용
        game_data = pipeline_module.run_pipeline(
            source              = req.url,
            language            = req.language,
            blanks_per_sentence = MAX_BLANKS_PER_SENTENCE,
            fall_speed          = BASE_FALL_SPEED,
            lead_time           = BASE_LEAD_TIME,
            stt_prompt          = req.stt_prompt,
            refine              = req.refine,
            generate_shorts     = req.shorts,
        )
        return game_data

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── YouTube URL 스트리밍 처리 (SSE) ───────────────────────────────────────────
# 챕터 단위로 처리 완료 즉시 SSE 이벤트로 전달

from fastapi.responses import StreamingResponse

@app.post("/api/process-url/stream")
async def process_url_stream(req: UrlRequest):
    """
    챕터별 스트리밍 처리 — SSE (Server-Sent Events)

    이벤트 흐름:
        1. init:          챕터 목록 + 전체 정보
        2. chapter_ready: 챕터별 게임 데이터 (빈칸 자막 + 퀴즈)
        3. complete:      처리 완료 + 통계
    """
    import json

    # YouTube URL 유효성 검사
    if not req.url.startswith(("https://www.youtube.com", "https://youtu.be", "http://www.youtube.com")):
        raise HTTPException(status_code=400, detail="YouTube URL만 지원합니다")

    print(f"[TADAC] API /process-url/stream: {req.url}")

    def event_generator():
        try:
            for chunk in pipeline_module.run_pipeline_streaming(
                source              = req.url,
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

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection":    "keep-alive",
            "X-Accel-Buffering": "no",  # nginx 버퍼링 방지
        },
    )


# ── 파일 업로드 스트리밍 처리 (SSE) ───────────────────────────────────────────
# 비디오 파일을 받아서 챕터 단위로 처리 완료 즉시 SSE 이벤트로 전달

@app.post("/api/process/stream")
async def process_file_stream(
    file:       UploadFile = File(...),
    language:   str        = Form("ko"),
    stt_prompt: str        = Form(None),
    refine:     bool       = Form(True),
    shorts:     bool       = Form(False),
):
    """
    챕터별 스트리밍 처리 — SSE (Server-Sent Events)

    이벤트 흐름:
        1. init:          챕터 목록 + 전체 정보
        2. chapter_ready: 챕터별 게임 데이터 (빈칸 자막 + 퀴즈)
        3. complete:      처리 완료 + 통계
    """
    import json

    # 파일 확장자 검사
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식: '{suffix}'. 허용: {ALLOWED_EXTENSIONS}",
        )

    # 업로드 파일을 임시 폴더에 저장 (SSE generator가 끝날 때 정리)
    tmp_dir   = tempfile.mkdtemp(prefix="tadac_upload_")
    file_path = os.path.join(tmp_dir, f"input{suffix}")
    content   = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    print(f"[TADAC] API /process/stream: {file.filename} ({len(content)/1024/1024:.1f} MB)")

    def event_generator():
        try:
            for chunk in pipeline_module.run_pipeline_streaming(
                source              = file_path,
                language            = language,
                blanks_per_sentence = MAX_BLANKS_PER_SENTENCE,
                fall_speed          = BASE_FALL_SPEED,
                lead_time           = BASE_LEAD_TIME,
                stt_prompt          = stt_prompt,
                refine              = refine,
                generate_shorts     = shorts,
            ):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as e:
            error_event = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)  # 업로드 임시 파일 정리

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
