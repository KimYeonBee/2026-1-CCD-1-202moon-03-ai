# -*- coding: utf-8 -*-
"""
TADAC 유튜브 오디오 추출 API 서버 (집 노트북 배포용)

주거용 IP를 활용하여 yt-dlp로 YouTube 음성을 추출한 뒤
EC2 오케스트레이터에 오디오 파일을 반환한다.

실행:
    uvicorn app_extractor:app --host 0.0.0.0 --port 8001
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

app = FastAPI(title="TADAC YouTube Audio Extractor", version="1.0.0")

YOUTUBE_PREFIXES = (
    "https://www.youtube.com",
    "https://youtu.be",
    "http://www.youtube.com",
)

COOKIES_PATH = os.getenv("COOKIES_PATH", "./cookies.txt")


class ExtractRequest(BaseModel):
    url: str


def _get_cookie_args():
    if os.path.exists(COOKIES_PATH):
        print(f"[Extractor] 쿠키 파일 사용: {COOKIES_PATH}")
        return ["--cookies", COOKIES_PATH]
    return []


@app.post("/api/extract")
async def extract_audio(req: ExtractRequest):
    if not any(req.url.startswith(p) for p in YOUTUBE_PREFIXES):
        raise HTTPException(status_code=400, detail="YouTube URL만 지원합니다")

    tmp_dir = tempfile.mkdtemp(prefix="tadac_extract_")

    try:
        out_template = str(Path(tmp_dir) / "audio.%(ext)s")
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "-f", "bestaudio[ext=m4a]/bestaudio/18",
            "-o", out_template,
            "--extractor-args", "youtube:player_client=android",
        ]
        cmd.append(req.url)

        print(f"[Extractor] 오디오 추출 시작: {req.url}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"yt-dlp 실패: {result.stderr[:500]}",
            )

        audio_files = list(Path(tmp_dir).glob("audio.*"))
        if not audio_files:
            raise HTTPException(status_code=500, detail="추출된 오디오 파일 없음")

        audio_path = str(audio_files[0])
        file_size = os.path.getsize(audio_path)
        print(f"[Extractor] 추출 완료: {audio_path} ({file_size / 1024 / 1024:.1f} MB)")

        ext = Path(audio_path).suffix
        media_type = "audio/mp4" if ext == ".m4a" else "audio/mpeg"
        return FileResponse(
            path=audio_path,
            media_type=media_type,
            filename=f"audio{ext}",
            background=BackgroundTask(shutil.rmtree, tmp_dir, True),
        )

    except HTTPException:
        raise
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=504, detail="yt-dlp 타임아웃 (10분 초과)")
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "cookies": os.path.exists(COOKIES_PATH)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_extractor:app", host="0.0.0.0", port=8001, reload=True)
