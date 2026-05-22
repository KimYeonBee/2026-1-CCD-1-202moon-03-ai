# -*- coding: utf-8 -*-
"""
YouTube 오디오 추출 모듈 — 집 노트북 추출 서버 API 호출

- 데이터센터 IP 차단(유튜브) 문제를 우회하기 위해
  주거용 IP를 가진 집 노트북의 추출 서버를 HTTP로 호출
- 추출된 오디오 파일을 스트리밍으로 수신하여 로컬 디스크에 저장
"""

import os
import tempfile
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── 추출 서버 설정 ──────────────────────────────────────────────────────────────
EXTRACTOR_API_URL = os.getenv("EXTRACTOR_API_URL", "http://localhost:8001")

MAX_RETRIES    = 3
RETRY_BASE_SEC = 5


def extract_audio(youtube_url, out_dir):
    """
    집 노트북 추출 서버를 호출하여 YouTube 오디오를 추출.

    Args:
        youtube_url: YouTube URL
        out_dir: 오디오 파일 저장 디렉토리

    Returns:
        str: 저장된 오디오 파일 경로
    """
    url = f"{EXTRACTOR_API_URL}/api/extract"
    audio_path = os.path.join(out_dir, "audio.mp3")

    for attempt in range(MAX_RETRIES):
        try:
            print(f"[TADAC] 추출 서버 호출: {youtube_url}")
            with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
                with client.stream("POST", url, json={"url": youtube_url}) as response:
                    response.raise_for_status()
                    with open(audio_path, "wb") as f:
                        for chunk in response.iter_bytes(chunk_size=8192):
                            f.write(chunk)

            file_size = os.path.getsize(audio_path)
            print(f"[TADAC] 오디오 수신 완료: {audio_path} ({file_size / 1024 / 1024:.1f} MB)")
            return audio_path

        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_SEC * (2 ** attempt)
                print(f"[TADAC] 추출 서버 연결 실패: {e} → {delay}초 후 재시도 ({attempt + 1}/{MAX_RETRIES})")
                time.sleep(delay)
            else:
                raise RuntimeError(
                    f"추출 서버 연결 실패 (재시도 소진): {e}\n"
                    f"서버 주소: {EXTRACTOR_API_URL}\n"
                    f"집 노트북 추출 서버가 실행 중인지 확인하세요."
                ) from e

        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"추출 서버 오류 (HTTP {e.response.status_code}): "
                f"{e.response.text[:300]}"
            ) from e

        except httpx.ReadTimeout:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_SEC * (2 ** attempt)
                print(f"[TADAC] 추출 서버 응답 타임아웃 → {delay}초 후 재시도 ({attempt + 1}/{MAX_RETRIES})")
                time.sleep(delay)
            else:
                raise RuntimeError("추출 서버 응답 타임아웃 (재시도 소진)")


def extract_audio_temp(youtube_url):
    """임시 폴더에 오디오를 추출. 사용 후 tmp_dir 삭제 필요."""
    tmp_dir    = tempfile.mkdtemp(prefix="tadac_audio_")
    audio_path = extract_audio(youtube_url, tmp_dir)
    return audio_path, tmp_dir
