# -*- coding: utf-8 -*-
"""
YouTube 오디오 추출 모듈 — yt-dlp로 영상에서 mp3 추출

- 자막이 없는 YouTube 영상에서 오디오만 뽑아 Whisper STT로 넘기기 위해 사용
"""

import os
import subprocess
import tempfile
from pathlib import Path


# ── yt-dlp 명령어 실행 헬퍼 ──────────────────────────────────────────────────

def _run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


# ── 오디오 추출 ───────────────────────────────────────────────────────────────
# yt-dlp -x 옵션으로 영상에서 오디오만 mp3로 추출

def extract_audio(youtube_url, out_dir):
    out_template = str(Path(out_dir) / "audio.%(ext)s")  # 저장 경로 템플릿

    cmd = [
        "yt-dlp",
        "--js-runtimes", "node",
        "--no-playlist",
        "-x",                          # 오디오만 추출
        "--audio-format", "mp3",       # mp3로 변환
        "--audio-quality", "0",        # 최고 품질
        "-o", out_template,
    ]

    cookie_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cookies.txt")
    if os.path.exists(cookie_path):
        cmd.extend(["--cookies", cookie_path])
        
    cmd.append(youtube_url)

    print(f"[TADAC] YouTube 오디오 추출 중: {youtube_url}")
    rc, stdout, stderr = _run(cmd)

    if rc != 0:
        raise RuntimeError(f"yt-dlp 오디오 추출 실패:\n{stderr}")

    # 저장된 오디오 파일 찾기
    audio_files = list(Path(out_dir).glob("audio.*"))
    if not audio_files:
        raise FileNotFoundError(f"{out_dir} 에서 오디오 파일을 찾을 수 없음")

    audio_path = str(audio_files[0])
    print(f"[TADAC] 오디오 추출 완료: {audio_path}")
    return audio_path


# ── 임시 폴더와 함께 추출 ────────────────────────────────────────────────────
# 호출자가 직접 tmp_dir을 관리하고 싶을 때 사용

def extract_audio_temp(youtube_url):
    tmp_dir    = tempfile.mkdtemp(prefix="tadac_audio_")
    audio_path = extract_audio(youtube_url, tmp_dir)
    return audio_path, tmp_dir  # 사용 후 tmp_dir 직접 삭제 필요
