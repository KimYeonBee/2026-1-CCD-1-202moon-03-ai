# -*- coding: utf-8 -*-
"""
YouTube 오디오 추출 모듈 — yt-dlp + pytubefix 이중 폴백

- 자막이 없는 YouTube 영상에서 오디오만 뽑아 Whisper STT로 넘기기 위해 사용
- yt-dlp 우선 시도 → 봇 감지 실패 시 pytubefix로 폴백
"""

import os
import subprocess
import tempfile
from pathlib import Path


# ── pytubefix 라이브러리 (폴백용) ─────────────────────────────────────────────
try:
    from pytubefix import YouTube as PytubeYouTube
    HAS_PYTUBEFIX = True
except ImportError:
    HAS_PYTUBEFIX = False
    print("[TADAC] pytubefix 미설치 → yt-dlp 전용 모드")


# ── yt-dlp 명령어 실행 헬퍼 ──────────────────────────────────────────────────

def _run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


# ══════════════════════════════════════════════════════════════════════════════
# 방법 1: yt-dlp (우선 시도)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_audio_ytdlp(youtube_url, out_dir):
    """yt-dlp로 오디오 추출"""
    out_template = str(Path(out_dir) / "audio.%(ext)s")

    cmd = [
        "yt-dlp",
        "--js-runtimes", "node",
        "--no-playlist",
        "-x",                          # 오디오만 추출
        "--audio-format", "mp3",       # mp3로 변환
        "--audio-quality", "0",        # 최고 품질
        "-o", out_template,
    ]

    cmd.extend(["--extractor-args", "youtube:player_client=android"])
    cmd.append(youtube_url)

    print(f"[TADAC] [yt-dlp] YouTube 오디오 추출 중: {youtube_url}")
    rc, stdout, stderr = _run(cmd)

    if rc != 0:
        raise RuntimeError(f"yt-dlp 오디오 추출 실패:\n{stderr}")

    # 저장된 오디오 파일 찾기
    audio_files = list(Path(out_dir).glob("audio.*"))
    if not audio_files:
        raise FileNotFoundError(f"{out_dir} 에서 오디오 파일을 찾을 수 없음")

    audio_path = str(audio_files[0])
    print(f"[TADAC] [yt-dlp] 오디오 추출 완료: {audio_path}")
    return audio_path


# ══════════════════════════════════════════════════════════════════════════════
# 방법 2: pytubefix (폴백)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_audio_pytubefix(youtube_url, out_dir):
    """pytubefix로 오디오 추출 (yt-dlp 폴백)"""
    if not HAS_PYTUBEFIX:
        raise RuntimeError("pytubefix 미설치")

    print(f"[TADAC] [pytubefix] YouTube 오디오 추출 중: {youtube_url}")

    try:
        yt = PytubeYouTube(youtube_url, use_po_token=True)
    except Exception:
        # use_po_token 실패 시 일반 모드로 재시도
        try:
            yt = PytubeYouTube(youtube_url)
        except Exception as e:
            raise RuntimeError(f"pytubefix YouTube 객체 생성 실패: {e}")

    # 오디오 전용 스트림 가져오기
    audio_stream = yt.streams.get_audio_only()
    if not audio_stream:
        # 오디오 전용 없으면 가장 낮은 해상도 비디오라도 가져오기
        audio_stream = yt.streams.filter(only_audio=True).first()

    if not audio_stream:
        raise RuntimeError("pytubefix: 사용 가능한 오디오 스트림 없음")

    # 다운로드 (m4a 또는 webm 형태로 저장됨)
    downloaded_path = audio_stream.download(
        output_path=out_dir,
        filename="audio_pytubefix"
    )
    print(f"[TADAC] [pytubefix] 다운로드 완료: {downloaded_path}")

    # mp3로 변환 (ffmpeg 사용)
    mp3_path = str(Path(out_dir) / "audio.mp3")
    ret = os.system(f'ffmpeg -y -i "{downloaded_path}" -vn -acodec mp3 -q:a 0 "{mp3_path}" -loglevel quiet')

    if ret != 0 or not os.path.exists(mp3_path):
        # ffmpeg 변환 실패 시 원본 그대로 사용
        print(f"[TADAC] [pytubefix] mp3 변환 실패 → 원본 파일 사용")
        return downloaded_path

    # 원본 파일 삭제
    try:
        os.remove(downloaded_path)
    except Exception:
        pass

    print(f"[TADAC] [pytubefix] mp3 변환 완료: {mp3_path}")
    return mp3_path


# ══════════════════════════════════════════════════════════════════════════════
# 메인 오디오 추출 함수 — yt-dlp 우선 → pytubefix 폴백
# ══════════════════════════════════════════════════════════════════════════════

def extract_audio(youtube_url, out_dir):
    """
    YouTube 오디오 추출 (이중 폴백).
    
    우선순위:
        1. yt-dlp (가장 안정적)
        2. pytubefix (yt-dlp 봇 감지 시 폴백)
    """
    errors = []

    # ── 1차 시도: yt-dlp ──────────────────────────────────────────────────────
    try:
        return _extract_audio_ytdlp(youtube_url, out_dir)
    except Exception as e:
        print(f"[TADAC] yt-dlp 오디오 추출 실패: {e}")
        errors.append(f"yt-dlp: {e}")

    # ── 2차 시도: pytubefix ───────────────────────────────────────────────────
    if HAS_PYTUBEFIX:
        try:
            return _extract_audio_pytubefix(youtube_url, out_dir)
        except Exception as e:
            print(f"[TADAC] pytubefix 오디오 추출도 실패: {e}")
            errors.append(f"pytubefix: {e}")
    else:
        errors.append("pytubefix: 미설치")

    # ── 모두 실패 ────────────────────────────────────────────────────────────
    error_detail = "\n".join(errors)
    raise RuntimeError(
        f"YouTube 오디오 추출 실패 (모든 방법 시도 완료):\n{error_detail}\n\n"
        f"💡 해결 방법:\n"
        f"1. pip install --upgrade yt-dlp pytubefix\n"
        f"2. 서버 IP가 YouTube에 의해 차단된 경우 프록시 사용 필요"
    )


# ── 임시 폴더와 함께 추출 ────────────────────────────────────────────────────
# 호출자가 직접 tmp_dir을 관리하고 싶을 때 사용

def extract_audio_temp(youtube_url):
    tmp_dir    = tempfile.mkdtemp(prefix="tadac_audio_")
    audio_path = extract_audio(youtube_url, tmp_dir)
    return audio_path, tmp_dir  # 사용 후 tmp_dir 직접 삭제 필요
