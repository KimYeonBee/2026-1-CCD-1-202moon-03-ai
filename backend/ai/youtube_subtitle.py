# -*- coding: utf-8 -*-
"""
YouTube 자막 모듈 — 자막 확인 → VTT 다운로드 → 파싱 → 단어 타임스탬프 보간

- youtube-transcript-api 우선 사용 (yt-dlp 봇 감지 우회)
- 실패 시 yt-dlp로 폴백
- VTT 파싱 후 단어별 타임스탬프 균등 분배 (선형 보간)
- 결과 형태: Whisper transcribe()와 동일한 딕셔너리 구조
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path

MAX_SEGMENT_CHARS = 55
MIN_SEGMENT_CHARS = 30


# ── 쿠키 파일 경로 ───────────────────────────────────────────────────────
COOKIES_PATH = "/app/cookies.txt"

def _get_cookie_args():
    """yt-dlp 쿠키 인자 반환 (파일 없으면 빈 리스트)"""
    if os.path.exists(COOKIES_PATH):
        return ["--cookies", COOKIES_PATH]
    return []


# ── youtube-transcript-api 라이브러리 (우선 사용) ─────────────────────────────
# yt-dlp보다 가벼운 요청으로 봇 감지에 걸릴 확률이 낮음

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    HAS_YT_TRANSCRIPT_API = True
except ImportError:
    HAS_YT_TRANSCRIPT_API = False
    print("[TADAC] youtube-transcript-api 미설치 → yt-dlp 전용 모드")


# ── YouTube URL에서 video_id 추출 ─────────────────────────────────────────────

def _extract_video_id(url):
    """YouTube URL에서 video ID 추출"""
    # https://youtu.be/VIDEO_ID
    m = re.search(r"youtu\.be/([^?&]+)", url)
    if m:
        return m.group(1)
    # https://www.youtube.com/watch?v=VIDEO_ID
    m = re.search(r"[?&]v=([^&]+)", url)
    if m:
        return m.group(1)
    # https://www.youtube.com/embed/VIDEO_ID
    m = re.search(r"/embed/([^?&]+)", url)
    if m:
        return m.group(1)
    return None


# ── yt-dlp 명령어 실행 헬퍼 ──────────────────────────────────────────────────

def _run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


# ══════════════════════════════════════════════════════════════════════════════
# youtube-transcript-api 변환 헬퍼
# ══════════════════════════════════════════════════════════════════════════════


def _convert_api_result_to_transcript(raw_data):
    """
    youtube-transcript-api의 raw_data를 Whisper 결과와 동일한 구조로 변환.
    
    raw_data 형태: [{'text': '...', 'start': 0.0, 'duration': 1.54}, ...]
    """
    segments = []
    all_words = []
    
    for i, entry in enumerate(raw_data):
        text = entry.get("text", "").strip()
        if not text:
            continue
        
        start = entry.get("start", 0.0)
        duration = entry.get("duration", 0.0)
        end = start + duration
        
        # HTML 태그 제거 (자동 자막에 포함될 수 있음)
        text = re.sub(r"<[^>]+>", "", text).strip()
        if not text:
            continue
        
        # 줄바꿈을 공백으로 치환
        text = text.replace("\n", " ").strip()
        
        # 단어 타임스탬프 보간
        words = _interpolate_words(text, start, end)
        all_words.extend(words)
        
        segments.append({
            "id": i,
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
        })
    
    # 중복 세그먼트 제거 (자동 자막에서 발생 가능)
    deduplicated = []
    for seg in segments:
        if deduplicated and deduplicated[-1]["text"] == seg["text"]:
            continue
        deduplicated.append(seg)
    
    # ID 재할당
    for i, seg in enumerate(deduplicated):
        seg["id"] = i

    deduplicated = _split_long_subtitle_segments(deduplicated)
    deduplicated = _merge_short_subtitle_segments(deduplicated)

    full_text = " ".join(seg["text"] for seg in deduplicated)

    return {
        "text": full_text,
        "words": all_words,
        "segments": deduplicated,
        "language": "ko",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 방법 2: yt-dlp (폴백)
# ══════════════════════════════════════════════════════════════════════════════

# ── 자막 존재 여부 확인 ───────────────────────────────────────────────────────
# yt-dlp --list-subs 결과를 파싱해서 수동/자동 자막 언어 목록 반환

def check_subtitles(youtube_url):
    print(f"[TADAC] 자막 확인 중: {youtube_url}")
    cmd = ["yt-dlp", "--no-playlist", "--list-subs", "--skip-download"]
    cmd.extend(_get_cookie_args())
    cmd.extend(["--extractor-args", "youtube:player_client=web,default"])
    cmd.append(youtube_url)
    rc, stdout, stderr = _run(cmd)

    manual_langs  = []  # 사람이 직접 작성한 자막
    auto_langs    = []  # YouTube가 자동 생성한 자막
    current_section = None  # 현재 파싱 중인 섹션: None / "manual" / "auto"

    for line in stdout.splitlines():
        # 섹션 헤더 감지
        if "Available automatic captions" in line:
            current_section = "auto"
            continue
        if "Available subtitles" in line:
            current_section = "manual"
            continue
        # [info] 줄이나 자막 없음 메시지 → 섹션 초기화
        if re.match(r"\[info\]", line) or "has no subtitles" in line:
            current_section = None
            continue
        if current_section is None:
            continue

        # 줄 시작의 언어 코드 파싱 (예: "ko", "en", "ko-orig")
        m = re.match(r"^\s*([\w-]+)\s+\S", line)
        if m:
            lang = m.group(1)
            if lang.lower() in ("language",):  # 테이블 헤더 행 건너뜀
                continue
            if current_section == "auto":
                auto_langs.append(lang)
            else:
                manual_langs.append(lang)

    result = {
        "has_manual":   len(manual_langs) > 0,
        "has_auto":     len(auto_langs)   > 0,
        "manual_langs": manual_langs,
        "auto_langs":   auto_langs,
    }
    print(f"[TADAC] 자막 확인 결과: 수동={result['has_manual']} 자동={result['has_auto']}")
    print(f"[TADAC] 수동 자막 언어: {manual_langs[:5]}")
    print(f"[TADAC] 자동 자막 언어: {auto_langs[:5]}")
    return result


# ── 선호 언어 선택 ────────────────────────────────────────────────────────────
# 정확 일치 → 접두사 일치 (ko-KR 등) → 첫 번째 항목 순서로 시도

def _pick_lang(langs, preferred="ko"):
    if preferred in langs:
        return preferred
    for lang in langs:
        if lang.startswith(preferred):  # "ko-KR" 같은 형태 처리
            return lang
    return langs[0] if langs else "ko"


# ── VTT 파일 다운로드 ─────────────────────────────────────────────────────────
# 수동 자막 우선, 없으면 자동 생성 자막 다운로드

def download_vtt(youtube_url, out_dir, sub_info, preferred_lang="ko"):
    if sub_info["has_manual"]:
        # 수동 자막 우선 사용 (정확도 높음)
        lang        = _pick_lang(sub_info["manual_langs"], preferred_lang)
        source_type = "youtube_manual"
        print(f"[TADAC] 수동 자막 다운로드: 언어={lang}")
        cmd = [
            "yt-dlp",
            "--js-runtimes", "node",
            "--no-playlist",
            "--write-subs",
            "--sub-lang",   lang,
            "--sub-format", "vtt",
            "--skip-download",
            "-o", str(Path(out_dir) / "subtitle"),
        ]
    else:
        # 자동 생성 자막 사용
        lang        = _pick_lang(sub_info["auto_langs"], preferred_lang)
        source_type = "youtube_auto"
        print(f"[TADAC] 자동 생성 자막 다운로드: 언어={lang}")
        cmd = [
            "yt-dlp",
            "--js-runtimes", "node",
            "--no-playlist",
            "--write-auto-subs",
            "--sub-lang",   lang,
            "--sub-format", "vtt",
            "--skip-download",
            "-o", str(Path(out_dir) / "subtitle"),
        ]

    cmd.extend(_get_cookie_args())
    cmd.extend(["--extractor-args", "youtube:player_client=web,default"])
        
    cmd.append(youtube_url)

    rc, stdout, stderr = _run(cmd)
    if rc != 0:
        raise RuntimeError(f"yt-dlp 자막 다운로드 실패:\n{stderr}")

    # 다운로드된 .vtt 파일 찾기
    vtt_files = list(Path(out_dir).glob("*.vtt"))
    if not vtt_files:
        raise FileNotFoundError(f"{out_dir} 에서 VTT 파일을 찾을 수 없음")

    return str(vtt_files[0]), source_type


# ── VTT 타임스탬프 파싱 ───────────────────────────────────────────────────────
# "00:01:23.456" 형태를 초 단위 float으로 변환

def _parse_timestamp(ts):
    ts    = ts.strip()
    parts = ts.split(":")

    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)

    return float(ts)


# ── 단어 타임스탬프 선형 보간 ─────────────────────────────────────────────────
# VTT는 구간(phrase) 단위 타임스탬프만 제공 → 단어별로 균등 분배
#
# 예) "도파민 시스템이 다르게 작동합니다" (1.5s ~ 4.2s, 4단어)
#     단어당 (4.2-1.5)/4 = 0.675s
#     "도파민" 1.500~2.175, "시스템이" 2.175~2.850, ...

def _split_long_subtitle_segments(segments, max_chars=MAX_SEGMENT_CHARS):
    """글자 수가 max_chars를 초과하는 자막 세그먼트를 단어 경계에서 분할."""
    if not segments:
        return segments

    result = []
    for seg in segments:
        if len(seg["text"]) <= max_chars:
            result.append(seg)
            continue

        words = seg["text"].split()
        if not words:
            result.append(seg)
            continue

        duration = seg["end"] - seg["start"]
        total_words = len(words)

        parts = []
        current_words = []
        for w in words:
            candidate = " ".join(current_words + [w]) if current_words else w
            if len(candidate) > max_chars and current_words:
                parts.append(current_words)
                current_words = [w]
            else:
                current_words.append(w)
        if current_words:
            parts.append(current_words)

        w_offset = 0
        for part_words in parts:
            part_text = " ".join(part_words)
            n = len(part_words)
            p_start = seg["start"] + duration * (w_offset / total_words)
            p_end = seg["start"] + duration * ((w_offset + n) / total_words)
            result.append({
                "id": 0,
                "start": round(p_start, 3),
                "end": round(p_end, 3),
                "text": part_text,
            })
            w_offset += n

    for i, seg in enumerate(result):
        seg["id"] = i

    if len(result) != len(segments):
        print(f"[TADAC] 긴 자막 세그먼트 분할: {len(segments)}개 → {len(result)}개 (max {max_chars}자)")

    return result


def _merge_short_subtitle_segments(segments, min_chars=MIN_SEGMENT_CHARS):
    """글자 수가 min_chars 미만인 자막 세그먼트를 인접 세그먼트와 병합."""
    if not segments:
        return segments

    result = []
    buf = None

    for seg in segments:
        if buf is None:
            buf = dict(seg)
            continue

        if len(buf["text"]) < min_chars:
            buf["end"] = seg["end"]
            buf["text"] = (buf["text"] + " " + seg["text"]).strip()
        else:
            result.append(buf)
            buf = dict(seg)

    if buf is not None:
        if len(buf["text"]) < min_chars and result:
            last = result[-1]
            last["end"] = buf["end"]
            last["text"] = (last["text"] + " " + buf["text"]).strip()
        else:
            result.append(buf)

    for i, seg in enumerate(result):
        seg["id"] = i

    return result


def _interpolate_words(phrase, start, end):
    words    = phrase.strip().split()
    if not words:
        return []

    duration      = end - start
    word_duration = duration / len(words)  # 단어당 할당 시간

    result = []
    for i, word in enumerate(words):
        w_start = round(start + i       * word_duration, 3)
        w_end   = round(start + (i + 1) * word_duration, 3)
        result.append({"word": word, "start": w_start, "end": w_end})

    return result


# ── VTT 파일 파싱 ─────────────────────────────────────────────────────────────
# VTT 파일 전체를 읽어 세그먼트 + 단어 목록 반환 (Whisper 결과와 같은 구조)

def parse_vtt(vtt_path):
    content = Path(vtt_path).read_text(encoding="utf-8")
    lines   = content.splitlines()

    segments  = []
    all_words = []
    seg_id    = 0
    i         = 0

    while i < len(lines):
        # 타임스탬프 줄 탐색: "00:00:00.000 --> 00:00:05.000"
        if "-->" not in lines[i]:
            i += 1
            continue

        ts_line = lines[i]
        # 위치 속성 제거 (align: / position: / line:)
        ts_line = ts_line.split(" align:")[0].split(" position:")[0].split(" line:")[0]
        parts   = ts_line.split("-->")

        if len(parts) != 2:
            i += 1
            continue

        try:
            start = _parse_timestamp(parts[0])
            end   = _parse_timestamp(parts[1])
        except ValueError:
            i += 1
            continue

        # 타임스탬프 다음 줄부터 빈 줄까지 텍스트 수집
        i += 1
        text_lines = []
        while i < len(lines) and lines[i].strip():
            # VTT 인라인 태그 제거 예: <00:00:01.000><c>
            clean = re.sub(r"<[^>]+>", "", lines[i]).strip()
            if clean:
                text_lines.append(clean)
            i += 1

        if not text_lines:
            continue

        # YouTube 자동 자막 중복 줄 제거 (이전 줄이 반복되는 경우)
        unique_lines = [text_lines[0]]
        for tl in text_lines[1:]:
            if tl != unique_lines[-1]:
                unique_lines.append(tl)
        phrase = " ".join(unique_lines)

        # 이전 세그먼트와 텍스트가 완전히 같으면 건너뜀 (자동 자막 중복 블록)
        if segments and segments[-1]["text"] == phrase:
            continue

        # 단어 타임스탬프 보간 후 전체 목록에 추가
        words = _interpolate_words(phrase, start, end)
        all_words.extend(words)

        segments.append({
            "id":    seg_id,
            "start": start,
            "end":   end,
            "text":  phrase,
        })
        seg_id += 1

    print(f"[TADAC] VTT 파싱 완료: {len(segments)}개 세그먼트, {len(all_words)}개 단어")

    segments = _split_long_subtitle_segments(segments)
    segments = _merge_short_subtitle_segments(segments)

    return {
        "text":     " ".join(seg["text"] for seg in segments),
        "words":    all_words,
        "segments": segments,
        "language": "ko",
    }



# ══════════════════════════════════════════════════════════════════════════════
# 메인 함수 — youtube-transcript-api 우선 → yt-dlp 폴백
# ══════════════════════════════════════════════════════════════════════════════

def get_transcript_from_youtube(youtube_url, preferred_lang="ko"):
    """
    YouTube 수동 자막 추출 메인 함수.
    
    ⚠ 자동 자막은 품질이 떨어지므로 직접 사용하지 않음.
    자동 자막만 있거나 자막이 없으면 빈 transcript를 반환 → pipeline에서 Whisper STT로 전환.
    
    우선순위:
        1. youtube-transcript-api로 수동 자막 (봇 감지 우회에 유리)
        2. yt-dlp로 수동 자막 (폴백)
        3. 수동 자막 없으면 → 빈 결과 반환 (pipeline이 오디오 추출 → Whisper)
    
    Returns:
        (transcript_dict, source_type)
        - transcript_dict: Whisper 결과와 동일한 구조 (수동 자막만)
        - source_type: "youtube_manual" / "youtube_auto" / "no_subtitle"
          ※ "youtube_auto"/"no_subtitle" 일 때 transcript_dict는 빈 딕셔너리
    """
    
    # ── 1차 시도: youtube-transcript-api (수동 자막만) ─────────────────────────
    print("[TADAC] 자막 추출 1차 시도: youtube-transcript-api")
    transcript, source_type = _get_manual_transcript_via_api(youtube_url, preferred_lang)
    
    if transcript and transcript.get("segments"):
        print(f"[TADAC] ✅ youtube-transcript-api 수동 자막 성공")
        return transcript, "youtube_manual"
    
    # ── 2차 시도: yt-dlp (수동 자막만) ────────────────────────────────────────
    print("[TADAC] youtube-transcript-api 수동 자막 없음 → 2차 시도: yt-dlp")
    try:
        transcript, source_type = _get_manual_transcript_via_ytdlp(youtube_url, preferred_lang)
        if transcript and transcript.get("segments"):
            print(f"[TADAC] ✅ yt-dlp 수동 자막 성공")
            return transcript, "youtube_manual"
    except Exception as e:
        print(f"[TADAC] yt-dlp 수동 자막도 실패: {e}")
    
    # ── 수동 자막 없음 → pipeline이 오디오 추출 → Whisper STT ────────────────
    print("[TADAC] ❌ 수동 자막 없음 → 오디오 추출 후 Whisper STT 필요")
    return {}, "no_subtitle"


def _get_manual_transcript_via_api(youtube_url, preferred_lang="ko"):
    """youtube-transcript-api로 수동 자막만 가져오기"""
    if not HAS_YT_TRANSCRIPT_API:
        return None, None
    
    video_id = _extract_video_id(youtube_url)
    if not video_id:
        return None, None
    
    try:
        # 쿠키 파일이 있으면 인증에 사용
        cookie_path = COOKIES_PATH if os.path.exists(COOKIES_PATH) else None
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)
        
        # 수동 자막만 시도
        try:
            transcript = transcript_list.find_manually_created_transcript([preferred_lang, 'ko', 'en'])
            fetched = transcript.fetch()
            raw_data = fetched.to_raw_data()
            transcript_dict = _convert_api_result_to_transcript(raw_data)
            print(f"[TADAC] youtube-transcript-api: 수동 자막 발견 (언어={transcript.language_code})")
            return transcript_dict, "youtube_manual"
        except Exception:
            print("[TADAC] youtube-transcript-api: 수동 자막 없음")
            return None, None
            
    except Exception as e:
        print(f"[TADAC] youtube-transcript-api 실패: {e}")
        return None, None


def _get_manual_transcript_via_ytdlp(youtube_url, preferred_lang="ko"):
    """yt-dlp로 수동 자막만 가져오기"""
    try:
        sub_info = check_subtitles(youtube_url)
    except Exception as e:
        print(f"[TADAC] yt-dlp 자막 확인 실패: {e}")
        return {}, "no_subtitle"

    # 수동 자막이 없으면 빈 결과 반환
    if not sub_info["has_manual"]:
        print("[TADAC] yt-dlp: 수동 자막 없음")
        return {}, "no_subtitle"

    # 수동 자막만 다운로드
    with tempfile.TemporaryDirectory(prefix="tadac_vtt_") as tmp_dir:
        vtt_path, source_type = download_vtt(youtube_url, tmp_dir, sub_info, preferred_lang)
        transcript = parse_vtt(vtt_path)

    return transcript, source_type

