# -*- coding: utf-8 -*-
"""
YouTube 자막 모듈 — 자막 확인 → VTT 다운로드 → 파싱 → 단어 타임스탬프 보간

- yt-dlp로 수동 자막 / 자동 생성 자막 확인 및 다운로드
- VTT 파싱 후 단어별 타임스탬프 균등 분배 (선형 보간)
- 결과 형태: Whisper transcribe()와 동일한 딕셔너리 구조
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path


# ── yt-dlp 명령어 실행 헬퍼 ──────────────────────────────────────────────────

def _run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


# ── 자막 존재 여부 확인 ───────────────────────────────────────────────────────
# yt-dlp --list-subs 결과를 파싱해서 수동/자동 자막 언어 목록 반환

def check_subtitles(youtube_url):
    print(f"[TADAC] 자막 확인 중: {youtube_url}")
    cmd = ["yt-dlp", "--js-runtimes", "node", "--no-playlist", "--list-subs", "--skip-download"]
    cmd.extend(["--extractor-args", "youtube:player_client=android"])
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

    cmd.extend(["--extractor-args", "youtube:player_client=android"])
        
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

    return {
        "text":     " ".join(seg["text"] for seg in segments),
        "words":    all_words,
        "segments": segments,
        "language": "ko",
    }


# ── 메인 함수 ─────────────────────────────────────────────────────────────────
# 자막 확인 → 다운로드 → 파싱 한 번에 처리

def get_transcript_from_youtube(youtube_url, preferred_lang="ko"):
    sub_info = check_subtitles(youtube_url)

    # 자막이 전혀 없으면 Whisper로 넘김
    if not sub_info["has_manual"] and not sub_info["has_auto"]:
        print("[TADAC] 자막 없음 → Whisper STT로 전환")
        return {}, "no_subtitle"

    # 임시 폴더에 VTT 다운로드 후 파싱 (완료 후 자동 삭제)
    with tempfile.TemporaryDirectory(prefix="tadac_vtt_") as tmp_dir:
        vtt_path, source_type = download_vtt(youtube_url, tmp_dir, sub_info, preferred_lang)
        transcript = parse_vtt(vtt_path)

    return transcript, source_type
