# -*- coding: utf-8 -*-
"""
STT 모듈 — 학교 GPU 서버의 로컬 Whisper API를 호출하여 음성→텍스트 변환

- 학교 GPU 서버(app_whisper.py)가 OpenAI Audio Transcriptions API 규격을 모방하므로
  OpenAI SDK의 base_url만 변경하여 호출
- 서버에서 받은 단어 타임스탬프를 세그먼트로 그루핑 (문장 종결 부호 + 묵음 기반)
- 출력 포맷은 기존과 동일: {text, words, segments, language}
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── 원격 Whisper 서버 설정 ───────────────────────────────────────────────────────
WHISPER_API_URL = os.getenv("WHISPER_API_URL", "http://210.94.179.19:8002/v1")

_whisper_client = None

def _get_whisper_client():
    global _whisper_client
    if _whisper_client is None:
        import httpx
        _whisper_client = OpenAI(
            base_url=WHISPER_API_URL,
            api_key="local",
            timeout=httpx.Timeout(1800.0, connect=30.0),
        )
        print(f"[TADAC] Whisper API 클라이언트 초기화: {WHISPER_API_URL}")
    return _whisper_client


# ── words → segments 그루핑 ──────────────────────────────────────────────────────

SENTENCE_END_CHARS = {".", "?", "!", "。", "?", "!"}

MIN_SEGMENT_SEC = 2.5
MAX_SEGMENT_CHARS = 55
MIN_SEGMENT_CHARS = 30


def _dedupe_hallucination_loops(segments, min_duration=0.1):
    """Whisper 디코더 반복 루프로 생긴 zero-duration 중복 세그먼트 제거."""
    if not segments:
        return segments

    result  = [dict(segments[0])]
    dropped = 0
    for seg in segments[1:]:
        prev        = result[-1]
        is_zero_dur = (seg["end"] - seg["start"]) < min_duration
        same_text   = seg["text"].strip() == prev["text"].strip()
        if is_zero_dur and same_text:
            dropped += 1
            continue
        result.append(dict(seg))

    if dropped:
        print(f"[TADAC] 환각 루프 제거: {dropped}개 중복 세그먼트")

    for i, seg in enumerate(result):
        seg["id"] = i

    return result


def _merge_short_segments(segments, min_chars=MIN_SEGMENT_CHARS):
    """글자 수가 min_chars 미만인 세그먼트를 인접 세그먼트와 병합."""
    if not segments:
        return segments

    result = []
    buf    = None

    for seg in segments:
        if buf is None:
            buf = dict(seg)
            continue

        if len(buf["text"]) < min_chars:
            buf["end"]  = seg["end"]
            buf["text"] = (buf["text"] + " " + seg["text"]).strip()
        else:
            result.append(buf)
            buf = dict(seg)

    if buf is not None:
        if len(buf["text"]) < min_chars and result:
            last = result[-1]
            last["end"]  = buf["end"]
            last["text"] = (last["text"] + " " + buf["text"]).strip()
        else:
            result.append(buf)

    for i, seg in enumerate(result):
        seg["id"]    = i
        seg["start"] = round(seg["start"], 3)
        seg["end"]   = round(seg["end"],   3)

    return result


def _split_long_segments(segments, words, max_chars=MAX_SEGMENT_CHARS):
    """글자 수가 max_chars를 초과하는 세그먼트를 단어 경계에서 분할."""
    if not segments or not words:
        return segments

    word_idx = 0
    result = []

    for seg in segments:
        seg_text = seg["text"]
        if len(seg_text) <= max_chars:
            result.append(seg)
            word_idx += len(seg_text.split())
            continue

        seg_words_list = seg_text.split()
        seg_word_count = len(seg_words_list)
        part_start_word_idx = word_idx

        parts = []
        current_words = []
        for w in seg_words_list:
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
            n_words = len(part_words)

            abs_from = part_start_word_idx + w_offset
            abs_to = min(abs_from + n_words - 1, len(words) - 1)

            if abs_from < len(words) and abs_to < len(words):
                p_start = words[abs_from]["start"]
                p_end = words[abs_to]["end"]
            else:
                dur = seg["end"] - seg["start"]
                p_start = seg["start"] + dur * (w_offset / seg_word_count)
                p_end = seg["start"] + dur * ((w_offset + n_words) / seg_word_count)

            result.append({
                "id": 0,
                "start": round(p_start, 3),
                "end": round(p_end, 3),
                "text": part_text,
            })
            w_offset += n_words

        word_idx += seg_word_count

    for i, seg in enumerate(result):
        seg["id"] = i

    if len(result) != len(segments):
        print(f"[TADAC] 긴 세그먼트 분할: {len(segments)}개 → {len(result)}개 (max {max_chars}자)")

    return result


def _group_words_into_segments(words, max_gap_sec=1.5):
    """단어 리스트를 문장 종결 부호 또는 긴 묵음 기준으로 segment 단위로 묶기."""
    segments = []
    current  = []
    seg_id   = 0

    def _flush():
        nonlocal current, seg_id
        if not current:
            return
        segments.append({
            "id":    seg_id,
            "start": round(current[0]["start"], 3),
            "end":   round(current[-1]["end"],   3),
            "text":  "".join(w["word"] for w in current).strip(),
        })
        seg_id += 1
        current = []

    for i, w in enumerate(words):
        current.append(w)
        text_stripped = w["word"].strip()

        current_text = "".join(cw["word"] for cw in current).strip()
        if len(current_text) >= MAX_SEGMENT_CHARS:
            _flush()
            continue

        if text_stripped and text_stripped[-1] in SENTENCE_END_CHARS:
            _flush()
            continue

        if i + 1 < len(words):
            gap = words[i + 1]["start"] - w["end"]
            if gap >= max_gap_sec:
                _flush()

    _flush()
    return segments


# ── 원격 Whisper 호출 ────────────────────────────────────────────────────────────

MAX_RETRIES    = 3
RETRY_BASE_SEC = 5


def _run_transcribe(audio_path, language, prompt):
    """학교 GPU 서버의 Whisper API를 호출하여 추론 결과를 받아오고 후처리."""
    client = _get_whisper_client()

    for attempt in range(MAX_RETRIES):
        try:
            with open(audio_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=f,
                    language=language or "ko",
                    prompt=prompt or "",
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                )
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_SEC * (2 ** attempt)
                print(f"[TADAC] Whisper API 호출 실패: {e} → {delay}초 후 재시도 ({attempt + 1}/{MAX_RETRIES})")
                time.sleep(delay)
            else:
                raise RuntimeError(f"Whisper API 호출 실패 (재시도 소진): {e}") from e

    all_words = []
    for w in (response.words or []):
        all_words.append({
            "word":  w.word,
            "start": round(w.start, 3),
            "end":   round(w.end, 3),
        })

    segments = _group_words_into_segments(all_words)
    segments = _dedupe_hallucination_loops(segments)
    segments = _split_long_segments(segments, all_words)
    before   = len(segments)
    segments = _merge_short_segments(segments)
    if before != len(segments):
        print(f"[TADAC] 짧은 세그먼트 병합: {before}개 → {len(segments)}개 (min {MIN_SEGMENT_CHARS}자)")

    return {
        "text":     response.text.strip(),
        "words":    all_words,
        "segments": segments,
        "language": language or "ko",
    }


# ── 메인 함수 ────────────────────────────────────────────────────────────────────

def transcribe(audio_path, language="ko", stt_prompt=None, title=None):
    audio_path = str(Path(audio_path).resolve())
    file_size  = os.path.getsize(audio_path)

    print(f"[TADAC] STT 시작 (원격): {audio_path} ({file_size / 1024 / 1024:.1f} MB)")

    parts = []
    if title:
        if language == "ko":
            parts.append(f"이 영상은 '{title}'에 관한 강의입니다.")
        else:
            parts.append(f"This is a lecture about '{title}'.")
    if stt_prompt:
        parts.append(stt_prompt)
    effective_prompt = " ".join(parts) if parts else None
    if effective_prompt:
        print(f"[TADAC] STT 프롬프트 힌트: {effective_prompt[:120]}")

    result = _run_transcribe(audio_path, language, effective_prompt)

    print(f"[TADAC] STT 완료: 단어 {len(result.get('words', []))}개, "
          f"세그먼트 {len(result.get('segments', []))}개 | "
          f"언어: {result.get('language', '?')}")

    return result
