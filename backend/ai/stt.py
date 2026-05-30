# -*- coding: utf-8 -*-
"""
STT 모듈 — 학교 GPU 서버의 로컬 Whisper API를 호출하여 음성→텍스트 변환

- 학교 GPU 서버(app_whisper.py)가 OpenAI Audio Transcriptions API 규격을 모방하므로
  OpenAI SDK의 base_url만 변경하여 호출
- 서버에서 받은 세그먼트 타임스탬프를 중심으로 후처리
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
WHISPER_CLIENT_MAX_PARALLEL = os.getenv("WHISPER_CLIENT_MAX_PARALLEL")

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


# ── segments 후처리 ──────────────────────────────────────────────────────────────

SENTENCE_END_CHARS = {".", "?", "!", "。", "?", "!"}

MIN_SEGMENT_SEC = 2.5
MAX_SEGMENT_CHARS = 55
MIN_SEGMENT_CHARS = 30


def _get_field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _normalise_words(raw_words):
    """하위 호환용 words 필드를 유지하되, 없으면 빈 배열로 둔다."""
    all_words = []
    for w in raw_words or []:
        word = _get_field(w, "word", "")
        if not word:
            continue
        all_words.append({
            "word":  word,
            "start": round(float(_get_field(w, "start", 0.0) or 0.0), 3),
            "end":   round(float(_get_field(w, "end", 0.0) or 0.0), 3),
        })
    return all_words


def _normalise_segments(raw_segments, fallback_text="", fallback_duration=0.0):
    segments = []
    for i, seg in enumerate(raw_segments or []):
        text = (_get_field(seg, "text", "") or "").strip()
        if not text:
            continue
        start = float(_get_field(seg, "start", 0.0) or 0.0)
        end = float(_get_field(seg, "end", start) or start)
        segments.append({
            "id":    int(_get_field(seg, "id", len(segments)) or len(segments)),
            "start": round(start, 3),
            "end":   round(end, 3),
            "text":  text,
        })

    if not segments and fallback_text.strip():
        segments.append({
            "id":    0,
            "start": 0.0,
            "end":   round(float(fallback_duration or 0.0), 3),
            "text":  fallback_text.strip(),
        })

    for i, seg in enumerate(segments):
        seg["id"] = i
    return segments


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


def _merge_short_segments(segments, min_chars=MIN_SEGMENT_CHARS, max_chars=MAX_SEGMENT_CHARS):
    """글자 수가 min_chars 미만인 세그먼트를 인접 세그먼트와 병합 (max_chars 초과 방지)."""
    if not segments:
        return segments

    result = []
    buf    = None

    for seg in segments:
        if buf is None:
            buf = dict(seg)
            continue

        merged_text = (buf["text"] + " " + seg["text"]).strip()
        if len(buf["text"]) < min_chars and len(merged_text) <= max_chars:
            buf["end"]  = seg["end"]
            buf["text"] = merged_text
        else:
            result.append(buf)
            buf = dict(seg)

    if buf is not None:
        if len(buf["text"]) < min_chars and result:
            merged_text = (result[-1]["text"] + " " + buf["text"]).strip()
            if len(merged_text) <= max_chars:
                result[-1]["end"]  = buf["end"]
                result[-1]["text"] = merged_text
            else:
                result.append(buf)
        else:
            result.append(buf)

    for i, seg in enumerate(result):
        seg["id"]    = i
        seg["start"] = round(seg["start"], 3)
        seg["end"]   = round(seg["end"],   3)

    return result


def _split_long_segments(segments, words, max_chars=MAX_SEGMENT_CHARS):
    """글자 수가 max_chars를 초과하는 세그먼트를 단어 경계에서 분할."""
    if not segments:
        return segments
    if not words:
        words = []

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


def get_gpu_status():
    """Whisper 서버의 GPU 워커 풀 상태를 조회. 실패 시 None 반환."""
    import httpx
    base_url = WHISPER_API_URL.rstrip("/")

    # /v1/gpu-status 먼저 시도 (프록시 환경), 실패하면 루트 시도
    urls_to_try = [f"{base_url}/gpu-status"]
    if base_url.endswith("/v1"):
        root = base_url[:-3]
        urls_to_try.append(f"{root}/gpu-status")

    for url in urls_to_try:
        try:
            resp = httpx.get(url, timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            print(f"[TADAC] GPU 상태: {data.get('total_gpus', '?')}대 중 {data.get('free_gpus', '?')}대 유휴")
            return data
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                continue  # 다음 URL 시도
            print(f"[TADAC] GPU 상태 조회 실패: {e}")
            return None
        except Exception:
            continue

    # 모든 URL 실패 → 구버전 서버로 간주
    print(f"[TADAC] GPU 상태 엔드포인트 없음 (구버전 서버) → 단일 GPU로 처리")
    return {"total_gpus": 1, "free_gpus": 1, "workers": []}


def transcribe_parallel(chunk_list, language="ko", stt_prompt=None, title=None):
    """여러 오디오 청크를 Whisper 서버의 빈 GPU에 병렬로 전송하여 STT 수행.

    Args:
        chunk_list: [(audio_path, offset_sec), ...] — 각 청크의 파일 경로와 시간 오프셋
        language: STT 언어 코드
        stt_prompt: STT 힌트 프롬프트
        title: 영상 제목 (프롬프트 구성용)

    Returns:
        list of (transcript_result, offset_sec) — 각 청크의 STT 결과와 오프셋.
        순서는 chunk_list와 동일 (offset 기준 정렬 보장).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # GPU 상태 확인하여 동시 요청 수 결정
    gpu_status = get_gpu_status()
    if gpu_status and gpu_status.get("total_gpus", 0) > 0:
        max_parallel = gpu_status["total_gpus"]
        free_gpus = gpu_status.get("free_gpus", max_parallel)
        print(f"[TADAC] GPU 워커 풀: {max_parallel}대 총, {free_gpus}대 유휴")
    else:
        # 상태 조회 실패 시 순차 처리 (안전 폴백)
        max_parallel = 1
        print(f"[TADAC] GPU 상태 불명 → 순차 STT")

    # 프롬프트 구성
    parts = []
    if title:
        if language == "ko":
            parts.append(f"이 영상은 '{title}'에 관한 강의입니다.")
        else:
            parts.append(f"This is a lecture about '{title}'.")
    if stt_prompt:
        parts.append(stt_prompt)
    effective_prompt = " ".join(parts) if parts else None

    def _stt_one(chunk_idx, audio_path, offset_sec):
        """단일 청크 STT (스레드에서 실행)."""
        file_size = os.path.getsize(audio_path)
        print(f"[TADAC] STT chunk {chunk_idx+1}/{len(chunk_list)}: "
              f"{audio_path} ({file_size / 1024 / 1024:.1f} MB, offset={offset_sec:.0f}초)")

        result = _run_transcribe(audio_path, language, effective_prompt)
        return chunk_idx, result, offset_sec

    results = [None] * len(chunk_list)

    # 서버가 알아서 GPU 큐잉을 해주므로, 청크를 모두 동시에 보내도 됨.
    # 다만 max_parallel로 클라이언트 측에서도 동시 요청을 제한하여
    # 서버 메모리 부담과 네트워크 부하를 줄임.
    client_limit = None
    if WHISPER_CLIENT_MAX_PARALLEL:
        try:
            client_limit = max(1, int(WHISPER_CLIENT_MAX_PARALLEL))
        except ValueError:
            print(f"[TADAC] WHISPER_CLIENT_MAX_PARALLEL 파싱 실패: {WHISPER_CLIENT_MAX_PARALLEL}")

    effective_parallel = min(max_parallel, len(chunk_list))
    if client_limit is not None:
        effective_parallel = min(effective_parallel, client_limit)
        print(f"[TADAC] STT 클라이언트 병렬 제한: {client_limit}")
    print(f"[TADAC] 병렬 STT 시작: {len(chunk_list)}개 청크, 동시 {effective_parallel}개")

    with ThreadPoolExecutor(max_workers=effective_parallel) as executor:
        futures = {}
        for i, (audio_path, offset_sec) in enumerate(chunk_list):
            future = executor.submit(_stt_one, i, audio_path, offset_sec)
            futures[future] = i

        for future in as_completed(futures):
            try:
                chunk_idx, result, offset_sec = future.result()
                results[chunk_idx] = (result, offset_sec)
                print(f"[TADAC] STT chunk {chunk_idx+1} 완료: "
                      f"세그먼트 {len(result.get('segments', []))}개, "
                      f"단어 {len(result.get('words', []))}개")
            except Exception as e:
                chunk_idx = futures[future]
                print(f"[TADAC] STT chunk {chunk_idx+1} 실패: {e}")
                # 실패한 청크는 빈 결과로 대체
                results[chunk_idx] = (
                    {"text": "", "words": [], "segments": [], "language": language},
                    chunk_list[chunk_idx][1],
                )

    print(f"[TADAC] 병렬 STT 완료: {len(chunk_list)}개 청크 처리됨")
    return results


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
                    timestamp_granularities=["segment"],
                )
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_SEC * (2 ** attempt)
                print(f"[TADAC] Whisper API 호출 실패: {e} → {delay}초 후 재시도 ({attempt + 1}/{MAX_RETRIES})")
                time.sleep(delay)
            else:
                raise RuntimeError(f"Whisper API 호출 실패 (재시도 소진): {e}") from e

    text = (_get_field(response, "text", "") or "").strip()
    all_words = _normalise_words(_get_field(response, "words", None))
    segments = _normalise_segments(
        _get_field(response, "segments", None),
        fallback_text=text,
        fallback_duration=_get_field(response, "duration", 0.0),
    )
    if not segments and all_words:
        segments = _group_words_into_segments(all_words)
    segments = _dedupe_hallucination_loops(segments)
    segments = _split_long_segments(segments, all_words)
    before   = len(segments)
    segments = _merge_short_segments(segments)
    if before != len(segments):
        print(f"[TADAC] 짧은 세그먼트 병합: {before}개 → {len(segments)}개 (min {MIN_SEGMENT_CHARS}자)")

    return {
        "text":     text,
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
