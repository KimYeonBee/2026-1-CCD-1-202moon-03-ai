# -*- coding: utf-8 -*-
"""
STT 모듈 — Whisper API로 음성을 텍스트로 변환

- 단어별 타임스탬프 추출 (빈칸 낙하 동기화 핵심)
- 25MB 초과 파일은 10분씩 잘라서 처리 후 합치기
"""

import os
import math
import tempfile
from pathlib import Path

from openai import OpenAI
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()

# OpenAI 클라이언트 초기화 (Whisper는 게이트웨이 미지원 → OpenAI 직접 연결)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Whisper API 파일 크기 제한 25MB
WHISPER_SIZE_LIMIT = 25 * 1024 * 1024  # 25MB (바이트)
CHUNK_DURATION_MS  = 10 * 60 * 1000    # 청크 길이: 10분 (밀리초)


# ── 단일 파일 음성 인식 ────────────────────────────────────────────────────────
# Whisper API에 파일을 보내고 단어별 타임스탬프를 받아옴

def _transcribe_chunk(audio_path, language, prompt):
    with open(audio_path, "rb") as f:
        # timestamp_granularities=["word"] → 단어 단위 타임스탬프 (낙하 이벤트 동기화 핵심)
        kwargs = dict(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            language=language,           # 언어 힌트 (정확도 향상)
        )
        if prompt:
            kwargs["prompt"] = prompt    # 전문 용어 힌트 (선택)

        response = client.audio.transcriptions.create(**kwargs)

    return response.model_dump()


# ── 대용량 파일 분할 ──────────────────────────────────────────────────────────
# 25MB 초과 시 pydub으로 10분씩 잘라서 청크 파일 목록 반환

def _split_audio(audio_path, chunk_dir):
    ext = Path(audio_path).suffix.lower().lstrip(".")  # 확장자 추출

    # 확장자에 맞게 오디오 파일 읽기
    if ext == "mp3":
        audio = AudioSegment.from_mp3(audio_path)
    elif ext == "wav":
        audio = AudioSegment.from_wav(audio_path)
    else:
        audio = AudioSegment.from_file(audio_path)

    total_ms = len(audio)  # 전체 길이 (밀리초)
    n_chunks  = math.ceil(total_ms / CHUNK_DURATION_MS)  # 청크 개수

    print(f"[TADAC] 오디오 분할: {n_chunks}개 청크 (총 {total_ms/1000:.1f}초)")

    # 10분씩 자르고 임시 폴더에 저장
    chunk_paths = []
    for i in range(n_chunks):
        start = i * CHUNK_DURATION_MS
        end   = min((i + 1) * CHUNK_DURATION_MS, total_ms)
        chunk = audio[start:end]

        chunk_path = os.path.join(chunk_dir, f"chunk_{i:03d}.mp3")
        chunk.export(chunk_path, format="mp3")
        chunk_paths.append(chunk_path)

        print(f"[TADAC] 청크 {i+1}/{n_chunks}: {start/1000:.1f}s ~ {end/1000:.1f}s")

    return chunk_paths


# ── 청크 결과 합치기 ──────────────────────────────────────────────────────────
# 각 청크의 타임스탬프에 오프셋을 더해서 전체 타임라인으로 병합

def _merge_results(results, chunk_offsets_sec):
    merged_words    = []
    merged_segments = []
    full_text_parts = []
    seg_id_offset   = 0  # 세그먼트 ID가 청크마다 0부터 시작 → 전체 기준으로 보정

    for result, offset in zip(results, chunk_offsets_sec):
        full_text_parts.append(result.get("text", "").strip())

        # 단어 타임스탬프에 오프셋 더하기
        for word in result.get("words", []):
            merged_words.append({
                "word":  word["word"],
                "start": round(word["start"] + offset, 3),
                "end":   round(word["end"]   + offset, 3),
            })

        # 세그먼트 타임스탬프에도 오프셋 더하기
        for seg in result.get("segments", []):
            merged_seg          = dict(seg)
            merged_seg["id"]    = seg_id_offset + seg.get("id", 0)
            merged_seg["start"] = round(seg["start"] + offset, 3)
            merged_seg["end"]   = round(seg["end"]   + offset, 3)
            merged_segments.append(merged_seg)

        seg_id_offset += len(result.get("segments", []))

    return {
        "text":     " ".join(full_text_parts),
        "words":    merged_words,
        "segments": merged_segments,
        "language": results[0].get("language", "ko") if results else "ko",
    }


# ── 메인 함수 ─────────────────────────────────────────────────────────────────
# 음성 파일 경로를 받아 Whisper STT 결과 반환

def transcribe(audio_path, language="ko", stt_prompt=None, title=None):
    audio_path = str(Path(audio_path).resolve())
    file_size  = os.path.getsize(audio_path)  # 파일 크기 (바이트)

    print(f"[TADAC] STT 시작: {audio_path} ({file_size / 1024 / 1024:.1f} MB)")

    # 영상/파일 제목 + 사용자 prompt 결합.
    # Whisper prompt는 vocabulary biasing(단어 인식 정확도 향상) 효과뿐 아니라
    # 출력 스타일(문장 길이/구두점)까지 모방하므로, 제목을 그대로 넣으면
    # 짧고 단편적인 segments가 양산됨. → 자연스러운 강의 톤 문장으로 감싸서 전달.
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

    # 25MB 이하 → 바로 Whisper 전송
    if file_size <= WHISPER_SIZE_LIMIT:
        print("[TADAC] 25MB 이하 → 직접 변환")
        result = _transcribe_chunk(audio_path, language, effective_prompt)
        print(f"[TADAC] STT 완료: 단어 {len(result.get('words', []))}개, "
              f"세그먼트 {len(result.get('segments', []))}개")
        return result

    # 25MB 초과 → 청크 분할 후 병합
    print("[TADAC] 25MB 초과 → 청크 분할 처리")
    with tempfile.TemporaryDirectory(prefix="tadac_chunks_") as chunk_dir:
        chunk_paths = _split_audio(audio_path, chunk_dir)

        # 각 청크의 시작 오프셋 계산 (이전 청크 길이 누적)
        offsets = [0.0]
        for cp in chunk_paths[:-1]:
            duration = len(AudioSegment.from_file(cp)) / 1000.0  # 밀리초 → 초
            offsets.append(offsets[-1] + duration)

        # 청크별 음성 인식
        results = []
        for i, (cp, offset) in enumerate(zip(chunk_paths, offsets)):
            print(f"[TADAC] 청크 {i+1}/{len(chunk_paths)} 변환 중 (오프셋 {offset:.1f}s)")
            results.append(_transcribe_chunk(cp, language, effective_prompt))

        # 전체 결과 병합
        merged = _merge_results(results, offsets)
        print(f"[TADAC] 병합 완료: 단어 {len(merged['words'])}개, 세그먼트 {len(merged['segments'])}개")
        return merged
