# -*- coding: utf-8 -*-
"""
TADAC AI 파이프라인 — 메인 오케스트레이터

입력: YouTube URL / 로컬 오디오(.mp3 .wav .m4a) / 로컬 비디오(.mp4 .webm)
      단, FastAPI 업로드 엔드포인트는 현재 백엔드 연동 범위를 줄이기 위해 비디오(.mp4 .webm)만 허용
출력: 프론트엔드가 바로 쓰는 game_data.json

실행 예시:
    python pipeline.py ./lecture.mp3
    python pipeline.py "https://www.youtube.com/watch?v=..."
    python pipeline.py ./lecture.mp4 --no-refine --prompt "ADHD,도파민,전두엽"

난이도 파라미터 (fall_speed, lead_time) 는 프론트엔드가 관리하므로 CLI에서 제거.
blanks_per_sentence 는 항상 최대치(2)로 생성하고 프론트가 몇 개 보여줄지 결정.
"""

import argparse
import json
import os
import shutil

import sys
import tempfile
import time as _time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 같은 폴더의 모듈 임포트
sys.path.insert(0, str(Path(__file__).parent))

import base64
import subprocess

import stt as stt_module
import transcript_refiner
import keyword_extractor
import blank_subtitle
import quiz_generator
import youtube_subtitle
import youtube_audio
import combined_processor
import shorts_generator
import shorts_builder

# 지원 형식 정의
YOUTUBE_PREFIXES = ("https://www.youtube.com", "https://youtu.be", "http://www.youtube.com")
VIDEO_EXTENSIONS = {".mp4", ".webm"}

# 내부 고정값 — 난이도는 프론트엔드가 관리
MAX_BLANKS_PER_SENTENCE = 2    # 세그먼트당 최대 빈칸 수
BASE_FALL_SPEED         = 1.0  # 프론트가 target_time 기준으로 재계산
BASE_LEAD_TIME          = 3.0  # 프론트가 target_time 기준으로 재계산


# ── 입력 종류 판별 ────────────────────────────────────────────────────────────

def _is_youtube_url(src):
    return any(src.startswith(p) for p in YOUTUBE_PREFIXES)



# ── 오디오 10분 단위 분할 ────────────────────────────────────────────────────
CHUNK_DURATION_SEC = int(os.getenv("STT_CHUNK_DURATION_SEC", "600"))  # 10분
STT_MODE = os.getenv("STT_MODE", "single").strip().lower()
STT_CHUNKING_ENABLED = STT_MODE in {"chunk", "chunked", "parallel"}
STT_SEQUENTIAL_CHUNKING_ENABLED = STT_MODE in {"sequential_chunk", "chunk_sequential", "streaming_chunk"}
STT_CHUNK_THRESHOLD_SEC = float(os.getenv("STT_CHUNK_THRESHOLD_SEC", "3600"))  # 60분

print(
    "[TADAC] STT chunk 설정: "
    f"mode={STT_MODE}, parallel={STT_CHUNKING_ENABLED}, sequential={STT_SEQUENTIAL_CHUNKING_ENABLED}, "
    f"threshold={STT_CHUNK_THRESHOLD_SEC/60:.1f}분, "
    f"chunk={CHUNK_DURATION_SEC/60:.1f}분"
)


def _probe_audio_duration(audio_path):
    import subprocess as _sp
    probe = _sp.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe 오디오 길이 확인 실패: {audio_path}")
    return float(probe.stdout.strip())


def _build_backward_chunk_ranges(total_duration, chunk_sec=CHUNK_DURATION_SEC):
    ranges = []
    end = float(total_duration)
    while end > 0:
        start = max(0.0, end - chunk_sec)
        ranges.append((start, end))
        end = start
    ranges.reverse()
    return ranges


def _split_audio_into_chunks(audio_path, tmp_dir, chunk_sec=CHUNK_DURATION_SEC):
    """오디오를 chunk_sec 단위로 분할. 반환: [(chunk_path, offset_sec), ...]

    뒤에서부터 chunk_sec 단위로 자른 뒤 시간순으로 반환한다.
    예: 17분 영상 → 0~7분, 7~17분. 병렬 STT에서 앞 영상 청크가 먼저 끝나기 쉽다.
    """
    import subprocess as _sp
    total_duration = _probe_audio_duration(audio_path)
    print(f"[TADAC] 오디오 총 길이: {total_duration:.1f}초 ({total_duration/60:.1f}분)")

    if total_duration <= chunk_sec * 1.3:
        # 분할 불필요 (chunk 1개 + 짧은 꼬리만 남을 때)
        return [(audio_path, 0.0)]

    ranges = _build_backward_chunk_ranges(total_duration, chunk_sec)

    chunks = []
    for idx, (offset, end) in enumerate(ranges):
        chunk_path = os.path.join(tmp_dir, f"chunk_{idx:03d}.mp3")
        duration = end - offset
        _sp.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-ss", str(offset), "-t", str(duration),
             "-acodec", "mp3", "-loglevel", "quiet", chunk_path],
            check=True,
        )
        chunks.append((chunk_path, offset))
        print(f"[TADAC] chunk {idx}: {offset:.0f}초~{end:.0f}초")

    print(f"[TADAC] 오디오 분할 완료: {len(chunks)}개 chunk")
    return chunks


def _build_streaming_chunk_ranges(audio_path, chunk_sec=CHUNK_DURATION_SEC):
    """스트리밍용 청크 범위 계산. 10분을 넘으면 앞 청크가 짧아지도록 뒤에서부터 자른다."""
    total_duration = _probe_audio_duration(audio_path)
    print(f"[TADAC] 스트리밍 청크 오디오 길이: {total_duration:.1f}초 ({total_duration/60:.1f}분)")

    if total_duration <= chunk_sec:
        return total_duration, [(0.0, total_duration)]

    ranges = _build_backward_chunk_ranges(total_duration, chunk_sec)
    for idx, (offset, end) in enumerate(ranges):
        print(f"[TADAC] streaming chunk range {idx}: {offset:.0f}초~{end:.0f}초")
    return total_duration, ranges


def _materialize_streaming_chunk(audio_path, tmp_dir, chunk_idx, offset, end):
    """현재 처리할 청크만 파일로 자른다. 첫 재생 전 전체 청크 생성을 피하기 위함."""
    if offset <= 0.0 and end >= _probe_audio_duration(audio_path) - 0.01:
        return audio_path

    chunk_path = os.path.join(tmp_dir, f"stream_chunk_{chunk_idx:03d}.mp3")
    duration = end - offset
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path,
         "-ss", str(offset), "-t", str(duration),
         "-acodec", "mp3", "-loglevel", "quiet", chunk_path],
        check=True,
    )
    print(f"[TADAC] streaming chunk {chunk_idx} 파일 생성: {offset:.0f}초~{end:.0f}초")
    return chunk_path


def _transcribe_auto_parallel(audio_path, language, stt_prompt, content_title, tmp_dirs, request_id=None):
    """환경 설정에 따라 단일 STT 또는 멀티 GPU 청크 STT를 수행.

    기본값은 단일 STT다. word_timestamps=False 환경에서는 청크 분할의 속도 이득보다
    중간 청크 누락 리스크가 더 커서, 긴 영상 병렬화가 필요할 때만 env로 켠다.

    Args:
        tmp_dirs: 임시 폴더 추적 리스트 (finally에서 정리용)

    Returns:
        dict: {"text": str, "words": list, "segments": list, "language": str}
    """
    total_duration = _probe_audio_duration(audio_path)
    should_chunk = (
        STT_CHUNKING_ENABLED
        and total_duration > STT_CHUNK_THRESHOLD_SEC
        and total_duration > CHUNK_DURATION_SEC * 1.3
    )

    if should_chunk:
        print(
            "[TADAC] STT 모드: 청크 병렬 "
            f"(duration={total_duration/60:.1f}분, "
            f"threshold={STT_CHUNK_THRESHOLD_SEC/60:.1f}분, "
            f"chunk={CHUNK_DURATION_SEC/60:.1f}분)"
        )
        chunk_tmp_dir = tempfile.mkdtemp(prefix="tadac_chunks_")
        tmp_dirs.append(chunk_tmp_dir)
        audio_chunks = _split_audio_into_chunks(audio_path, chunk_tmp_dir)
    else:
        reason = "비활성화" if not STT_CHUNKING_ENABLED else "threshold 미만"
        print(
            "[TADAC] STT 모드: 단일 파일 "
            f"(duration={total_duration/60:.1f}분, chunking={reason})"
        )
        audio_chunks = [(audio_path, 0.0)]

    all_segments = []
    all_words    = []
    global_seg_id = 0

    if len(audio_chunks) > 1:
        # 멀티 GPU 병렬 STT
        chunk_results = stt_module.transcribe_parallel(
            audio_chunks,
            language=language,
            stt_prompt=stt_prompt,
            title=content_title,
            request_id=request_id,
        )

        for transcript_result, offset_sec in chunk_results:
            ch_segs  = transcript_result.get("segments", [])
            ch_words = transcript_result.get("words", [])

            if not ch_segs:
                continue

            for seg in ch_segs:
                seg["start"] = round(seg["start"] + offset_sec, 3)
                seg["end"]   = round(seg["end"]   + offset_sec, 3)
                seg["id"]    = global_seg_id
                global_seg_id += 1
            for w in ch_words:
                w["start"] = round(w["start"] + offset_sec, 3)
                w["end"]   = round(w["end"]   + offset_sec, 3)

            all_segments.extend(ch_segs)
            all_words.extend(ch_words)
    else:
        # 단일 청크 — 분할 불필요
        chunk_path = audio_chunks[0][0]
        result = stt_module.transcribe(
            chunk_path, language=language, stt_prompt=stt_prompt, title=content_title
        )
        ch_segs  = result.get("segments", [])
        ch_words = result.get("words", [])
        for seg in ch_segs:
            seg["id"] = global_seg_id
            global_seg_id += 1
        all_segments.extend(ch_segs)
        all_words.extend(ch_words)

    return {
        "text":     " ".join(seg.get("text", "") for seg in all_segments),
        "words":    all_words,
        "segments": all_segments,
        "language": language,
    }


def _log_large_transcript_gaps(segments, threshold_sec=120.0):
    """STT 결과에 큰 시간 공백이 있으면 디버깅용 로그를 남긴다."""
    if len(segments) < 2:
        return

    gaps = []
    ordered = sorted(segments, key=lambda s: s.get("start", 0.0))
    for prev, cur in zip(ordered, ordered[1:]):
        gap = float(cur.get("start", 0.0) or 0.0) - float(prev.get("end", 0.0) or 0.0)
        if gap >= threshold_sec:
            gaps.append((prev, cur, gap))

    if not gaps:
        return

    print(f"[TADAC] ⚠️ STT 큰 공백 감지: {len(gaps)}개 (기준 {threshold_sec:.0f}초)")
    for prev, cur, gap in gaps[:10]:
        print(
            "[TADAC]   "
            f"{prev.get('end', 0.0)/60:.1f}분~{cur.get('start', 0.0)/60:.1f}분 "
            f"({gap/60:.1f}분), "
            f"prev_id={prev.get('id')}, next_id={cur.get('id')}"
        )


# ── 비디오 → 오디오 추출 (로컬 파일용) ───────────────────────────────────────
# ffmpeg으로 영상에서 오디오 트랙만 mp3로 추출

def _extract_audio_from_video(video_path, tmp_dir):
    audio_path = os.path.join(tmp_dir, "extracted_audio.mp3")
    ret = os.system(f'ffmpeg -y -i "{video_path}" -vn -acodec mp3 "{audio_path}" -loglevel quiet')

    if ret != 0 or not os.path.exists(audio_path):
        raise RuntimeError(f"ffmpeg 오디오 추출 실패: {video_path}")

    print(f"[TADAC] 비디오 오디오 추출 완료: {audio_path}")
    return audio_path


# ── Whisper 원본 저장 ─────────────────────────────────────────────────────────
# STT 결과를 raw_transcript.json으로 보존 (재처리 및 디버깅용)

def _save_raw_transcript(transcript, output_dir=None):
    """Whisper STT 원본 결과를 JSON 파일로 저장"""
    if output_dir is None:
        output_dir = Path(__file__).parent
    
    raw_path = Path(output_dir) / "raw_transcript.json"
    
    # words는 크기가 크므로 타임스탬프만 보존
    save_data = {
        "segments": [
            {
                "id": seg.get("id", i),
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": seg.get("text", ""),
            }
            for i, seg in enumerate(transcript.get("segments", []))
        ],
        "text": transcript.get("text", ""),
        "language": transcript.get("language", "ko"),
    }
    
    raw_path.write_text(json.dumps(save_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[TADAC] Whisper 원본 저장: {raw_path}")
    return raw_path


# ── 키워드 폴백 보완 ─────────────────────────────────────────────────────────
# GPT가 놓친 키워드를 global_keywords 문자열 매칭으로 채움

def _fill_missing_keywords(enriched_segments, global_keywords, all_words, max_per_segment=2):
    """GPT가 놓친 키워드를 전역 목록에서 문자열 매칭으로 보완.

    all_words는 하위 호환용 파라미터이며, 키워드 스케줄은 세그먼트 기준으로 계산한다.
    """
    filled_count = 0
    for seg in enriched_segments:
        existing_kws = [kw["keyword"] for kw in seg.get("keywords", [])]
        if len(existing_kws) >= max_per_segment:
            continue

        remaining = max_per_segment - len(existing_kws)
        text = seg.get("text", "")

        for term in global_keywords:
            if remaining <= 0:
                break
            if len(term) < 2 or term in existing_kws:
                continue
            if term in text:
                seg["keywords"].append({
                    "keyword": term,
                    "found": False,
                })
                existing_kws.append(term)
                remaining -= 1
                filled_count += 1
        keyword_extractor.schedule_segment_keywords(seg)

    if filled_count:
        print(f"[TADAC] 키워드 폴백 보완: {filled_count}개 추가")

    return enriched_segments


# ── GPT 결과 후처리 ───────────────────────────────────────────────────────────
# corrections(diff) 적용 + segment_keywords → enriched segments 생성

def _apply_gpt_results(ch_segs, combined_result, all_words, name_corrections=None):
    """
    GPT 경량 output을 원본 세그먼트에 적용하여 enriched segments를 생성.
    
    Args:
        ch_segs: 원본 세그먼트 목록
        combined_result: GPT 반환값 {corrections, segment_keywords, quizzes}
        all_words: 하위 호환용 파라미터. 현재 키워드 스케줄 계산에는 사용하지 않음.
        name_corrections: {"잘못된 표기": "올바른 표기", ...} GPT가 누락한 고유명사를 일괄 치환
    
    Returns:
        enriched_segments: 키워드가 매핑된 세그먼트 목록
    """
    # 1. corrections 적용 (수정된 세그먼트만 덮어쓰기)
    corrections_dict = {}
    for c in combined_result.get("corrections", []):
        if isinstance(c, dict) and "id" in c and "text" in c:
            corrections_dict[c["id"]] = c["text"]
    
    corrected_count = 0
    for seg in ch_segs:
        seg_id = seg.get("id", seg.get("segment_id", 0))
        if seg_id in corrections_dict:
            old_text = seg.get("text", "")
            new_text = corrections_dict[seg_id]
            if old_text != new_text:
                seg["text"] = new_text
                corrected_count += 1
    
    if corrected_count > 0:
        print(f"[TADAC]   교정 적용: {corrected_count}개 세그먼트 수정됨")
    
    # 1.5. 고유명사 일괄 치환 (GPT가 누락한 교정을 보완)
    if name_corrections:
        name_fixed_count = 0
        for seg in ch_segs:
            text = seg.get("text", "")
            new_text = text
            for wrong, correct in name_corrections.items():
                if wrong in new_text:
                    new_text = new_text.replace(wrong, correct)
            if new_text != text:
                seg["text"] = new_text
                name_fixed_count += 1
        if name_fixed_count > 0:
            print(f"[TADAC]   고유명사 일괄 치환: {name_fixed_count}개 세그먼트 추가 수정")
    
    # 2. 교정 후 세그먼트 길이 재검증 — 55자 초과 시 분할
    from stt import MAX_SEGMENT_CHARS, MIN_SEGMENT_CHARS
    resplit = []
    for seg in ch_segs:
        text = seg.get("text", "")
        if len(text) <= MAX_SEGMENT_CHARS:
            resplit.append(seg)
            continue
        # 단어 경계에서 분할
        words = text.split()
        if not words:
            resplit.append(seg)
            continue
        duration = seg.get("end", 0.0) - seg.get("start", 0.0)
        total_w = len(words)
        parts, cur = [], []
        for w in words:
            candidate = " ".join(cur + [w]) if cur else w
            if len(candidate) > MAX_SEGMENT_CHARS and cur:
                parts.append(cur)
                cur = [w]
            else:
                cur.append(w)
        if cur:
            parts.append(cur)
        w_off = 0
        for pw in parts:
            pt = " ".join(pw)
            n = len(pw)
            ps = seg.get("start", 0.0) + duration * (w_off / total_w)
            pe = seg.get("start", 0.0) + duration * ((w_off + n) / total_w)
            resplit.append({
                "id": 0, "start": round(ps, 3), "end": round(pe, 3), "text": pt,
            })
            w_off += n
    if len(resplit) != len(ch_segs):
        print(f"[TADAC]   교정 후 재분할: {len(ch_segs)}개 → {len(resplit)}개 (max {MAX_SEGMENT_CHARS}자)")
        for i, seg in enumerate(resplit):
            seg["id"] = i
        ch_segs.clear()
        ch_segs.extend(resplit)

    # 3. segment_keywords → keyword_map 변환
    keyword_map = {}
    for sk in combined_result.get("segment_keywords", []):
        if isinstance(sk, dict) and "id" in sk:
            keyword_map[sk["id"]] = sk.get("keywords", [])

    total_keywords = sum(len(v) for v in keyword_map.values())
    print(f"[TADAC]   키워드 추출: {len(keyword_map)}개 세그먼트에서 {total_keywords}개 키워드")

    # 4. 키워드에 타임스탬프 매핑
    enriched = keyword_extractor.enrich_segments_with_keywords(ch_segs, keyword_map, all_words)

    return enriched


def _build_corrected_subtitle_data(enriched_segments):
    """교정이 적용된 세그먼트 자막을 빈칸 없이 송출하기 위한 JSON 구조 생성."""
    subtitles = []
    for seg in enriched_segments:
        subtitles.append({
            "segment_id": seg.get("segment_id", seg.get("id", len(subtitles))),
            "start":      seg.get("start", 0.0),
            "end":        seg.get("end", 0.0),
            "text":       seg.get("text", ""),
        })

    return {
        "subtitles": subtitles,
        "config": {
            "total_segments": len(subtitles),
        },
    }


def _compose_ai_summary(topic_summary, chapter_summaries):
    """
    챕터별 요약을 마크다운 복습 노트로 합친다.

    - chapter_summaries 가 있으면: topic_summary 한 줄 + 챕터별 ## 헤더 블록.
    - 없으면(refine=false / 수동 자막 / GPT가 빈 값을 줌): topic_summary 한 줄만.
    - 둘 다 없으면 빈 문자열.
    """
    parts = []
    if topic_summary:
        parts.append(topic_summary.strip())
    for title, body in chapter_summaries:
        parts.append(f"## {title}\n{body.strip()}")
    return "\n\n".join(parts)


def _generate_and_build_shorts(chapter_payloads, topic_summary=""):
    """챕터별 최종 자막 텍스트를 기반으로 숏폼 데이터를 만들고 영상을 렌더링한다."""
    shorts_data = []
    for payload in chapter_payloads:
        ch_idx = payload.get("chapter_index", len(shorts_data))
        ch_title = payload.get("chapter_title", f"Chapter {ch_idx + 1}")
        chapter_text = payload.get("chapter_text", "")

        print(f"[TADAC] 숏폼 생성 {ch_idx + 1}/{len(chapter_payloads)}: '{ch_title}'")
        result = shorts_generator.generate_shorts_prompt(
            chapter_title=ch_title,
            chapter_text=chapter_text,
            topic_summary=topic_summary,
        )
        if result:
            result["chapter_index"] = ch_idx
            result["chapter_title"] = ch_title
        shorts_data.append(result)

    shorts_output = [s for s in shorts_data if s is not None]
    print(f"[TADAC] 숏폼 생성 완료: {len(shorts_output)}/{len(chapter_payloads)}개 챕터")

    if shorts_output:
        print("[TADAC] 숏폼 영상 자동 빌드 시작")
        shorts_video_dir = os.path.join(str(Path(__file__).parent), "shorts_rendered")
        shorts_video_paths = shorts_builder.build_all_shorts_videos(
            shorts_output, output_dir=shorts_video_dir,
        )
        for ch_data, vpath in zip(shorts_output, shorts_video_paths):
            if vpath:
                ch_data["video_path"] = vpath

    return shorts_output


def _normalize_quiz(q):
    """GPT가 explanation 대신 correct_feedback/incorrect_feedback으로 반환할 때 통일"""
    cf = q.pop("correct_feedback", None)
    icf = q.pop("incorrect_feedback", None)
    if not q.get("explanation"):
        q["explanation"] = cf or icf or ""
    return q


def _format_time_range(start_sec, end_sec):
    return f"{start_sec / 60:.1f}분~{end_sec / 60:.1f}분"


def _build_chunk_chapters(chunk_ranges):
    chapters = []
    for idx, (offset_sec, end_sec) in enumerate(chunk_ranges):
        chapters.append({
            "title":     f"Part {idx + 1} ({_format_time_range(offset_sec, end_sec)})",
            "start_sec": round(offset_sec, 3),
            "end_sec":   round(end_sec, 3),
        })
    return chapters


def _apply_offset_and_ids(transcript_result, offset_sec, global_seg_id):
    """청크 STT 결과를 전체 타임라인 기준으로 보정하고 segment id를 이어 붙인다."""
    ch_segs = transcript_result.get("segments", [])
    ch_words = transcript_result.get("words", [])

    for seg in ch_segs:
        seg["start"] = round(seg.get("start", 0.0) + offset_sec, 3)
        seg["end"]   = round(seg.get("end", 0.0)   + offset_sec, 3)
        seg["id"]    = global_seg_id
        global_seg_id += 1

    for w in ch_words:
        w["start"] = round(w.get("start", 0.0) + offset_sec, 3)
        w["end"]   = round(w.get("end", 0.0)   + offset_sec, 3)

    return ch_segs, ch_words, global_seg_id


def _run_pipeline_sequential_chunk_streaming(
    audio_path,
    language,
    blanks_per_sentence,
    fall_speed,
    lead_time,
    stt_prompt,
    refine,
    generate_shorts,
    content_title,
    tmp_dirs,
    request_id=None,
):
    """CUDA 병렬 없이 10분 청크를 앞에서부터 STT/교정하여 먼저 흘려보내는 파이프라인."""
    _t_pipeline_start = _time.time()

    chunk_tmp_dir = tempfile.mkdtemp(prefix="tadac_seq_chunks_")
    tmp_dirs.append(chunk_tmp_dir)
    total_duration, chunk_ranges = _build_streaming_chunk_ranges(audio_path)
    chunk_chapters = _build_chunk_chapters(chunk_ranges)

    yield {
        "type":           "init",
        "total_duration": round(total_duration, 3),
        "chapters":       chunk_chapters,
        "topic_summary":  "",
        "streaming_mode": "sequential_chunk",
    }

    all_segments = []
    all_words = []
    total_enriched_segments = []
    shorts_chapter_payloads = []
    global_seg_id = 0
    log_prefix = f"[TADAC][{request_id}]" if request_id else "[TADAC]"

    for chunk_idx, (offset_sec, end_sec) in enumerate(chunk_ranges):
        chunk_meta = chunk_chapters[chunk_idx]
        _t_chunk_start = _time.time()
        print(f"{log_prefix} 순차 청크 {chunk_idx + 1}/{len(chunk_ranges)} STT 시작: {chunk_meta['title']}")
        chunk_path = _materialize_streaming_chunk(
            audio_path, chunk_tmp_dir, chunk_idx, offset_sec, end_sec
        )

        result = stt_module.transcribe(
            chunk_path,
            language=language,
            stt_prompt=stt_prompt,
            title=content_title,
        )
        ch_segs, ch_words, global_seg_id = _apply_offset_and_ids(result, offset_sec, global_seg_id)
        if not ch_segs:
            print(f"{log_prefix} 순차 청크 {chunk_idx + 1}: 세그먼트 없음")
            continue

        all_segments.extend(ch_segs)
        all_words.extend(ch_words)

        if refine:
            summary_hint = (
                f"영상 제목: {content_title or '알 수 없음'}\n"
                f"현재 청크: {chunk_meta['title']}\n"
                "전체 챕터 분석 전이므로 이 청크 안에서만 자막 교정과 빈칸 키워드를 처리한다."
            )
            combined_result = combined_processor.process_chapter_unified(
                ch_segs,
                summary_hint,
                chunk_meta["title"],
                blanks_per_sentence=blanks_per_sentence,
                global_keywords=None,
                questions_count=0,
            )
            ch_enriched = _apply_gpt_results(ch_segs, combined_result, ch_words, {})
            ch_summary = (combined_result.get("chapter_summary") or "").strip()
        else:
            ch_transcript = {
                "text": " ".join(seg.get("text", "") for seg in ch_segs),
                "words": ch_words,
                "segments": ch_segs,
                "language": language,
            }
            ch_enriched = keyword_extractor.extract_keywords(
                ch_transcript, blanks_per_sentence=blanks_per_sentence
            )
            ch_summary = ""

        total_enriched_segments.extend(ch_enriched)
        ch_game_data = blank_subtitle.build_game_data(
            ch_enriched,
            fall_speed=fall_speed,
            lead_time=lead_time,
        )
        ch_corrected = _build_corrected_subtitle_data(ch_enriched)
        if generate_shorts:
            shorts_chapter_payloads.append({
                "chapter_index": chunk_idx,
                "chapter_title": chunk_meta["title"],
                "chapter_text":  " ".join(seg.get("text", "") for seg in ch_enriched),
            })

        print(f"{log_prefix} 순차 청크 {chunk_idx + 1} 완료: {_time.time() - _t_chunk_start:.1f}초")
        yield {
            "type":                "chapter_ready",
            "chapter_index":       chunk_idx,
            "chapter_title":       chunk_meta["title"],
            "chapter_summary":     ch_summary,
            "corrected_subtitles": ch_corrected.get("subtitles", []),
            "subtitles":           ch_game_data.get("subtitles", []),
            "fall_events":         ch_game_data.get("fall_events", []),
            "quizzes":             [],
            "streaming_mode":      "sequential_chunk",
        }

    if not all_segments:
        raise ValueError("세그먼트가 없어 게임 데이터를 만들 수 없음")

    full_transcript = {
        "text":     " ".join(seg.get("text", "") for seg in all_segments),
        "words":    all_words,
        "segments": all_segments,
        "language": language,
    }
    _log_large_transcript_gaps(all_segments)
    _save_raw_transcript(full_transcript)

    print(f"{log_prefix} 전체 STT 완료 후 챕터/퀴즈 분석 시작")
    _t_analysis_start = _time.time()
    if refine:
        summary, chapters, name_corrections, global_keywords, topic_summary = (
            transcript_refiner._analyze_content(all_segments, title=content_title)
        )
    else:
        summary = ""
        name_corrections = {}
        global_keywords = []
        chapters, topic_summary = transcript_refiner.analyze_chapters_only(full_transcript)

    chapter_segments_map = quiz_generator._map_segments_to_chapters(all_segments, chapters)
    all_quizzes = []
    chapter_summaries = []

    for ch_idx, (chapter, ch_segs) in enumerate(zip(chapters, chapter_segments_map)):
        if not ch_segs:
            continue

        if refine and summary:
            combined_result = combined_processor.process_chapter_unified(
                ch_segs,
                summary,
                chapter["title"],
                blanks_per_sentence=blanks_per_sentence,
                global_keywords=global_keywords,
                questions_count=3,
            )
            ch_quizzes = combined_result.get("quizzes", [])
            ch_summary = (combined_result.get("chapter_summary") or "").strip()
            if ch_summary:
                chapter_summaries.append((chapter["title"], ch_summary))
        else:
            ch_quizzes = quiz_generator.generate_quizzes(ch_segs, chapters=[chapter])
            ch_summary = ""

        last_seg = ch_segs[-1] if ch_segs else {}
        trigger_time = last_seg.get("end", 0.0)
        seg_id_start = ch_segs[0].get("id", 0) if ch_segs else 0
        seg_id_end = last_seg.get("id", 0)

        for q in ch_quizzes:
            _normalize_quiz(q)
            q["ai_quiz_index"] = len(all_quizzes) + ch_quizzes.index(q)
            q["chapter_index"] = ch_idx
            q["chapter_title"] = chapter["title"]
            q["trigger_time"] = round(trigger_time, 3)
            q["segment_range"] = [seg_id_start, seg_id_end]

        all_quizzes.extend(ch_quizzes)

    print(f"{log_prefix} 챕터/퀴즈 분석 완료: {_time.time() - _t_analysis_start:.1f}초")
    yield {
        "type":          "analysis_ready",
        "chapters":      chapters,
        "quizzes":       all_quizzes,
        "ai_summary":    _compose_ai_summary(topic_summary, chapter_summaries),
        "topic_summary": topic_summary,
    }

    _t_total = _time.time() - _t_pipeline_start
    print(f"{log_prefix} 순차 청크 파이프라인 완료: {_t_total:.1f}초")
    yield {
        "type":       "complete",
        "ai_summary": _compose_ai_summary(topic_summary, chapter_summaries),
        "chapters":   chapters,
        "quizzes":    all_quizzes,
        "stats": {
            "transcript_source": "whisper",
            "total_words":       len(all_words),
            "language":          language,
            "gpt_refined":       refine,
            "total_quizzes":     len(all_quizzes),
            "total_segments":    len(total_enriched_segments),
            "streaming_mode":    "sequential_chunk",
        },
    }
    if generate_shorts:
        print(f"{log_prefix} 게임 데이터 스트리밍 완료 → 숏폼 생성 시작")
        shorts_output = _generate_and_build_shorts(shorts_chapter_payloads, topic_summary=topic_summary)
        yield {
            "type":   "shorts_ready",
            "shorts": shorts_output,
        }


def _branch_output_paths(output_path):
    """CLI -o 경로를 기준으로 두 브랜치 JSON 파일명을 만든다."""
    output_path = Path(output_path)
    suffix = output_path.suffix or ".json"
    base = output_path.with_suffix("")
    return {
        "corrected_subtitles": base.with_name(f"{base.name}_corrected_subtitles{suffix}"),
        "blank_game_data":     base.with_name(f"{base.name}_blank_game_data{suffix}"),
    }


# ── 썸네일 추출 ──────────────────────────────────────────────────────────────

def _get_youtube_thumbnail(url):
    """YouTube URL에서 썸네일 URL 반환 (maxresdefault → hqdefault 폴백)"""
    video_id = youtube_subtitle._extract_video_id(url)
    if not video_id:
        return None
    return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"


def _extract_local_thumbnail(video_path, tmp_dir):
    """로컬 비디오의 첫 프레임을 추출하여 base64 data URI로 반환"""
    thumb_path = os.path.join(tmp_dir, "thumbnail.jpg")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",
                thumb_path,
            ],
            capture_output=True,
            timeout=30,
        )
        if os.path.exists(thumb_path) and os.path.getsize(thumb_path) > 0:
            with open(thumb_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            print(f"[TADAC] 로컬 영상 썸네일 추출 완료")
            return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        print(f"[TADAC] 썸네일 추출 실패 (무시): {e}")
    return None


# ── 메인 파이프라인 ───────────────────────────────────────────────────────────
# Step 0: 입력 분기 → Step 1: STT → Step 1.5: 내용 분석 → Step 2: 통합 처리 → Step 3: 게임 데이터 생성
#
# blanks_per_sentence, fall_speed, lead_time 은 api.py 에서 고정값으로 전달됨.
# run_pipeline 파라미터로 남겨두는 이유: CLI 테스트 및 직접 임포트 시 유연성 확보.

def run_pipeline(
    source,
    language            = "ko",
    blanks_per_sentence = MAX_BLANKS_PER_SENTENCE,  # 기본값: 최대 빈칸
    fall_speed          = BASE_FALL_SPEED,
    lead_time           = BASE_LEAD_TIME,
    stt_prompt          = None,
    refine              = True,   # Whisper 결과를 GPT로 교정할지 여부
    return_branches     = False,  # True면 교정 자막 / 빈칸 게임 데이터 두 브랜치를 함께 반환
    generate_shorts     = False,  # True면 챕터별 숏폼 대본 + 영상 프롬프트 생성
):
    tmp_dirs = []  # 처리 완료 후 삭제할 임시 폴더 목록

    try:
        transcript        = {}
        transcript_source = "whisper"
        content_title     = None    # 영상/파일 제목 (교정 맥락용)
        thumbnail         = None    # 영상 썸네일 (URL 또는 base64 data URI)

        # ── Step 0: 입력 분기 ─────────────────────────────────────────────────
        if _is_youtube_url(source):
            print(f"[TADAC] 입력: YouTube URL")
            thumbnail = _get_youtube_thumbnail(source)

            # YouTube 수동 자막 추출 시도 (자동 자막은 품질 이슈로 사용 안 함)
            transcript, transcript_source = youtube_subtitle.get_transcript_from_youtube(
                source, preferred_lang=language
            )

            if not transcript.get("segments"):
                # 수동 자막 없음 → 오디오 추출 → 멀티 GPU 병렬 STT
                print("[TADAC] 수동 자막 없음 → 오디오 추출 후 멀티 GPU STT")
                tmp_dir = tempfile.mkdtemp(prefix="tadac_yt_audio_")
                tmp_dirs.append(tmp_dir)
                audio_path        = youtube_audio.extract_audio(source, tmp_dir)
                transcript        = _transcribe_auto_parallel(audio_path, language, stt_prompt, content_title, tmp_dirs)
                transcript_source = "whisper"

        else:
            # 로컬 파일
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"파일을 찾을 수 없음: {source}")

            # 파일명을 제목으로 사용
            content_title = path.stem  # 확장자 제외한 파일명

            suffix     = path.suffix.lower()
            audio_path = source

            if suffix in VIDEO_EXTENSIONS:
                # 비디오 파일 → 오디오 추출 먼저
                print(f"[TADAC] 입력: 로컬 비디오 파일 ({suffix})")
                tmp_dir = tempfile.mkdtemp(prefix="tadac_video_")
                tmp_dirs.append(tmp_dir)
                audio_path = _extract_audio_from_video(source, tmp_dir)
                thumbnail = _extract_local_thumbnail(source, tmp_dir)
            else:
                print(f"[TADAC] 입력: 로컬 오디오 파일 ({suffix})")

            transcript        = _transcribe_auto_parallel(audio_path, language, stt_prompt, content_title, tmp_dirs)
            transcript_source = "whisper"

        if not transcript.get("segments"):
            raise ValueError("세그먼트가 없어 게임 데이터를 만들 수 없음")

        # Whisper 원본 저장 (재처리 및 디버깅용)
        if transcript_source == "whisper":
            _log_large_transcript_gaps(transcript.get("segments", []))
            _save_raw_transcript(transcript)

        print(f"[TADAC] 트랜스크립트 준비 완료: 출처={transcript_source}, "
              f"세그먼트 {len(transcript['segments'])}개, 단어 {len(transcript.get('words', []))}개")

        # ── Step 1.5: 내용 분석 + 텍스트 교정 ─────────────────────────────────
        chapters = None
        summary  = None
        topic_summary = ""  # ai_summary 폴백/헤더용 한 줄 요약

        name_corrections = {}  # 고유명사 교정 사전
        global_keywords  = []  # 전체 강의 핵심 키워드 (빈칸 출제 풀)

        if transcript_source == "whisper" and refine:
            print("[TADAC] Whisper 원본 → GPT 내용 분석 (교정은 챕터별 통합처리)")
            summary, chapters, name_corrections, global_keywords, topic_summary = transcript_refiner._analyze_content(transcript.get("segments", []), title=content_title)

            # GPT 검수 패스: key_terms를 ground truth로 추가 STT 오인식 식별
            verified = transcript_refiner.verify_with_key_terms(transcript.get("segments", []), global_keywords)
            new_verified = {
                w: c for w, c in verified.items()
                if w not in name_corrections
                and transcript_refiner._is_safe_name_correction(w, c)
            }
            if new_verified:
                print(f"[TADAC] GPT 검수 패스 추가 교정: {len(new_verified)}개")
                for wrong, correct in new_verified.items():
                    print(f"  {wrong} → {correct}")
                name_corrections.update(new_verified)
        else:
            print("[TADAC] 교정 스킵 → 챕터 분할만 수행")
            chapters, topic_summary = transcript_refiner.analyze_chapters_only(transcript)

        # ── Step 2, 3, 4: 통합 처리 (교정 + 키워드 + 퀴즈) ─────────────────────
        all_segments = transcript.get("segments", [])
        chapter_segments_map = quiz_generator._map_segments_to_chapters(all_segments, chapters)

        all_enriched_segments = []
        enriched_by_chapter = []  # 숏폼 생성용 챕터별 교정 세그먼트
        all_quizzes = []
        chapter_summaries = []  # [(chapter_title, chapter_summary), ...] — ai_summary 합성용

        for ch_idx, (chapter, ch_segs) in enumerate(zip(chapters, chapter_segments_map)):
            if not ch_segs:
                continue
                
            print(f"[TADAC] 챕터 {ch_idx+1}/{len(chapters)}: '{chapter['title']}' 통합 처리 시작")
            
            if transcript_source == "whisper" and refine and summary:
                # 3-in-1 경량 API 호출 (corrections + segment_keywords + quizzes)
                combined_result = combined_processor.process_chapter_unified(
                    ch_segs, summary, chapter["title"],
                    blanks_per_sentence=blanks_per_sentence,
                    global_keywords=global_keywords,
                )

                # GPT 결과 후처리: corrections 적용 + 키워드 매핑
                ch_enriched = _apply_gpt_results(ch_segs, combined_result, transcript.get("words", []), name_corrections)

                # GPT가 놓친 키워드를 전역 목록에서 결정론적으로 보완
                ch_enriched = _fill_missing_keywords(ch_enriched, global_keywords, transcript.get("words", []))

                # 퀴즈 적용
                ch_quizzes = combined_result.get("quizzes", [])
                last_seg = ch_segs[-1] if ch_segs else {}
                trigger_time = last_seg.get("end", 0.0)
                seg_id_start = ch_segs[0].get("id", 0) if ch_segs else 0
                seg_id_end = last_seg.get("id", 0)

                for q in ch_quizzes:
                    _normalize_quiz(q)
                    q["ai_quiz_index"] = len(all_quizzes) + ch_quizzes.index(q)
                    q["chapter_index"] = ch_idx
                    q["chapter_title"] = chapter["title"]
                    q["trigger_time"] = round(trigger_time, 3)
                    q["segment_range"] = [seg_id_start, seg_id_end]

                all_quizzes.extend(ch_quizzes)
                all_enriched_segments.extend(ch_enriched)
                enriched_by_chapter.append(ch_enriched)

                # 챕터 복습 요약 수집 (GPT가 빈 문자열을 줄 수도 있음 → 그땐 스킵)
                ch_summary = (combined_result.get("chapter_summary") or "").strip()
                if ch_summary:
                    chapter_summaries.append((chapter["title"], ch_summary))

            else:
                # 수동 자막인 경우 기존 방식 사용 (교정 생략)
                ch_transcript = {
                    "text": " ".join(seg.get("text", "") for seg in ch_segs),
                    "words": transcript.get("words", []),
                    "segments": ch_segs,
                    "language": language,
                }
                ch_enriched = keyword_extractor.extract_keywords(
                    ch_transcript, blanks_per_sentence=blanks_per_sentence
                )

                ch_quizzes = quiz_generator.generate_quizzes(
                    ch_segs, chapters=[chapter]
                )
                for q in ch_quizzes:
                    _normalize_quiz(q)
                    q["ai_quiz_index"] = len(all_quizzes) + ch_quizzes.index(q)
                    q["chapter_index"] = ch_idx

                all_quizzes.extend(ch_quizzes)
                all_enriched_segments.extend(ch_enriched)
                enriched_by_chapter.append(ch_enriched)

        # ── Step 5-A: 교정 자막 브랜치 생성 ───────────────────────────────────
        corrected_subtitle_data = _build_corrected_subtitle_data(all_enriched_segments)

        # ── Step 5-B: 빈칸 게임 데이터 브랜치 생성 ─────────────────────────────
        game_data = blank_subtitle.build_game_data(
            all_enriched_segments,
            fall_speed=fall_speed,
            lead_time=lead_time,
        )
        game_data["quizzes"] = all_quizzes

        # 디버깅용 — 풀(key_terms)과 매칭 통계를 함께 저장
        # 풀이 작은지 / 풀에 있는데 매칭 실패인지 사후 분석 가능
        all_text = " ".join(seg.get("text", "") for seg in all_enriched_segments)
        pool_stats = []
        for term in global_keywords:
            occurrences = all_text.count(term)
            used = sum(
                1 for seg in all_enriched_segments
                for kw in seg.get("keywords", [])
                if (kw["keyword"] if isinstance(kw, dict) else kw) == term
            )
            pool_stats.append({
                "term":        term,
                "in_text":     occurrences,   # 자막 텍스트 안 등장 횟수
                "as_blank":    used,          # 실제 빈칸으로 뽑힌 횟수
            })

        # ai_summary 합성 — 챕터별 복습 요약을 마크다운 목차로 합치고,
        # 챕터 요약이 비어 있으면(refine=false / 수동 자막) topic_summary 한 줄로 폴백.
        game_data["ai_summary"] = _compose_ai_summary(topic_summary, chapter_summaries)

        if thumbnail:
            game_data["thumbnail"] = thumbnail

        # 파이프라인 메타데이터 추가
        game_data["stats"] = {
            "transcript_source": transcript_source,  # "whisper" / "youtube_manual"
            "total_words":       len(transcript.get("words", [])),
            "language":          language,
            "gpt_refined":       (transcript_source == "whisper" and refine),
            "total_quizzes":     len(all_quizzes),
        }
        game_data["debug"] = {
            "key_terms_pool":     global_keywords,   # 풀 전체
            "key_terms_pool_size": len(global_keywords),
            "name_corrections":   name_corrections,  # STT 교정 사전
            "pool_term_stats":    pool_stats,        # 풀 단어별 등장/사용 통계
        }

        corrected_subtitle_data["stats"] = game_data["stats"].copy()
        corrected_subtitle_data["debug"] = {
            "name_corrections": game_data["debug"]["name_corrections"],
        }

        shorts_payloads = []
        for ch_idx, (chapter, ch_enriched) in enumerate(zip(chapters, enriched_by_chapter)):
            shorts_payloads.append({
                "chapter_index": ch_idx,
                "chapter_title": chapter["title"],
                "chapter_text":  " ".join(seg.get("text", "") for seg in ch_enriched),
            })

        # ── Step 6: 게임 데이터 브랜치 완성 후 숏폼 생성 ─────────────────────
        shorts_output = []
        if generate_shorts:
            print("[TADAC] 게임 데이터 생성 완료 → 숏폼 생성 시작")
            shorts_output = _generate_and_build_shorts(shorts_payloads, topic_summary=topic_summary)

        if return_branches:
            result = {
                "corrected_subtitles": corrected_subtitle_data,
                "blank_game_data":     game_data,
                "shorts_payloads":     shorts_payloads,
            }
            if shorts_output:
                result["shorts"] = shorts_output
            return result

        if shorts_output:
            game_data["shorts"] = shorts_output

        return game_data

    finally:
        # 임시 파일 정리
        for d in tmp_dirs:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
                print(f"[TADAC] 임시 폴더 삭제: {d}")


# ── 스트리밍 파이프라인 ───────────────────────────────────────────────────────
# 챕터 단위로 처리 완료 즉시 yield하는 generator 함수
# SSE 엔드포인트에서 사용

def run_pipeline_streaming(
    source,
    language            = "ko",
    blanks_per_sentence = MAX_BLANKS_PER_SENTENCE,
    fall_speed          = BASE_FALL_SPEED,
    lead_time           = BASE_LEAD_TIME,
    stt_prompt          = None,
    refine              = True,
    generate_shorts     = False,
):
    """
    챕터 단위 스트리밍 파이프라인 (generator).

    Yields:
        dict — 이벤트 타입별 데이터
        - {"type": "init", "chapters": [...], "total_duration": float}
        - {"type": "chapter_ready", "chapter_index": int, "data": {...}}
        - {"type": "complete", "stats": {...}}
    """
    tmp_dirs = []

    try:
        transcript        = {}
        transcript_source = "whisper"
        content_title     = None
        thumbnail         = None

        # ── Phase A: 선행 작업 (전체 처리 필수) ───────────────────────────────
        # STT → 내용 분석 (챕터 경계 확정 필요)
        _t_pipeline_start = _time.time()

        if _is_youtube_url(source):
            print(f"[TADAC] [스트리밍] 입력: YouTube URL")
            thumbnail = _get_youtube_thumbnail(source)

            # YouTube 수동 자막 추출 시도 (자동 자막은 품질 이슈로 사용 안 함)
            transcript, transcript_source = youtube_subtitle.get_transcript_from_youtube(
                source, preferred_lang=language
            )

            if not transcript.get("segments"):
                # 수동 자막 없음 → 오디오 추출 → 멀티 GPU 병렬 STT
                print("[TADAC] [스트리밍] 수동 자막 없음 → 오디오 추출 후 멀티 GPU STT")
                tmp_dir = tempfile.mkdtemp(prefix="tadac_yt_audio_")
                tmp_dirs.append(tmp_dir)
                audio_path        = youtube_audio.extract_audio(source, tmp_dir)
                transcript        = _transcribe_auto_parallel(audio_path, language, stt_prompt, content_title, tmp_dirs)
                transcript_source = "whisper"
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"파일을 찾을 수 없음: {source}")

            content_title = path.stem
            suffix     = path.suffix.lower()
            audio_path = source

            if suffix in VIDEO_EXTENSIONS:
                print(f"[TADAC] [스트리밍] 입력: 로컬 비디오 ({suffix})")
                tmp_dir = tempfile.mkdtemp(prefix="tadac_video_")
                tmp_dirs.append(tmp_dir)
                audio_path = _extract_audio_from_video(source, tmp_dir)
                thumbnail = _extract_local_thumbnail(source, tmp_dir)
            else:
                print(f"[TADAC] [스트리밍] 입력: 로컬 오디오 ({suffix})")

            transcript        = _transcribe_auto_parallel(audio_path, language, stt_prompt, content_title, tmp_dirs)
            transcript_source = "whisper"

        if not transcript.get("segments"):
            raise ValueError("세그먼트가 없어 게임 데이터를 만들 수 없음")

        # Whisper 원본 저장 (재처리 및 디버깅용)
        if transcript_source == "whisper":
            _log_large_transcript_gaps(transcript.get("segments", []))
            _save_raw_transcript(transcript)

        _t_stt_done = _time.time()
        print(f"[TADAC] ⏱ STT 완료: {_t_stt_done - _t_pipeline_start:.1f}초")

        all_segments = transcript.get("segments", [])
        total_duration = max(seg.get("end", 0.0) for seg in all_segments)

        # 내용 분석 + 챕터 분할 (+ Whisper일 때 교정 맥락)
        chapters = None
        summary  = None
        topic_summary = ""  # ai_summary 폴백/헤더용 한 줄 요약

        name_corrections = {}  # 고유명사 교정 사전
        global_keywords  = []  # 전체 강의 핵심 키워드 (빈칸 출제 풀)

        if transcript_source == "whisper" and refine:
            print("[TADAC] [스트리밍] Whisper → 내용 분석 + 배치 교정")
            _t_analyze_start = _time.time()
            summary, chapters, name_corrections, global_keywords, topic_summary = transcript_refiner._analyze_content(all_segments, title=content_title)

            # GPT 검수 패스: key_terms를 ground truth로 추가 STT 오인식 식별
            verified = transcript_refiner.verify_with_key_terms(all_segments, global_keywords)
            new_verified = {
                w: c for w, c in verified.items()
                if w not in name_corrections
                and transcript_refiner._is_safe_name_correction(w, c)
            }
            if new_verified:
                print(f"[TADAC] GPT 검수 패스 추가 교정: {len(new_verified)}개")
                for wrong, correct in new_verified.items():
                    print(f"  {wrong} → {correct}")
                name_corrections.update(new_verified)
        else:
            _t_analyze_start = _time.time()
            print("[TADAC] [스트리밍] 교정 스킵 → 챕터 분할만")
            chapters, topic_summary = transcript_refiner.analyze_chapters_only(transcript)

        _t_analyze_done = _time.time()
        print(f"[TADAC] ⏱ 내용 분석 완료: {_t_analyze_done - _t_analyze_start:.1f}초")

        # ── init 이벤트: 챕터 목록 전달 ───────────────────────────────────────
        init_event = {
            "type":           "init",
            "total_duration": round(total_duration, 3),
            "chapters":       chapters,
            "topic_summary":  topic_summary,  # 한 줄 주제 — 프론트가 즉시 노출 가능
        }
        if thumbnail:
            init_event["thumbnail"] = thumbnail
        yield init_event

        # ── Phase B: 챕터별 스트리밍 처리 ─────────────────────────────────────
        chapter_segments_map = quiz_generator._map_segments_to_chapters(all_segments, chapters)

        all_quizzes = []
        total_enriched_segments = []
        chapter_summaries = []  # [(chapter_title, chapter_summary), ...] — complete에서 합성
        shorts_chapter_payloads = []

        for ch_idx, (chapter, ch_segs) in enumerate(zip(chapters, chapter_segments_map)):
            if not ch_segs:
                continue

            _t_ch_start = _time.time()
            print(f"[TADAC] [스트리밍] 챕터 {ch_idx+1}/{len(chapters)}: '{chapter['title']}' 통합 처리 시작")

            ch_summary = ""  # 챕터 복습 요약 (refine=true 경로에서만 채워짐)

            if transcript_source == "whisper" and refine and summary:
                # 3-in-1 경량 API 호출 (corrections + segment_keywords + quizzes)
                combined_result = combined_processor.process_chapter_unified(
                    ch_segs, summary, chapter["title"],
                    blanks_per_sentence=blanks_per_sentence,
                    global_keywords=global_keywords,
                )

                # GPT 결과 후처리: corrections 적용 + 키워드 매핑
                ch_enriched = _apply_gpt_results(ch_segs, combined_result, transcript.get("words", []), name_corrections)

                # GPT가 놓친 키워드를 전역 목록에서 결정론적으로 보완
                ch_enriched = _fill_missing_keywords(ch_enriched, global_keywords, transcript.get("words", []))

                # 퀴즈 적용
                ch_quizzes = combined_result.get("quizzes", [])
                last_seg = ch_segs[-1] if ch_segs else {}
                trigger_time = last_seg.get("end", 0.0)
                seg_id_start = ch_segs[0].get("id", 0) if ch_segs else 0
                seg_id_end = last_seg.get("id", 0)

                for q in ch_quizzes:
                    _normalize_quiz(q)
                    q["ai_quiz_index"] = len(all_quizzes) + ch_quizzes.index(q)
                    q["chapter_index"] = ch_idx
                    q["chapter_title"] = chapter["title"]
                    q["trigger_time"] = round(trigger_time, 3)
                    q["segment_range"] = [seg_id_start, seg_id_end]

                ch_summary = (combined_result.get("chapter_summary") or "").strip()
                if ch_summary:
                    chapter_summaries.append((chapter["title"], ch_summary))

            else:
                # 기존 분리 호출 (수동 자막)
                ch_transcript = {
                    "text": " ".join(seg.get("text", "") for seg in ch_segs),
                    "words": transcript.get("words", []),
                    "segments": ch_segs,
                    "language": language,
                }
                ch_enriched = keyword_extractor.extract_keywords(ch_transcript, blanks_per_sentence=blanks_per_sentence)

                ch_quizzes = quiz_generator.generate_quizzes(ch_segs, chapters=[chapter])
                for q in ch_quizzes:
                    _normalize_quiz(q)
                    q["ai_quiz_index"] = len(all_quizzes) + ch_quizzes.index(q)
                    q["chapter_index"] = ch_idx

            total_enriched_segments.extend(ch_enriched)
            all_quizzes.extend(ch_quizzes)
            if generate_shorts:
                shorts_chapter_payloads.append({
                    "chapter_index": ch_idx,
                    "chapter_title": chapter["title"],
                    "chapter_text":  " ".join(seg.get("text", "") for seg in ch_enriched),
                })

            # 게임 데이터 생성 (챕터 분)
            ch_game_data = blank_subtitle.build_game_data(
                ch_enriched,
                fall_speed=fall_speed,
                lead_time=lead_time,
            )
            ch_corrected_subtitle_data = _build_corrected_subtitle_data(ch_enriched)

            # ── chapter_ready 이벤트 ──────────────────────────────────────────
            event = {
                "type":                "chapter_ready",
                "chapter_index":       ch_idx,
                "chapter_title":       chapter["title"],
                "chapter_summary":     ch_summary,
                "corrected_subtitles": ch_corrected_subtitle_data.get("subtitles", []),
                "subtitles":           ch_game_data.get("subtitles", []),
                "fall_events":         ch_game_data.get("fall_events", []),
                "quizzes":             ch_quizzes,
            }

            print(f"[TADAC] ⏱ 챕터 {ch_idx+1} 완료: {_time.time() - _t_ch_start:.1f}초")
            yield event

        # ── complete 이벤트 ───────────────────────────────────────────────────
        _t_total = _time.time() - _t_pipeline_start
        print(f"[TADAC] ⏱ 전체 파이프라인 완료: {_t_total:.1f}초 (STT 제외 후처리: {_t_total - (_t_stt_done - _t_pipeline_start):.1f}초)")
        yield {
            "type":       "complete",
            "ai_summary": _compose_ai_summary(topic_summary, chapter_summaries),
            "stats": {
                "transcript_source": transcript_source,
                "total_words":       len(transcript.get("words", [])),
                "language":          language,
                "gpt_refined":       (transcript_source == "whisper" and refine),
                "total_quizzes":     len(all_quizzes),
                "total_segments":    len(total_enriched_segments),
            },
        }
        if generate_shorts:
            print("[TADAC] 게임 데이터 스트리밍 완료 → 숏폼 생성 시작")
            shorts_output = _generate_and_build_shorts(shorts_chapter_payloads, topic_summary=topic_summary)
            yield {
                "type":   "shorts_ready",
                "shorts": shorts_output,
            }

    finally:
        for d in tmp_dirs:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
                print(f"[TADAC] 임시 폴더 삭제: {d}")


# ── 챕터 스트리밍 파이프라인 ─────────────────────────────────────────────────
# STT 완료 후 챕터 기반 후처리
# 기본 STT는 단일 파일 처리이며, env로 청크 병렬화를 켠 경우에만 10분 단위로 분할한다.
# 기존 run_pipeline_streaming과 동일한 이벤트 형태 (chapter_ready에 퀴즈 포함)

def run_pipeline_chunked_streaming(
    source,
    language            = "ko",
    blanks_per_sentence = MAX_BLANKS_PER_SENTENCE,
    fall_speed          = BASE_FALL_SPEED,
    lead_time           = BASE_LEAD_TIME,
    stt_prompt          = None,
    refine              = True,
    generate_shorts     = False,
    request_id          = None,
):
    """
    STT + 챕터 기반 스트리밍 파이프라인 (generator).

    기본은 단일 파일 STT로 안정성을 우선한다.
    STT_CHUNKING_ENABLED=true이고 영상이 threshold보다 길 때만
    10분 chunk 병렬 STT를 사용한다. 후처리는 기존과 동일하게 챕터 단위로
    교정+키워드+퀴즈를 포함하여 스트리밍한다.

    흐름:
      1. 오디오 STT 수행 (기본 단일 파일, 옵션 청크 병렬)
      2. 전체 STT 합치기 → 내용 분석 + 챕터 분할
      3. 챕터별 교정+키워드+퀴즈 → chapter_ready 즉시 스트리밍

    Yields:
        - {"type": "init", "chapters": [...], "total_duration": float}
        - {"type": "chapter_ready", ...}  (자막 + 퀴즈 포함)
        - {"type": "complete", "stats": {...}}
    """
    tmp_dirs = []

    try:
        _t_pipeline_start = _time.time()

        transcript_source = "whisper"
        content_title     = None
        audio_path        = None

        # ── Phase 0: 입력 분기 + 오디오 준비 ─────────────────────────────────
        if _is_youtube_url(source):
            print(f"[TADAC] [chunked] 입력: YouTube URL")

            # YouTube 수동 자막 시도
            transcript, transcript_source = youtube_subtitle.get_transcript_from_youtube(
                source, preferred_lang=language
            )

            if transcript.get("segments"):
                # 수동 자막 있음 → 기존 파이프라인으로 폴백
                print("[TADAC] [chunked] 수동 자막 발견 → 기존 스트리밍으로 폴백")
                yield from run_pipeline_streaming(
                    source=source, language=language,
                    blanks_per_sentence=blanks_per_sentence,
                    fall_speed=fall_speed, lead_time=lead_time,
                    stt_prompt=stt_prompt, refine=refine,
                    generate_shorts=generate_shorts,
                )
                return

            # 수동 자막 없음 → 오디오 추출
            print("[TADAC] [chunked] 수동 자막 없음 → 오디오 추출")
            tmp_dir = tempfile.mkdtemp(prefix="tadac_yt_audio_")
            tmp_dirs.append(tmp_dir)
            audio_path = youtube_audio.extract_audio(source, tmp_dir)

        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"파일을 찾을 수 없음: {source}")

            content_title = path.stem
            suffix = path.suffix.lower()
            audio_path = source

            if suffix in VIDEO_EXTENSIONS:
                print(f"[TADAC] [chunked] 입력: 로컬 비디오 ({suffix})")
                tmp_dir = tempfile.mkdtemp(prefix="tadac_video_")
                tmp_dirs.append(tmp_dir)
                audio_path = _extract_audio_from_video(source, tmp_dir)
            else:
                print(f"[TADAC] [chunked] 입력: 로컬 오디오 ({suffix})")

        if STT_SEQUENTIAL_CHUNKING_ENABLED:
            print("[TADAC] [chunked] 순차 청크 스트리밍 모드 사용")
            yield from _run_pipeline_sequential_chunk_streaming(
                audio_path           = audio_path,
                language             = language,
                blanks_per_sentence  = blanks_per_sentence,
                fall_speed           = fall_speed,
                lead_time            = lead_time,
                stt_prompt           = stt_prompt,
                refine               = refine,
                generate_shorts      = generate_shorts,
                content_title        = content_title,
                tmp_dirs             = tmp_dirs,
                request_id           = request_id,
            )
            return

        # ── Phase 1: STT (기본 단일 파일, 옵션 청크 병렬) ───────────────────
        full_transcript = _transcribe_auto_parallel(
            audio_path, language, stt_prompt, content_title, tmp_dirs, request_id=request_id
        )
        all_segments = full_transcript.get("segments", [])
        all_words    = full_transcript.get("words", [])

        if not all_segments:
            raise ValueError("세그먼트가 없어 게임 데이터를 만들 수 없음")

        _t_stt_total = _time.time() - _t_pipeline_start
        print(f"\n[TADAC] ⏱ 전체 STT 완료: {_t_stt_total:.1f}초 (세그먼트 {len(all_segments)}개, 단어 {len(all_words)}개)")
        if request_id:
            first_text = (all_segments[0].get("text", "") if all_segments else "")[:120]
            last_text = (all_segments[-1].get("text", "") if all_segments else "")[:120]
            print(f"[TADAC][{request_id}] STT 샘플 처음: {first_text}")
            print(f"[TADAC][{request_id}] STT 샘플 끝: {last_text}")

        _log_large_transcript_gaps(all_segments)

        # Whisper 원본 저장
        _save_raw_transcript(full_transcript)

        total_duration = max(seg.get("end", 0.0) for seg in all_segments)

        # ── Phase 2: 내용 분석 + 챕터 분할 ───────────────────────────────────
        # 기존 run_pipeline_streaming의 Phase A와 동일
        chapters = None
        summary  = None
        topic_summary    = ""
        name_corrections = {}
        global_keywords  = []

        if transcript_source == "whisper" and refine:
            _t_analyze_start = _time.time()
            print("[TADAC] [chunked] 내용 분석 + 챕터 분할")
            summary, chapters, name_corrections, global_keywords, topic_summary = (
                transcript_refiner._analyze_content(all_segments, title=content_title)
            )

            # GPT 검수 패스
            verified = transcript_refiner.verify_with_key_terms(all_segments, global_keywords)
            new_verified = {
                w: c for w, c in verified.items()
                if w not in name_corrections
                and transcript_refiner._is_safe_name_correction(w, c)
            }
            if new_verified:
                print(f"[TADAC] GPT 검수 패스 추가 교정: {len(new_verified)}개")
                for wrong, correct in new_verified.items():
                    print(f"  {wrong} → {correct}")
                name_corrections.update(new_verified)

            _t_analyze_end = _time.time()
            print(f"[TADAC] ⏱ 내용 분석 완료: {_t_analyze_end - _t_analyze_start:.1f}초")
        else:
            _t_analyze_start = _time.time()
            print("[TADAC] [chunked] 교정 스킵 → 챕터 분할만")
            chapters, topic_summary = transcript_refiner.analyze_chapters_only(full_transcript)
            _t_analyze_end = _time.time()
            print(f"[TADAC] ⏱ 챕터 분할 완료: {_t_analyze_end - _t_analyze_start:.1f}초")

        # ── init 이벤트: 챕터 목록 전달 ──────────────────────────────────────
        init_event = {
            "type":           "init",
            "total_duration": round(total_duration, 3),
            "chapters":       chapters,
            "topic_summary":  topic_summary,
        }
        yield init_event

        # ── Phase 3: 챕터별 교정+키워드+퀴즈 → 스트리밍 ──────────────────────
        # 기존 run_pipeline_streaming의 Phase B와 동일
        chapter_segments_map = quiz_generator._map_segments_to_chapters(all_segments, chapters)

        all_quizzes = []
        total_enriched_segments = []
        chapter_summaries = []
        shorts_chapter_payloads = []

        for ch_idx, (chapter, ch_segs) in enumerate(zip(chapters, chapter_segments_map)):
            if not ch_segs:
                continue

            _t_ch_start = _time.time()
            print(f"[TADAC] [chunked] 챕터 {ch_idx+1}/{len(chapters)}: '{chapter['title']}' 통합 처리 시작")

            ch_summary = ""

            if transcript_source == "whisper" and refine and summary:
                # 3-in-1: corrections + segment_keywords + quizzes
                combined_result = combined_processor.process_chapter_unified(
                    ch_segs, summary, chapter["title"],
                    blanks_per_sentence=blanks_per_sentence,
                    global_keywords=global_keywords,
                )

                ch_enriched = _apply_gpt_results(ch_segs, combined_result, all_words, name_corrections)
                ch_enriched = _fill_missing_keywords(ch_enriched, global_keywords, all_words)

                # 퀴즈
                ch_quizzes = combined_result.get("quizzes", [])
                last_seg = ch_segs[-1] if ch_segs else {}
                trigger_time = last_seg.get("end", 0.0)
                seg_id_start = ch_segs[0].get("id", 0) if ch_segs else 0
                seg_id_end = last_seg.get("id", 0)

                for q in ch_quizzes:
                    _normalize_quiz(q)
                    q["ai_quiz_index"] = len(all_quizzes) + ch_quizzes.index(q)
                    q["chapter_index"] = ch_idx
                    q["chapter_title"] = chapter["title"]
                    q["trigger_time"] = round(trigger_time, 3)
                    q["segment_range"] = [seg_id_start, seg_id_end]

                ch_summary = (combined_result.get("chapter_summary") or "").strip()
                if ch_summary:
                    chapter_summaries.append((chapter["title"], ch_summary))

            else:
                # 수동 자막 경로
                ch_transcript = {
                    "text": " ".join(seg.get("text", "") for seg in ch_segs),
                    "words": all_words,
                    "segments": ch_segs,
                    "language": language,
                }
                ch_enriched = keyword_extractor.extract_keywords(
                    ch_transcript, blanks_per_sentence=blanks_per_sentence
                )
                ch_quizzes = quiz_generator.generate_quizzes(ch_segs, chapters=[chapter])
                for q in ch_quizzes:
                    _normalize_quiz(q)
                    q["ai_quiz_index"] = len(all_quizzes) + ch_quizzes.index(q)
                    q["chapter_index"] = ch_idx

            total_enriched_segments.extend(ch_enriched)
            all_quizzes.extend(ch_quizzes)
            if generate_shorts:
                shorts_chapter_payloads.append({
                    "chapter_index": ch_idx,
                    "chapter_title": chapter["title"],
                    "chapter_text":  " ".join(seg.get("text", "") for seg in ch_enriched),
                })

            # 게임 데이터 생성
            ch_game_data = blank_subtitle.build_game_data(
                ch_enriched, fall_speed=fall_speed, lead_time=lead_time,
            )
            ch_corrected = _build_corrected_subtitle_data(ch_enriched)

            # chapter_ready 이벤트 (자막 + 퀴즈 포함)
            event = {
                "type":                "chapter_ready",
                "chapter_index":       ch_idx,
                "chapter_title":       chapter["title"],
                "chapter_summary":     ch_summary,
                "corrected_subtitles": ch_corrected.get("subtitles", []),
                "subtitles":           ch_game_data.get("subtitles", []),
                "fall_events":         ch_game_data.get("fall_events", []),
                "quizzes":             ch_quizzes,
            }

            print(f"[TADAC] ⏱ 챕터 {ch_idx+1} 완료: {_time.time() - _t_ch_start:.1f}초")
            yield event

        # ── complete 이벤트 ──────────────────────────────────────────────────
        _t_total = _time.time() - _t_pipeline_start
        print(f"[TADAC] ⏱ 전체 파이프라인 완료: {_t_total:.1f}초 (STT: {_t_stt_total:.1f}초, 후처리: {_t_total - _t_stt_total:.1f}초)")

        yield {
            "type":       "complete",
            "ai_summary": _compose_ai_summary(topic_summary, chapter_summaries),
            "stats": {
                "transcript_source": transcript_source,
                "total_words":       len(all_words),
                "language":          language,
                "gpt_refined":       (transcript_source == "whisper" and refine),
                "total_quizzes":     len(all_quizzes),
                "total_segments":    len(total_enriched_segments),
            },
        }
        if generate_shorts:
            print("[TADAC] 게임 데이터 스트리밍 완료 → 숏폼 생성 시작")
            shorts_output = _generate_and_build_shorts(shorts_chapter_payloads, topic_summary=topic_summary)
            yield {
                "type":   "shorts_ready",
                "shorts": shorts_output,
            }

    finally:
        for d in tmp_dirs:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
                print(f"[TADAC] 임시 폴더 삭제: {d}")


# ── CLI 실행 ──────────────────────────────────────────────────────────────────
# python pipeline.py <source> [옵션]
# fall_speed, lead_time, blanks 는 프론트가 관리하므로 CLI에서 제거

def main():
    parser = argparse.ArgumentParser(
        description="TADAC AI 파이프라인 — 빈칸 자막 게임 데이터 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source",         help="YouTube URL 또는 로컬 파일 경로")
    parser.add_argument("-o", "--output",  default="game_data.json", help="출력 JSON 파일명 (기본값: game_data.json)")
    parser.add_argument("--lang",          default="ko",              help="STT 언어 코드 (기본값: ko)")
    parser.add_argument("--prompt",        default=None,              help="STT 전문 용어 힌트")
    parser.add_argument("--no-refine",     action="store_true",       help="GPT 교정 스킵 (API 비용 절약)")
    parser.add_argument("--shorts",        action="store_true",       help="챕터별 숏폼 대본 + 영상 프롬프트 생성")
    parser.add_argument("--stream",        action="store_true",       help="스트리밍 모드 (챕터별 출력, 기본 단일 STT)")
    args = parser.parse_args()

    print(f"[TADAC] 파이프라인 시작")
    print(f"[TADAC] 소스: {args.source}")
    print(f"[TADAC] 설정: lang={args.lang}, blanks={MAX_BLANKS_PER_SENTENCE}(고정), refine={not args.no_refine}, shorts={args.shorts}")

    if args.stream:
        # 스트리밍 모드: STT + 챕터별 교정+키워드+퀴즈
        print("[TADAC] 스트리밍 모드 (기본 단일 STT)")

        aggregated_corrected_subtitle_data = {
            "subtitles": [],
            "config": {"total_segments": 0},
        }
        aggregated_blank_game_data = {
            "subtitles": [],
            "fall_events": [],
            "quizzes": [],
            "config": {
                "fall_speed": BASE_FALL_SPEED,
                "lead_time": BASE_LEAD_TIME,
                "blanks_per_sentence": MAX_BLANKS_PER_SENTENCE,
            },
        }
        aggregated_shorts = []
        stream_shorts_payloads = []
        stream_topic_summary = ""

        for event in run_pipeline_chunked_streaming(
            source          = args.source,
            language        = args.lang,
            stt_prompt      = args.prompt,
            refine          = not args.no_refine,
            generate_shorts = False,
        ):
            print(f"\n[TADAC] === 이벤트: {event['type']} ===")
            print(json.dumps(event, ensure_ascii=False, indent=2)[:500])

            if event["type"] == "init":
                stream_topic_summary = event.get("topic_summary", stream_topic_summary)
            elif event["type"] == "chapter_ready":
                corrected_subtitles = event.get("corrected_subtitles", [])
                aggregated_corrected_subtitle_data["subtitles"].extend(corrected_subtitles)
                aggregated_blank_game_data["subtitles"].extend(event.get("subtitles", []))
                aggregated_blank_game_data["fall_events"].extend(event.get("fall_events", []))
                aggregated_blank_game_data["quizzes"].extend(event.get("quizzes", []))
                if args.shorts:
                    stream_shorts_payloads.append({
                        "chapter_index": event.get("chapter_index", len(stream_shorts_payloads)),
                        "chapter_title": event.get("chapter_title", f"Chapter {len(stream_shorts_payloads) + 1}"),
                        "chapter_text":  " ".join(sub.get("text", "") for sub in corrected_subtitles),
                    })
            elif event["type"] == "analysis_ready":
                aggregated_blank_game_data["chapters"] = event.get("chapters", [])
                aggregated_blank_game_data["quizzes"] = event.get("quizzes", [])
                aggregated_blank_game_data["ai_summary"] = event.get("ai_summary", "")
                stream_topic_summary = event.get("topic_summary", stream_topic_summary)
            elif event["type"] == "complete":
                aggregated_corrected_subtitle_data["stats"] = event.get("stats", {})
                aggregated_blank_game_data["stats"] = event.get("stats", {})
                aggregated_blank_game_data["ai_summary"] = event.get("ai_summary", "")
                if event.get("chapters"):
                    aggregated_blank_game_data["chapters"] = event.get("chapters", [])
                if event.get("quizzes"):
                    aggregated_blank_game_data["quizzes"] = event.get("quizzes", [])
            elif event["type"] == "shorts_ready":
                aggregated_shorts = event.get("shorts", [])

        print("[TADAC] 스트리밍 완료")

        aggregated_corrected_subtitle_data["config"]["total_segments"] = len(
            aggregated_corrected_subtitle_data["subtitles"]
        )
        aggregated_blank_game_data["config"]["total_segments"] = len(
            aggregated_blank_game_data["subtitles"]
        )
        aggregated_blank_game_data["config"]["total_blanks"] = sum(
            len(sub.get("blanks", [])) for sub in aggregated_blank_game_data["subtitles"]
        )

        output_paths = _branch_output_paths(args.output)
        output_paths["corrected_subtitles"].write_text(
            json.dumps(aggregated_corrected_subtitle_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output_paths["blank_game_data"].write_text(
            json.dumps(aggregated_blank_game_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[TADAC] 교정 자막 저장: {output_paths['corrected_subtitles'].resolve()}")
        print(f"[TADAC] 빈칸 게임 데이터 저장: {output_paths['blank_game_data'].resolve()}")
        if args.shorts and not aggregated_shorts:
            print("[TADAC] 게임 데이터 파일 저장 완료 → 숏폼 생성 시작")
            aggregated_shorts = _generate_and_build_shorts(
                stream_shorts_payloads,
                topic_summary=stream_topic_summary,
            )
        if aggregated_shorts:
            shorts_path = Path(__file__).parent / "shorts_output.json"
            shorts_path.write_text(
                json.dumps(aggregated_shorts, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[TADAC] 숏폼 데이터 저장 완료: {shorts_path.resolve()}")

    else:
        # 일괄 모드
        branch_data = run_pipeline(
            source          = args.source,
            language        = args.lang,
            stt_prompt      = args.prompt,
            refine          = not args.no_refine,
            return_branches = True,
            generate_shorts = False,
        )
        corrected_subtitle_data = branch_data["corrected_subtitles"]
        game_data = branch_data["blank_game_data"]

        output_paths = _branch_output_paths(args.output)
        output_paths["corrected_subtitles"].write_text(
            json.dumps(corrected_subtitle_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output_paths["blank_game_data"].write_text(
            json.dumps(game_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"[TADAC] 교정 자막 저장 완료: {output_paths['corrected_subtitles'].resolve()}")
        print(f"[TADAC] 빈칸 게임 데이터 저장 완료: {output_paths['blank_game_data'].resolve()}")

        if args.shorts:
            print("[TADAC] 게임 데이터 파일 저장 완료 → 숏폼 생성 시작")
            branch_data["shorts"] = _generate_and_build_shorts(
                branch_data.get("shorts_payloads", []),
                topic_summary=game_data.get("ai_summary", ""),
            )

        if branch_data.get("shorts"):
            shorts_path = Path(__file__).parent / "shorts_output.json"
            shorts_path.write_text(
                json.dumps(branch_data["shorts"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[TADAC] 숏폼 데이터 저장 완료: {shorts_path.resolve()}")

        print(f"[TADAC] 결과: 세그먼트 {game_data['config']['total_segments']}개, "
              f"빈칸 {game_data['config']['total_blanks']}개, "
              f"낙하 이벤트 {len(game_data['fall_events'])}개")


if __name__ == "__main__":
    main()
