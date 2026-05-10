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
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 같은 폴더의 모듈 임포트
sys.path.insert(0, str(Path(__file__).parent))

import stt as stt_module
import transcript_refiner
import keyword_extractor
import blank_subtitle
import quiz_generator
import youtube_subtitle
import youtube_audio
import combined_processor

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


# ── YouTube 영상 제목+설명 추출 ───────────────────────────────────────────────
# yt-dlp로 영상 메타데이터에서 제목과 설명을 가져오기

def _get_youtube_metadata(youtube_url):
    """YouTube 영상 제목 + 설명 추출"""
    title = None
    description = None

    # OAuth2가 더 이상 지원되지 않으므로, 모바일 클라이언트로 위장하여 봇 우회
    bypass_args = ["--extractor-args", "youtube:player_client=android"]

    try:
        # 제목 추출
        cmd = ["yt-dlp", "--no-playlist", "--get-title", "--skip-download"] + bypass_args + [youtube_url]
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            title = proc.stdout.strip()
            print(f"[TADAC] 영상 제목: {title}")
    except Exception as e:
        print(f"[TADAC] 제목 추출 실패: {e}")

    try:
        # 설명 추출
        cmd = ["yt-dlp", "--no-playlist", "--get-description", "--skip-download"] + bypass_args + [youtube_url]
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            description = proc.stdout.strip()
            print(f"[TADAC] 영상 설명: {description[:100]}...")
    except Exception as e:
        print(f"[TADAC] 설명 추출 실패: {e}")

    return title, description



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


# ── 유사 오인식 자동 탐지 ─────────────────────────────────────────────────────
# key_terms 목록을 정답 기준으로 삼아, 텍스트에서 1글자만 다른 오인식을 찾아냄
# 예: key_term="문벌귀족" vs 텍스트 내 "문벌기족" → 자동 교정 대상

def _build_fuzzy_corrections(segments, key_terms):
    """key_terms를 ground truth로 삼아 STT 오인식을 결정론적으로 찾아 교정.

    설계 원칙:
    1. 길이 — 2글자 단어는 오탐 위험이 커서 3글자 이상만 대상.
    2. 띄어쓰기 비대칭 — 풀 term이 공백을 포함해도 STT는 보통 공백 없이 받아쓰므로
       풀 term의 공백 제거 변형도 매칭 후보에 포함.
    3. 빈도 비교 — 풀 term이 자막에 정확히 등장해도, 변형이 정답보다 더 자주 나오면
       그 변형을 STT 오타로 등록. (정답이 한두 번 끼어 있는 경우에 대응)
    4. diff 허용치 — 1글자 차이는 항상 허용. 2글자 차이는 4글자 이상 단어에서만 허용
       (짧은 단어에서 2자 차이는 별개 entity일 가능성이 높음).
    5. entity 스왑 보호 — 후보가 다른 풀 term과 일치하면 별개 단어이므로 skip.
    """
    all_text = " ".join(seg.get("text", "") for seg in segments)

    # 매칭 후보: 풀 term과 그 공백 제거 변형 (원본 term으로 매핑 보존)
    term_variants = {}  # variant 표기 → 자막 치환 시 적용할 원본 term
    for term in key_terms:
        if len(term) < 3:
            continue
        term_variants[term] = term
        no_space = term.replace(" ", "")
        if no_space != term and len(no_space) >= 3:
            term_variants.setdefault(no_space, term)

    all_variants = set(term_variants.keys())
    key_terms_set = set(key_terms)
    corrections = {}

    for variant, original in term_variants.items():
        v_len = len(variant)
        variant_count = all_text.count(variant)

        seen = set()
        for i in range(len(all_text) - v_len + 1):
            candidate = all_text[i:i + v_len]
            if candidate in seen or candidate == variant:
                continue
            seen.add(candidate)
            # 별개 entity 보호 — 후보가 풀의 다른 단어/변형이면 skip
            if candidate in all_variants or candidate in key_terms_set:
                continue

            diff_count = sum(1 for a, b in zip(variant, candidate) if a != b)
            if diff_count == 0:
                continue
            if not (diff_count == 1 or (diff_count == 2 and v_len >= 4)):
                continue

            # 빈도 검증 — 정답이 자막에 없거나 후보가 더 자주 등장해야 STT 오타로 인정
            cand_count = all_text.count(candidate)
            if variant_count == 0 or cand_count > variant_count:
                corrections[candidate] = original

    return corrections


# ── 키워드 폴백 보완 ─────────────────────────────────────────────────────────
# GPT가 놓친 키워드를 global_keywords 문자열 매칭으로 채움

def _fill_missing_keywords(enriched_segments, global_keywords, all_words, max_per_segment=2):
    """GPT가 놓친 키워드를 전역 목록에서 문자열 매칭으로 보완"""
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
                word_info = keyword_extractor._find_word_in_segment(term, seg, all_words)
                if word_info:
                    seg["keywords"].append({
                        "keyword": term,
                        "start": word_info["start"],
                        "end": word_info["end"],
                        "found": True,
                    })
                else:
                    mid = (seg.get("start", 0.0) + seg.get("end", 0.0)) / 2
                    seg["keywords"].append({
                        "keyword": term,
                        "start": mid,
                        "end": mid + 0.5,
                        "found": False,
                    })
                existing_kws.append(term)
                remaining -= 1
                filled_count += 1

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
        all_words: Whisper word timestamps
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
    
    # 2. segment_keywords → keyword_map 변환
    keyword_map = {}
    for sk in combined_result.get("segment_keywords", []):
        if isinstance(sk, dict) and "id" in sk:
            keyword_map[sk["id"]] = sk.get("keywords", [])
    
    total_keywords = sum(len(v) for v in keyword_map.values())
    print(f"[TADAC]   키워드 추출: {len(keyword_map)}개 세그먼트에서 {total_keywords}개 키워드")
    
    # 3. 키워드에 타임스탬프 매핑
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


def _branch_output_paths(output_path):
    """CLI -o 경로를 기준으로 두 브랜치 JSON 파일명을 만든다."""
    output_path = Path(output_path)
    suffix = output_path.suffix or ".json"
    base = output_path.with_suffix("")
    return {
        "corrected_subtitles": base.with_name(f"{base.name}_corrected_subtitles{suffix}"),
        "blank_game_data":     base.with_name(f"{base.name}_blank_game_data{suffix}"),
    }


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
):
    tmp_dirs = []  # 처리 완료 후 삭제할 임시 폴더 목록

    try:
        transcript        = {}
        transcript_source = "whisper"
        content_title     = None    # 영상/파일 제목 (교정 맥락용)

        # ── Step 0: 입력 분기 ─────────────────────────────────────────────────
        if _is_youtube_url(source):
            print(f"[TADAC] 입력: YouTube URL")

            # 영상 메타데이터 추출 (제목 + 설명)
            content_title, description = _get_youtube_metadata(source)

            # YouTube 자막 확인
            sub_info = youtube_subtitle.check_subtitles(source)

            if sub_info["has_manual"]:
                # ✅ 수동 자막 있음 → 그대로 사용 (품질 좋음)
                print("[TADAC] 수동 자막 사용")
                transcript, transcript_source = youtube_subtitle.get_transcript_from_youtube(
                    source, preferred_lang=language
                )

            else:
                # ⚠ 자동 자막만 or 자막 없음 → Whisper STT
                if sub_info["has_auto"]:
                    print("[TADAC] 자동 자막만 있음 → 품질 이슈로 Whisper STT 전환")
                else:
                    print("[TADAC] 자막 없음 → Whisper STT로 전환")


                tmp_dir = tempfile.mkdtemp(prefix="tadac_yt_audio_")
                tmp_dirs.append(tmp_dir)
                audio_path        = youtube_audio.extract_audio(source, tmp_dir)
                transcript        = stt_module.transcribe(audio_path, language=language, stt_prompt=stt_prompt, title=content_title)
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
            else:
                print(f"[TADAC] 입력: 로컬 오디오 파일 ({suffix})")

            transcript        = stt_module.transcribe(audio_path, language=language, stt_prompt=stt_prompt, title=content_title)
            transcript_source = "whisper"

        if not transcript.get("segments"):
            raise ValueError("세그먼트가 없어 게임 데이터를 만들 수 없음")

        # Whisper 원본 저장 (재처리 및 디버깅용)
        if transcript_source == "whisper":
            _save_raw_transcript(transcript)

        print(f"[TADAC] 트랜스크립트 준비 완료: 출처={transcript_source}, "
              f"세그먼트 {len(transcript['segments'])}개, 단어 {len(transcript.get('words', []))}개")

        # ── Step 1.5: 내용 분석 + 텍스트 교정 ─────────────────────────────────
        chapters = None
        summary  = None

        name_corrections = {}  # 고유명사 교정 사전
        global_keywords  = []  # 전체 강의 핵심 키워드 (빈칸 출제 풀)

        if transcript_source == "whisper" and refine:
            print("[TADAC] Whisper 원본 → GPT 내용 분석 (교정은 챕터별 통합처리)")
            summary, chapters, name_corrections, global_keywords = transcript_refiner._analyze_content(transcript.get("segments", []), title=content_title)

            # key_terms 기반 결정론적 1글자 오인식 탐지 (3글자 이상)
            fuzzy = _build_fuzzy_corrections(transcript.get("segments", []), global_keywords)
            fuzzy = {
                wrong: correct
                for wrong, correct in fuzzy.items()
                if transcript_refiner._is_safe_name_correction(wrong, correct)
            }
            if fuzzy:
                print(f"[TADAC] 유사 오인식 자동 탐지: {len(fuzzy)}개")
                for wrong, correct in fuzzy.items():
                    print(f"  {wrong} → {correct}")
                name_corrections.update(fuzzy)

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
            chapters = transcript_refiner.analyze_chapters_only(transcript)

        # ── Step 2, 3, 4: 통합 처리 (교정 + 키워드 + 퀴즈) ─────────────────────
        all_segments = transcript.get("segments", [])
        chapter_segments_map = quiz_generator._map_segments_to_chapters(all_segments, chapters)
        
        all_enriched_segments = []
        all_quizzes = []

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
                    q["ai_quiz_index"] = len(all_quizzes) + ch_quizzes.index(q)
                    q["chapter_index"] = ch_idx
                    q["chapter_title"] = chapter["title"]
                    q["trigger_time"] = round(trigger_time, 3)
                    q["segment_range"] = [seg_id_start, seg_id_end]

                all_quizzes.extend(ch_quizzes)
                all_enriched_segments.extend(ch_enriched)

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
                    q["ai_quiz_index"] = len(all_quizzes) + ch_quizzes.index(q)
                    q["chapter_index"] = ch_idx
                
                all_quizzes.extend(ch_quizzes)
                all_enriched_segments.extend(ch_enriched)

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

        # 파이프라인 메타데이터 추가
        game_data["stats"] = {
            "transcript_source": transcript_source,  # "whisper" / "youtube_manual"
            "content_title":     content_title,
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

        if return_branches:
            return {
                "corrected_subtitles": corrected_subtitle_data,
                "blank_game_data":     game_data,
            }

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

        # ── Phase A: 선행 작업 (전체 처리 필수) ───────────────────────────────
        # STT → 내용 분석 (챕터 경계 확정 필요)

        if _is_youtube_url(source):
            print(f"[TADAC] [스트리밍] 입력: YouTube URL")
            content_title, description = _get_youtube_metadata(source)

            sub_info = youtube_subtitle.check_subtitles(source)

            if sub_info["has_manual"]:
                print("[TADAC] 수동 자막 사용")
                transcript, transcript_source = youtube_subtitle.get_transcript_from_youtube(
                    source, preferred_lang=language
                )
            else:
                if sub_info["has_auto"]:
                    print("[TADAC] 자동 자막만 있음 → Whisper STT 전환")
                else:
                    print("[TADAC] 자막 없음 → Whisper STT 전환")

                tmp_dir = tempfile.mkdtemp(prefix="tadac_yt_audio_")
                tmp_dirs.append(tmp_dir)
                audio_path        = youtube_audio.extract_audio(source, tmp_dir)
                transcript        = stt_module.transcribe(audio_path, language=language, stt_prompt=stt_prompt, title=content_title)
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
            else:
                print(f"[TADAC] [스트리밍] 입력: 로컬 오디오 ({suffix})")

            transcript        = stt_module.transcribe(audio_path, language=language, stt_prompt=stt_prompt, title=content_title)
            transcript_source = "whisper"

        if not transcript.get("segments"):
            raise ValueError("세그먼트가 없어 게임 데이터를 만들 수 없음")

        # Whisper 원본 저장 (재처리 및 디버깅용)
        if transcript_source == "whisper":
            _save_raw_transcript(transcript)

        all_segments = transcript.get("segments", [])
        total_duration = max(seg.get("end", 0.0) for seg in all_segments)

        # 내용 분석 + 챕터 분할 (+ Whisper일 때 교정 맥락)
        chapters = None
        summary  = None

        name_corrections = {}  # 고유명사 교정 사전
        global_keywords  = []  # 전체 강의 핵심 키워드 (빈칸 출제 풀)

        if transcript_source == "whisper" and refine:
            print("[TADAC] [스트리밍] Whisper → 내용 분석 + 배치 교정")
            summary, chapters, name_corrections, global_keywords = transcript_refiner._analyze_content(all_segments, title=content_title)

            # key_terms 기반 결정론적 1글자 오인식 탐지 (3글자 이상)
            fuzzy = _build_fuzzy_corrections(all_segments, global_keywords)
            fuzzy = {
                wrong: correct
                for wrong, correct in fuzzy.items()
                if transcript_refiner._is_safe_name_correction(wrong, correct)
            }
            if fuzzy:
                print(f"[TADAC] 유사 오인식 자동 탐지: {len(fuzzy)}개")
                for wrong, correct in fuzzy.items():
                    print(f"  {wrong} → {correct}")
                name_corrections.update(fuzzy)

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
            print("[TADAC] [스트리밍] 교정 스킵 → 챕터 분할만")
            chapters = transcript_refiner.analyze_chapters_only(transcript)

        # ── init 이벤트: 챕터 목록 전달 ───────────────────────────────────────
        yield {
            "type":           "init",
            "content_title":  content_title,
            "total_duration": round(total_duration, 3),
            "chapters":       chapters,
        }

        # ── Phase B: 챕터별 스트리밍 처리 ─────────────────────────────────────
        chapter_segments_map = quiz_generator._map_segments_to_chapters(all_segments, chapters)

        all_quizzes = []
        total_enriched_segments = []

        for ch_idx, (chapter, ch_segs) in enumerate(zip(chapters, chapter_segments_map)):
            if not ch_segs:
                continue

            print(f"[TADAC] [스트리밍] 챕터 {ch_idx+1}/{len(chapters)}: '{chapter['title']}' 통합 처리 시작")

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
                    q["ai_quiz_index"] = len(all_quizzes) + ch_quizzes.index(q)
                    q["chapter_index"] = ch_idx
                    q["chapter_title"] = chapter["title"]
                    q["trigger_time"] = round(trigger_time, 3)
                    q["segment_range"] = [seg_id_start, seg_id_end]

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
                    q["ai_quiz_index"] = len(all_quizzes) + ch_quizzes.index(q)
                    q["chapter_index"] = ch_idx

            total_enriched_segments.extend(ch_enriched)
            all_quizzes.extend(ch_quizzes)

            # 게임 데이터 생성 (챕터 분)
            ch_game_data = blank_subtitle.build_game_data(
                ch_enriched,
                fall_speed=fall_speed,
                lead_time=lead_time,
            )
            ch_corrected_subtitle_data = _build_corrected_subtitle_data(ch_enriched)

            # ── chapter_ready 이벤트 ──────────────────────────────────────────
            yield {
                "type":                "chapter_ready",
                "chapter_index":       ch_idx,
                "chapter_title":       chapter["title"],
                "corrected_subtitles": ch_corrected_subtitle_data.get("subtitles", []),
                "subtitles":           ch_game_data.get("subtitles", []),
                "fall_events":         ch_game_data.get("fall_events", []),
                "quizzes":             ch_quizzes,
            }

        # ── complete 이벤트 ───────────────────────────────────────────────────
        yield {
            "type":  "complete",
            "stats": {
                "transcript_source": transcript_source,
                "content_title":     content_title,
                "total_words":       len(transcript.get("words", [])),
                "language":          language,
                "gpt_refined":       (transcript_source == "whisper" and refine),
                "total_quizzes":     len(all_quizzes),
                "total_segments":    len(total_enriched_segments),
            },
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
    parser.add_argument("--stream",        action="store_true",       help="스트리밍 모드 (챕터별 출력)")
    args = parser.parse_args()

    print(f"[TADAC] 파이프라인 시작")
    print(f"[TADAC] 소스: {args.source}")
    print(f"[TADAC] 설정: lang={args.lang}, blanks={MAX_BLANKS_PER_SENTENCE}(고정), refine={not args.no_refine}")

    if args.stream:
        # 스트리밍 모드: 챕터별로 출력
        print("[TADAC] 스트리밍 모드")
        
        aggregated_corrected_subtitle_data = {
            "subtitles": [],
            "config": {
                "total_segments": 0,
            },
        }
        aggregated_blank_game_data = {
            "subtitles": [],
            "fall_events": [],
            "quizzes": [],
            "config": {
                "fall_speed": BASE_FALL_SPEED,
                "lead_time": BASE_LEAD_TIME,
                "blanks_per_sentence": MAX_BLANKS_PER_SENTENCE
            }
        }
        
        for event in run_pipeline_streaming(
            source     = args.source,
            language   = args.lang,
            stt_prompt = args.prompt,
            refine     = not args.no_refine,
        ):
            print(f"\n[TADAC] === 이벤트: {event['type']} ===")
            print(json.dumps(event, ensure_ascii=False, indent=2)[:500])
            
            # 이벤트 타입에 따라 데이터 수합
            if event["type"] == "chapter_ready":
                aggregated_corrected_subtitle_data["subtitles"].extend(event.get("corrected_subtitles", []))
                aggregated_blank_game_data["subtitles"].extend(event.get("subtitles", []))
                aggregated_blank_game_data["fall_events"].extend(event.get("fall_events", []))
                aggregated_blank_game_data["quizzes"].extend(event.get("quizzes", []))
            elif event["type"] == "complete":
                stats = event.get("stats", {})
                aggregated_corrected_subtitle_data["stats"] = stats
                aggregated_blank_game_data["stats"] = stats
                
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
        
        # 전체 데이터 합쳐서 브랜치별 파일로 저장
        output_paths = _branch_output_paths(args.output)
        output_paths["corrected_subtitles"].write_text(
            json.dumps(aggregated_corrected_subtitle_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output_paths["blank_game_data"].write_text(
            json.dumps(aggregated_blank_game_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[TADAC] 교정 자막 저장 완료: {output_paths['corrected_subtitles'].resolve()}")
        print(f"[TADAC] 빈칸 게임 데이터 저장 완료: {output_paths['blank_game_data'].resolve()}")

    else:
        # 일괄 모드
        branch_data = run_pipeline(
            source     = args.source,
            language   = args.lang,
            stt_prompt = args.prompt,
            refine     = not args.no_refine,
            return_branches = True,
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
        print(f"[TADAC] 결과: 세그먼트 {game_data['config']['total_segments']}개, "
              f"빈칸 {game_data['config']['total_blanks']}개, "
              f"낙하 이벤트 {len(game_data['fall_events'])}개")


if __name__ == "__main__":
    main()
