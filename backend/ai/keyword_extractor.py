# -*- coding: utf-8 -*-
"""
키워드 추출 모듈 — GPT로 세그먼트별 핵심 단어 추출 + 세그먼트 내 스케줄 배치

- GPT-5.4-nano 문장당 N개 키워드 추출
- Whisper 단어 타임스탬프에 의존하지 않고 세그먼트 내 순서대로 균등 배치
- 20개 세그먼트씩 묶어서 배치 처리 (API 호출 최소화)
"""

import json
import os

from openai import OpenAI
from dotenv import load_dotenv

from transcript_refiner import filter_filler_keywords

load_dotenv()

# OpenAI 클라이언트 초기화 (공식 API 사용)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# 한 번의 GPT 호출에 처리할 세그먼트 수
BATCH_SIZE = 20


def _keyword_text(keyword_info):
    token = keyword_info.get("keyword", "") if isinstance(keyword_info, dict) else keyword_info
    return str(token or "")


def _scheduled_target_time(segment, blank_index, blank_count):
    start = float(segment.get("start", 0.0) or 0.0)
    end = float(segment.get("end", start) or start)
    duration = max(end - start, 0.0)
    if blank_count <= 0 or duration == 0:
        return round(start, 3)
    return round(start + duration * ((blank_index + 1) / (blank_count + 1)), 3)


def schedule_segment_keywords(segment):
    """키워드 메타데이터의 start/end를 세그먼트 내 균등 스케줄 시점으로 채운다."""
    keywords = segment.get("keywords", [])
    blank_count = len(keywords)
    seg_end = float(segment.get("end", segment.get("start", 0.0)) or 0.0)

    for idx, kw_info in enumerate(keywords):
        if not isinstance(kw_info, dict):
            continue
        target = _scheduled_target_time(segment, idx, blank_count)
        kw_info["start"] = target
        kw_info["end"] = round(min(target + 0.5, seg_end), 3)
        kw_info["found"] = False

    return segment


# ── GPT 프롬프트 ──────────────────────────────────────────────────────────────
# 고유명사, 전문 용어 우선 / 조사·접속사·일반 부사 제외

SYSTEM_PROMPT = """You are a Korean language keyword extractor for an educational ADHD learning game.
Extract key vocabulary from each sentence that learners should ACTIVELY RECALL — words whose
meaning carries the lecture's content. The chosen words become fill-in-the-blank questions.

# Selection Priority (by educational domain)
- History (역사): Proper nouns (인명/지명/사건명), Years/Dates (연도/시기)
- Social/Economics (사회/경제): Concepts (개념), Policies (정책), Key figures (핵심 인물)
- Science (화학/물리/생물 등): Elements (원소), Chemical formulas (화학식), Principles (원리/법칙)

# Quality Over Quantity — THIS IS THE MOST IMPORTANT RULE
- Return UP TO N keywords per sentence. Fewer is ALWAYS better than padding with weak words.
- If a sentence has only 1 meaningful keyword → return 1. DO NOT pad.
- If a sentence has 0 educationally meaningful keywords → return an empty array.
- A sentence that asks the learner to type a generic word like "양" is a FAILED question
  because they can guess from context without learning anything.

# Hard Exclusions — NEVER select these even if they fit the requested count
- Particles (조사), conjunctions (접속사), common adverbs, filler words
- Greetings / conversational phrases ("안녕하세요", "반갑습니다", "자 그럼", "네 맞습니다", "감사합니다")
- Generic placeholder nouns: "양", "것", "정도", "부분", "경우", "방식", "상태", "내용", "물질", "단계"
- Speech-act verbs/nouns: "이야기", "해보도록", "알아보도록", "살펴보도록", "설명"
- Question / demonstrative words: "어떤", "이런", "저런", "그런", "무엇", "어느"
- Time / sequence adverbs: "처음", "시작", "마지막", "다음", "이제", "결국", "먼저"
- Evaluation adjectives (with no domain content): "중요", "필요", "가능", "다양"
- Slang / idioms with no educational value ("기고만장", "아무튼", "결국")
- Filler-only segments ("짠!", "우하하", "아유", "오", "우와", "어...") → return empty array

# Format
- Return ONLY the keywords as they appear in the original text (exact match preferred)
- Response must be valid JSON

Output format:
{
  "results": [
    {"segment_id": 0, "keywords": ["키워드1", "키워드2"]},
    {"segment_id": 1, "keywords": []},
    ...
  ]
}"""


# ── 배치 프롬프트 만들기 ──────────────────────────────────────────────────────
# 세그먼트 목록을 [번호] 텍스트 형태로 나열

def _build_user_prompt(segments, blanks_per_sentence):
    lines = [
        f"Extract UP TO {blanks_per_sentence} keywords per sentence.",
        "Quality over quantity — return fewer (or empty) rather than padding with filler.\n",
    ]
    for seg in segments:
        lines.append(f"[{seg['id']}] {seg['text']}")
    return "\n".join(lines)


# ── GPT 배치 호출 ─────────────────────────────────────────────────────────────
# 세그먼트 20개씩 GPT에 보내고 키워드 목록 받기

def _call_gpt_batch(segments, blanks_per_sentence):
    user_prompt = _build_user_prompt(segments, blanks_per_sentence)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,                             # 일관된 추출을 위해 낮게 설정
        response_format={"type": "json_object"},     # JSON 형식 강제
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )

    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
        results = data.get("results", [])
        # GPT 무관 결정론적 stopword 필터 — generic word 강제 제거
        for item in results:
            if isinstance(item, dict) and "keywords" in item:
                item["keywords"] = filter_filler_keywords(item["keywords"])
        return results
    except json.JSONDecodeError as e:
        print(f"[TADAC] GPT JSON 파싱 오류: {e}")
        return []


# ── Legacy helpers ───────────────────────────────────────────────────────────
# 예전 word timestamp 매핑 API를 참조하는 외부 코드가 있을 수 있어 남겨둔다.
# 현재 파이프라인은 아래 함수를 호출하지 않고, 세그먼트 기반 스케줄을 사용한다.

def _find_word_timestamp(keyword, words):
    # 1차: 정확히 일치하는 단어 탐색
    for w in words:
        if w["word"].strip() == keyword:
            return w

    # 2차: 포함 관계로 탐색 (조사 붙은 형태 처리)
    for w in words:
        if keyword in w["word"] or w["word"] in keyword:
            return w

    return None  # 매핑 실패


def _find_word_in_segment(keyword, segment, all_words):
    # 해당 세그먼트 시간 범위 안의 단어만 탐색
    # ⚠️ 전체 all_words 폴백 제거 — 키워드가 여러 번 등장할 때
    #    다른 세그먼트의 시간이 매핑되는 버그 방지
    #    (못 찾으면 None → 호출 측에서 세그먼트 중간 시점을 폴백으로 사용)
    seg_words = [
        w for w in all_words
        if w["start"] >= segment["start"] - 0.1
        and w["end"]   <= segment["end"]   + 0.1
    ]
    return _find_word_timestamp(keyword, seg_words)


# ── 메인 함수 ─────────────────────────────────────────────────────────────────
# Whisper transcript → GPT 키워드 추출 → 세그먼트 내 스케줄 배치 → 결과 반환

def extract_keywords(transcript, blanks_per_sentence=2):
    segments  = transcript.get("segments", [])

    if not segments:
        print("[TADAC] 세그먼트가 없어 키워드 추출 불가")
        return []

    print(f"[TADAC] 키워드 추출 시작: {len(segments)}개 세그먼트, "
          f"문장당 {blanks_per_sentence}개, 배치 크기 {BATCH_SIZE}")

    # 세그먼트 정규화 — id / start / end / text 필드 통일
    normalised = []
    for i, seg in enumerate(segments):
        normalised.append({
            "id":    seg.get("id", i),
            "start": seg.get("start", 0.0),
            "end":   seg.get("end",   0.0),
            "text":  seg.get("text",  "").strip(),
        })

    # 20개씩 배치로 GPT 호출
    keyword_map = {}  # {segment_id: [키워드 목록]}
    for batch_start in range(0, len(normalised), BATCH_SIZE):
        batch = normalised[batch_start: batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        print(f"[TADAC] GPT 배치 {batch_num}: 세그먼트 {batch_start} ~ {batch_start + len(batch) - 1}")

        batch_results = _call_gpt_batch(batch, blanks_per_sentence)
        for item in batch_results:
            if not isinstance(item, dict):
                print(f"[TADAC] GPT 응답 형식 오류 (dict 아님, 건너뜀): {type(item).__name__} → {str(item)[:120]}")
                continue
            seg_id = item.get("segment_id")
            if seg_id is None:
                print(f"[TADAC] segment_id 누락, 건너뜀: {str(item)[:120]}")
                continue
            keyword_map[seg_id] = item.get("keywords", [])

    # 키워드 스케줄 배치
    enriched       = []
    total_keywords = 0

    for seg in normalised:
        raw_keywords    = keyword_map.get(seg["id"], [])
        mapped_keywords = []

        for kw in raw_keywords:
            keyword = _keyword_text(kw)
            if not keyword:
                continue
            total_keywords += 1
            mapped_keywords.append({
                "keyword": keyword,
                "found":   False,
            })

        enriched_seg = {
            "segment_id": seg["id"],
            "start":      seg["start"],
            "end":        seg["end"],
            "text":       seg["text"],
            "keywords":   mapped_keywords,
        }
        enriched.append(schedule_segment_keywords(enriched_seg))

    print(f"[TADAC] 키워드 추출 완료: {total_keywords}개 추출, 세그먼트 내 균등 스케줄 배치")

    return enriched

def enrich_segments_with_keywords(segments, keyword_map, all_words):
    """
    GPT 추출을 제외하고, 주어진 키워드 맵을 바탕으로 세그먼트 내 균등 스케줄 배치만 수행.
    keyword_map: {segment_id: [keyword1, keyword2, ...]}
    all_words: 하위 호환용 파라미터. 현재 로직에서는 사용하지 않음.
    """
    enriched       = []
    total_keywords = 0

    for seg in segments:
        seg_id = seg.get("id", seg.get("segment_id", 0))
        raw_keywords    = keyword_map.get(seg_id, [])
        mapped_keywords = []

        for kw in raw_keywords:
            keyword = _keyword_text(kw)
            if not keyword:
                continue
            total_keywords += 1
            mapped_keywords.append({
                "keyword": keyword,
                "found":   False,
            })

        enriched_seg = {
            "segment_id": seg_id,
            "start":      seg.get("start", 0.0),
            "end":        seg.get("end", 0.0),
            "text":       seg.get("text", ""),
            "keywords":   mapped_keywords,
        }
        enriched.append(schedule_segment_keywords(enriched_seg))

    print(f"[TADAC] 키워드 스케줄 배치: {total_keywords}개")

    return enriched
