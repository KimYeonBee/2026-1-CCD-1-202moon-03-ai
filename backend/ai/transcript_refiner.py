# -*- coding: utf-8 -*-
"""
STT 텍스트 교정 + 챕터 분할 모듈

STT = "초안", GPT = "교정 편집자"

- 전체 스크립트를 1번 읽고 맥락 요약 + 챕터 분할을 동시에 수행 (API 1회)
- Whisper가 잘못 받아쓴 전문 용어, 띄어쓰기, 어색한 표현을 자연스럽게 수정
- 챕터 정보는 퀴즈 생성 모듈로 전달되어 단원별 퀴즈 생성에 사용
"""

import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# OpenAI 클라이언트 초기화 (공식 API 사용)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ── 빈칸 키워드 stopword ──────────────────────────────────────────────────────
# GPT(특히 gpt-4o-mini)가 프롬프트 지시를 무시하고 generic word를 골라내는 경우가
# 잦아서, 결정론적 후처리로 강제 제거. 학습자가 강의를 듣지 않고도 추측할 수
# 있는 단어들이라 빈칸으로 만들면 학습 효과가 없음.
#
# 단어 목록은 ai/keyword_stopwords.txt 에 외부화 — 도메인별 보강을 위해 비코더도
# 편집 가능. 새 도메인을 다룰 때 학습 효과가 떨어지는 단어를 발견하면 그 파일에
# 추가.

STOPWORDS_FILE = Path(__file__).parent / "keyword_stopwords.txt"


def _load_stopwords(path):
    if not path.exists():
        print(f"[TADAC] stopword 파일 없음: {path} (필터 비활성)")
        return frozenset()

    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        words.add(token)
    return frozenset(words)


KEYWORD_STOPWORDS = _load_stopwords(STOPWORDS_FILE)
print(f"[TADAC] stopword 로드: {len(KEYWORD_STOPWORDS)}개 ({STOPWORDS_FILE.name})")


def filter_filler_keywords(keywords):
    """KEYWORD_STOPWORDS에 해당하는 키워드 제거. dict/str 양쪽 지원."""
    out = []
    for kw in keywords:
        token = kw["keyword"] if isinstance(kw, dict) else kw
        if token and token not in KEYWORD_STOPWORDS:
            out.append(kw)
    return out


# ── 세그먼트 압축 (챕터 감지용) ───────────────────────────────────────────────
# 잘게 쪼개진 세그먼트를 1분 단위 블록으로 합쳐서 GPT 입력 크기 줄이기

def _condense_segments(segments, window_sec=60):
    """세그먼트를 window_sec 간격 블록으로 합쳐서 요약용 텍스트 생성"""
    if not segments:
        return []

    blocks = []
    current_texts = []
    block_start = segments[0].get("start", 0.0)

    for seg in segments:
        seg_start = seg.get("start", 0.0)
        seg_text  = seg.get("text", "").strip()

        if seg_start - block_start >= window_sec and current_texts:
            blocks.append({
                "start": block_start,
                "end":   seg_start,
                "text":  " ".join(current_texts),
            })
            current_texts = []
            block_start   = seg_start

        if seg_text:
            current_texts.append(seg_text)

    if current_texts:
        last_end = segments[-1].get("end", block_start)
        blocks.append({
            "start": block_start,
            "end":   last_end,
            "text":  " ".join(current_texts),
        })

    return blocks


# ── Step 1: 전체 내용 분석 (맥락 요약 + 챕터 분할) ────────────────────────────
# 1번의 GPT 호출로 교정용 맥락과 챕터 경계를 동시에 파악

ANALYSIS_PROMPT = """너는 강의 녹취록을 분석하는 전문가다.
아래 강의 녹취록을 읽고 세 가지 작업을 수행하라.

[작업 1] 교정 맥락 파악
- 강의 주제를 한 문장으로 요약
- 학습자가 반드시 기억해야 할 핵심 키워드를 최대 100개 추출하라.
  우선순위: 인물명 > 사건명 > 제도/정책명 > 지명/왕조 > 전문 용어/원리/공식 > 핵심 개념
  → 이 키워드들은 빈칸 문제로 출제되므로, **교육적으로 의미 있는 것만** 포함하라.
  → 학습자가 강의를 듣지 않으면 떠올릴 수 없는 단어를 우선하라.
  → 품질 > 수량. 50개를 채우려고 약한 단어를 넣지 마라. 부족하면 부족한 대로 둬라.

  ★ 절대 포함 금지 (filler/generic 단어):
  - 조사, 접속사, 부사, 감탄사, 일상 구어체 표현
  - 일반 명사: "양", "것", "정도", "부분", "경우", "방식", "상태", "내용", "물질", "단계"
  - 발화 동사/표현: "이야기", "해보도록", "알아보도록", "살펴보도록", "설명"
  - 의문/지시어: "어떤", "이런", "저런", "그런", "무엇", "어느"
  - 시간/순서 부사: "처음", "시작", "마지막", "다음", "이제", "결국", "먼저"
  - 평가형 형용사: "중요", "필요", "가능", "다양"
  - 특정 강의 맥락 없이 의미가 통하는 일반어는 모두 제외

  → 위 단어들은 강의 주제와 무관하게 어디서나 등장하므로 빈칸으로 만들면
     학습자가 추측만으로 맞출 수 있어 학습 효과가 없다.

[작업 2] 고유명사 추론
- 녹취록은 음성 인식(STT)으로 생성되어 인물명, 지명, 역사 용어 등이 잘못 적혀 있을 수 있다
- 영상 제목과 강의 맥락은 참고만 하되, 텍스트에 없는 정보를 추론해서 바꾸지 마라
- 발음이 거의 같은 STT 오인식만 "잘못 적힌 형태 → 올바른 형태"로 나열하라
- 지칭 대상 추론/의역/정규화/축약/확장/직함 제거는 절대 하지 마라
  예: "삼성 회장님 → 이재용", "부산시 → 부산", "울프 → 울프 아저씨", "LCT → 엘시티" 금지
- 확실하지 않은 것은 포함하지 마라

[작업 3] 챕터(단원) 분할
- 주제가 바뀌는 지점을 기준으로 챕터를 나눠라
- 각 챕터는 약 10~15분 분량 (최소 5분, 최대 20분)
- 챕터 제목은 해당 구간의 핵심 주제를 간결하게 요약
- 챕터 사이에 빈 구간이 없어야 함
- 전체 강의를 빠짐없이 커버해야 함

출력 형식 (JSON):
{
  "topic_summary": "강의 주제 한 문장 요약",
  "key_terms": ["용어1", "용어2", ...],
  "proper_noun_corrections": [
    {"wrong": "STT가 잘못 적은 형태", "correct": "올바른 표기"},
    ...
  ],
  "chapters": [
    {"title": "챕터 제목", "start_min": 0, "end_min": 12.5},
    {"title": "챕터 제목", "start_min": 12.5, "end_min": 25},
    ...
  ]
}"""


def _is_safe_name_correction(wrong, correct):
    """Return True only for conservative STT typo fixes, not semantic rewrites."""
    wrong_norm = "".join(str(wrong).split()).lower()
    correct_norm = "".join(str(correct).split()).lower()

    if not wrong_norm or not correct_norm or wrong_norm == correct_norm:
        return False

    # Do not accept expansions/contractions or suffix trimming.
    if wrong_norm in correct_norm or correct_norm in wrong_norm:
        return False

    # Do not convert acronyms/Latin text into Korean or vice versa automatically.
    wrong_has_ascii = any("a" <= ch <= "z" or "A" <= ch <= "Z" for ch in str(wrong))
    correct_has_ascii = any("a" <= ch <= "z" or "A" <= ch <= "Z" for ch in str(correct))
    if wrong_has_ascii != correct_has_ascii:
        return False

    # Big length changes are usually entity resolution, not ASR correction.
    if abs(len(wrong_norm) - len(correct_norm)) > 1:
        return False

    # 정규화된(공백 제거) 형태가 단일 토큰이어야 함.
    # 원본 토큰 수가 아닌 정규화 후로 검사 — "쿠션왕조"(wrong) → "쿠샨 왕조"(correct) 같이
    # 공백 유무만 다른 정상 매핑을 막는 false negative 방지.
    if len(wrong_norm.split()) != 1 or len(correct_norm.split()) != 1:
        return False

    return True


def _analyze_content(segments, title=None):
    """
    전체 강의를 1번 읽고 맥락 요약 + 챕터 분할을 동시에 수행.

    Args:
        segments: 세그먼트 목록
        title: 영상/파일 제목 (있으면 분석 맥락에 활용)

    Returns:
        (summary_text, chapters, name_corrections, key_terms)
        - summary_text: 교정용 맥락 텍스트
        - chapters: [{"title": str, "start_sec": float, "end_sec": float}, ...]
        - name_corrections: {"잘못된 표기": "올바른 표기", ...}
        - key_terms: 전체 강의 핵심 키워드 목록 (빈칸 출제 풀로 사용)
    """
    total_duration = max(seg.get("end", 0.0) for seg in segments)
    total_min = total_duration / 60

    # 세그먼트를 1분 블록으로 압축
    blocks = _condense_segments(segments)

    print(f"[TADAC] 내용 분석: {len(segments)}개 세그먼트 → {len(blocks)}개 블록 ({total_min:.0f}분)")

    # 블록을 [시간] 텍스트 형식으로 구성 (전체 텍스트 보존 — STT 오인식 탐지 정확도 향상)
    lines = []
    if title:
        lines.append(f"영상 제목: {title}")
    lines.append(f"강의 총 길이: {total_min:.0f}분\n")
    for b in blocks:
        start_m = b["start"] / 60
        end_m   = b["end"]   / 60
        lines.append(f"[{start_m:.1f}분~{end_m:.1f}분] {b['text']}")

    user_prompt = "\n".join(lines)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": ANALYSIS_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )

    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[TADAC] 내용 분석 JSON 파싱 오류: {e}")
        summary_text = ""
        chapters = [{"title": "전체 내용", "start_sec": 0.0, "end_sec": total_duration}]
        return summary_text, chapters, {}

    # 교정용 맥락 텍스트 구성
    topic = data.get("topic_summary", "")
    raw_key_terms = data.get("key_terms", [])

    # GPT 무관 결정론적 stopword 필터 — generic word 강제 제거
    key_terms = filter_filler_keywords(raw_key_terms)
    dropped = [kw for kw in raw_key_terms if kw not in key_terms]
    if dropped:
        print(f"[TADAC] key_terms stopword 제거 {len(dropped)}개: {', '.join(dropped)}")

    terms = key_terms  # 하위 호환용 alias
    noun_corrections_list = data.get("proper_noun_corrections", [])

    # 고유명사 교정 사전 구성 (wrong → correct 매핑)
    name_corrections = {}
    for c in noun_corrections_list:
        if isinstance(c, dict) and c.get("wrong") and c.get("correct"):
            wrong = c["wrong"].strip()
            correct = c["correct"].strip()
            if _is_safe_name_correction(wrong, correct):
                name_corrections[wrong] = correct

    # 고유명사 교정 목록을 맥락 텍스트에 포함
    summary_parts = [
        f"## 강의 주제\n{topic}",
        f"\n## 주요 전문 용어\n{', '.join(terms)}",
    ]
    if name_corrections:
        corrections_str = "\n".join(
            f"  {wrong} → {correct}" for wrong, correct in name_corrections.items()
        )
        summary_parts.append(f"\n## 고유명사 교정 목록 (STT 오인식 → 올바른 표기)\n{corrections_str}")
    summary_text = "\n".join(summary_parts)

    print(f"[TADAC] 주제: {topic}")
    print(f"[TADAC] 핵심 키워드 {len(key_terms)}개: {', '.join(key_terms[:15])}")
    if name_corrections:
        for wrong, correct in name_corrections.items():
            print(f"[TADAC] 고유명사 교정: {wrong} → {correct}")

    # 챕터 분 → 초 변환
    chapters = []
    for ch in data.get("chapters", []):
        if not isinstance(ch, dict):
            continue
        chapters.append({
            "title":     ch.get("title", ""),
            "start_sec": ch.get("start_min", 0) * 60,
            "end_sec":   ch.get("end_min", 0)   * 60,
        })

    # 챕터 감지 실패 시 폴백
    if not chapters:
        chapters = [{"title": "전체 내용", "start_sec": 0.0, "end_sec": total_duration}]

    print(f"[TADAC] 챕터 {len(chapters)}개 감지:")
    for i, ch in enumerate(chapters):
        print(f"  [{i+1}] {ch['start_sec']/60:.0f}분~{ch['end_sec']/60:.0f}분: {ch['title']}")

    return summary_text, chapters, name_corrections, key_terms


# ── 검수 패스: key_terms 기반 STT 오인식 탐지 ────────────────────────────────
# _analyze_content가 추출한 key_terms를 ground truth로 보고, 전체 텍스트 안에
# 그 용어와 비슷하지만 정확히 일치하지 않는 단어를 모두 찾아 교정 페어로 반환.
# 도메인 무관: GPT는 어떤 과목이든 "이 용어와 STT가 잘못 받아쓴 형태"를 식별 가능.

VERIFY_PROMPT = """너는 STT(음성 인식) 결과를 검수하는 전문가다.
아래 "정답 용어 목록"은 강의에서 반드시 정확하게 표기되어야 할 핵심 용어들이다.
"강의 텍스트"를 처음부터 끝까지 읽고, 정답 용어 목록 중 어떤 용어가 텍스트 안에서 잘못 표기되어 있는지 모두 찾아라.

판단 기준:
- 정답 용어와 발음이 비슷하지만 1~2글자가 다르게 적혀 있다면 STT 오인식 가능성이 높음
  예시: "문벌귀족"(정답) vs 텍스트의 "문벌기족" → 오인식
  예시: "최충헌"(정답) vs 텍스트의 "최충원" → 오인식
- 단, 다른 정답 용어와 정확히 일치하는 단어는 절대 교정 대상이 아니다
  예시: "선종"과 "교종"이 둘 다 정답 목록에 있으면, "선종"이 적혀있다고 해서 "교종"으로 바꾸지 마라
- 확실하지 않으면 포함하지 마라
- 같은 오인식이 여러 형태로 등장하면 모두 나열하라 (예: "이의민" 정답 vs "이유민", "이여민")

출력 형식 (JSON):
{
  "corrections": [
    {"wrong": "텍스트에 적힌 잘못된 형태", "correct": "정답 용어 목록의 올바른 형태"}
  ]
}
교정할 항목이 없으면 {"corrections": []} 를 반환하라."""


def verify_with_key_terms(segments, key_terms):
    """key_terms를 정답 기준으로 삼아, 텍스트 안의 STT 오인식 단어들을 GPT로 식별.

    Returns:
        dict: {"잘못된 형태": "올바른 형태", ...}
    """
    if not key_terms or not segments:
        return {}

    full_text = " ".join(seg.get("text", "").strip() for seg in segments)

    user_prompt = f"""정답 용어 목록 ({len(key_terms)}개):
{', '.join(key_terms)}

강의 텍스트:
{full_text}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": VERIFY_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        print(f"[TADAC] 검수 패스 오류: {e}")
        return {}

    key_terms_set = set(key_terms)
    corrections = {}
    for c in data.get("corrections", []):
        if not isinstance(c, dict):
            continue
        wrong   = (c.get("wrong")   or "").strip()
        correct = (c.get("correct") or "").strip()
        if not wrong or not correct or wrong == correct:
            continue
        # 정답 형태가 key_terms에 있어야 함 (GPT 환각 방지)
        if correct not in key_terms_set:
            continue
        # 잘못된 형태가 다른 key_term과 정확히 겹치면 스킵 (false positive 방지)
        if wrong in key_terms_set:
            continue
        corrections[wrong] = correct

    return corrections


# ── Step 2: 세그먼트 배치 교정 ────────────────────────────────────────────────
# 5~10개 세그먼트를 한 번의 GPT 호출로 교정 (API 호출 횟수 대폭 감소)

BATCH_SIZE = 10  # 한 번에 교정할 세그먼트 수

def _correct_segments_batch(batch_segments, summary, full_text):
    """여러 세그먼트를 한 번의 GPT 호출로 배치 교정"""
    import time

    # 배치 입력 구성: [{"id": 0, "text": "원본 문장"}, ...]
    batch_input = []
    for seg in batch_segments:
        text = seg.get("text", "").strip()
        if text:
            batch_input.append({
                "id":   seg.get("id", seg.get("segment_id", 0)),
                "text": text,
            })

    if not batch_input:
        return {s.get("id", s.get("segment_id", 0)): s.get("text", "") for s in batch_segments}

    correction_system_prompt = f"""
너는 STT(음성 인식) 오류만 고치는 교정자다.
아래 JSON 배열의 각 문장에서 음성 인식 오류로 잘못 적힌 단어만 수정하라.

절대 지켜야 할 규칙:
- 원본의 말투, 어조, 뉘앙스를 그대로 유지하라 (반말→존댓말 변환 금지)
- 문장을 재작성하거나 의역하지 마라
- 단어 순서를 바꾸지 마라
- 없는 내용을 추가하지 마라
- 확실하지 않으면 원본을 그대로 출력하라
- 의성어, 감탄사, 구어체 표현은 원본 그대로 유지하라
- 인명, 지명 등 고유명사가 잘못 인식된 경우, 아래 맥락의 "고유명사 교정 목록"을 참고하여 수정하라

강의 맥락:
{summary}

참고 스크립트:
{full_text[:2000]}

출력 형식 (JSON):
{{"results": [{{"id": 0, "text": "교정된 문장"}}, ...]}}
"""

    import json

    user_prompt = json.dumps(batch_input, ensure_ascii=False)

    max_retries = 5
    base_delay  = 2

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": correction_system_prompt},
                    {"role": "user",   "content": user_prompt},
                ]
            )

            raw = response.choices[0].message.content
            data = json.loads(raw)

            # 결과 파싱 — id → 교정 텍스트 매핑
            corrections = {}
            results = data.get("results", [])
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict) and "id" in item and "text" in item:
                        corrections[item["id"]] = item["text"].strip()

            # 파싱 실패한 세그먼트는 원본 유지
            for seg in batch_segments:
                seg_id = seg.get("id", seg.get("segment_id", 0))
                if seg_id not in corrections:
                    corrections[seg_id] = seg.get("text", "").strip()

            return corrections

        except json.JSONDecodeError as e:
            print(f"[TADAC] 배치 교정 JSON 파싱 오류: {e} → 원본 유지")
            return {s.get("id", s.get("segment_id", 0)): s.get("text", "") for s in batch_segments}

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                delay = base_delay * (2 ** attempt)
                print(f"[TADAC] Rate limit → {delay}초 대기 후 재시도 ({attempt+1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"[TADAC] 배치 교정 오류: {e} → 원본 유지")
                return {s.get("id", s.get("segment_id", 0)): s.get("text", "") for s in batch_segments}

    # 모든 재시도 실패 → 원본 유지
    print("[TADAC] 재시도 초과 → 원본 유지")
    return {s.get("id", s.get("segment_id", 0)): s.get("text", "") for s in batch_segments}


# ── 메인 함수 ─────────────────────────────────────────────────────────────────

def refine(transcript: dict, title: str = None):
    """
    Whisper STT 결과를 GPT로 교정하고, 동시에 챕터 분할도 수행.

    1번의 GPT 호출로 맥락 요약 + 챕터 분할 → 이후 배치 교정

    Args:
        transcript: stt.transcribe()가 반환한 딕셔너리
        title: 영상/파일 제목 (내용 분석 맥락에 활용)

    Returns:
        (refined_transcript, chapters)
        - refined_transcript: 교정된 transcript (같은 구조, text 필드만 수정됨)
        - chapters: [{"title": str, "start_sec": float, "end_sec": float}, ...]
    """
    segments  = transcript.get("segments", [])
    full_text = transcript.get("text", "")

    if not segments:
        print("[TADAC] 교정할 세그먼트가 없습니다")
        return transcript, []

    total_batches = (len(segments) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"[TADAC] 텍스트 교정 시작: 총 {len(segments)}개 세그먼트 → {total_batches}개 배치 (배치당 {BATCH_SIZE}개)")

    # 1. 전체 내용 분석 — 맥락 요약 + 챕터 분할 (GPT 1회)
    print("[TADAC] 강의 내용 분석 중 (맥락 요약 + 챕터 분할)...")
    summary, chapters, _, _ = _analyze_content(segments, title=title)
    print(f"[TADAC] 내용 분석 완료")

    # 2. 배치 단위로 GPT 교정
    corrected_segments = []
    batch_num = 0

    for i in range(0, len(segments), BATCH_SIZE):
        batch = segments[i:i + BATCH_SIZE]
        batch_num += 1

        # 빈 텍스트만 있는 세그먼트는 스킵
        has_text = any(seg.get("text", "").strip() for seg in batch)
        if not has_text:
            corrected_segments.extend(batch)
            continue

        print(f"[TADAC] 배치 {batch_num}/{total_batches}: 세그먼트 {i}~{i+len(batch)-1}")

        # 배치 교정 실행
        corrections = _correct_segments_batch(batch, summary, full_text)

        # 교정 결과 적용
        for seg in batch:
            seg_id = seg.get("id", seg.get("segment_id", 0))
            original_text  = seg.get("text", "").strip()
            corrected_text = corrections.get(seg_id, original_text)

            if corrected_text != original_text:
                print(f"  [{seg_id}] {original_text[:40]}...")
                print(f"     → {corrected_text[:40]}...")

            corrected_seg = dict(seg)
            corrected_seg["text"] = corrected_text
            corrected_segments.append(corrected_seg)

    # 전체 텍스트도 교정된 버전으로 재조합
    corrected_full_text = " ".join(
        seg.get("text", "") for seg in corrected_segments
    )

    # words 타임스탬프는 그대로 유지 (타임스탬프 기반 낙하 이벤트에 영향 없음)
    refined_transcript = dict(transcript)
    refined_transcript["text"] = corrected_full_text
    refined_transcript["segments"] = corrected_segments

    print(f"[TADAC] 텍스트 교정 완료: {len(corrected_segments)}개 세그먼트, {batch_num}개 배치 처리")
    return refined_transcript, chapters


# ── 챕터만 분석 (교정 없이) ───────────────────────────────────────────────────
# YouTube 자막 등 교정이 불필요할 때 챕터 분할만 수행

def analyze_chapters_only(transcript: dict):
    """
    교정 없이 챕터 분할만 수행.
    YouTube 자막처럼 이미 완성된 텍스트에 사용.

    Returns:
        chapters: [{"title": str, "start_sec": float, "end_sec": float}, ...]
    """
    segments = transcript.get("segments", [])

    if not segments:
        print("[TADAC] 세그먼트가 없어 챕터 분석 불가")
        return []

    total_duration = max(seg.get("end", 0.0) for seg in segments)

    # 5분 미만이면 단일 챕터
    if total_duration < 300:
        print("[TADAC] 5분 미만 → 단일 챕터")
        return [{"title": "전체 내용", "start_sec": 0.0, "end_sec": total_duration}]

    print("[TADAC] 챕터 분석 시작 (교정 없이)")
    _, chapters, _, _ = _analyze_content(segments)
    return chapters
