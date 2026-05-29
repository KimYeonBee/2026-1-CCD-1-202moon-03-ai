# -*- coding: utf-8 -*-
"""
자막 교정 + 키워드 추출 + 퀴즈 생성을 한 번에 수행하는 통합 처리 모듈 (v3 Claude Sonnet)

- 챕터 단위(수십 개의 세그먼트)로 Claude에 한 번의 API 호출을 수행
- Claude output을 최소화: corrections(diff만) + segment_keywords + quizzes
- 전체 텍스트를 다시 출력하지 않아 output 토큰 ~80% 절감
- 모델 환경변수: CHAPTER_MODEL (기본값: claude-sonnet-4-5)
"""

import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

# ── 모델 설정 ────────────────────────────────────────────────────────────────
# CHAPTER_MODEL 환경변수로 모델 교체 가능.
#   - "claude-*"  → Anthropic API
#   - 그 외       → OpenAI API (하위 호환)
CHAPTER_MODEL = os.getenv("CHAPTER_MODEL", "claude-sonnet-4-5")


def _is_anthropic_model(model_id: str) -> bool:
    return model_id.startswith("claude")


# ── Anthropic 클라이언트 (lazy init) ─────────────────────────────────────────
_anthropic_client = None

def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _anthropic_client


# ── OpenAI 클라이언트 (하위 호환) ────────────────────────────────────────────
_openai_client = None

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


print(f"[TADAC] 챕터 처리 모델: {CHAPTER_MODEL} (provider: {'anthropic' if _is_anthropic_model(CHAPTER_MODEL) else 'openai'})")


def _build_system_prompt(summary, global_keywords, questions_count):
    keywords_str = ", ".join(global_keywords) if global_keywords else "(없음)"

    # 키워드 작업 — global_keywords가 없으면 자체 판단
    if global_keywords:
        keyword_block = """[작업 2] 빈칸 키워드 매핑 (전역 키워드 기반)
- "전체 강의 핵심 키워드 목록"에서 각 세그먼트 텍스트에 실제로 등장하는 키워드를 골라라.
- ★ 반드시 해당 세그먼트 텍스트 안에 정확히 존재하는 문자열만 선택하라.
- ★ 한 세그먼트당 최대 2개. 해당 구간에서 가장 중요하게 쓰이는 키워드를 우선 선택하라.
- ★ 키워드는 반드시 5글자 이하만 선택하라. 6글자 이상인 키워드는 절대 선택하지 마라.
- 키워드 목록에 없는 단어는 절대 선택하지 마라.
- 키워드가 하나도 등장하지 않는 세그먼트는 segment_keywords에서 제외하라.
- 같은 키워드가 여러 세그먼트에 반복 등장하면 각 세그먼트마다 모두 선택하라 (반복 학습 효과)."""
    else:
        keyword_block = """[작업 2] 빈칸 키워드 추출
- 각 세그먼트에서 학습에 중요한 핵심 키워드(전문 용어, 고유명사, 핵심 개념)를 추출하라.
- ★ 반드시 해당 세그먼트 텍스트 안에 정확히 존재하는 문자열만 선택하라.
- ★ 한 세그먼트당 최대 2개. 해당 구간에서 가장 중요하게 쓰이는 키워드를 우선 선택하라.
- ★ 키워드는 반드시 5글자 이하만 선택하라. 6글자 이상인 키워드는 절대 선택하지 마라.
- "그래서", "여기서", "이것은" 같은 일반적인 접속사/지시사는 절대 키워드로 선택하지 마라.
- 키워드가 없는 세그먼트는 segment_keywords에서 제외하라."""

    # 퀴즈 작업 — questions_count가 0이면 생략
    if questions_count > 0:
        quiz_block = f"""[작업 3] 챕터 퀴즈 생성
- 챕터 전체 내용을 바탕으로 4지선다 퀴즈 {questions_count}개를 생성하라.
- 정답은 반드시 지문에 근거해야 하며, 오답은 그럴듯하게 구성하라.
- explanation: 정답이 왜 맞는지 간결하게 해설하라 (1~2문장)."""
        quiz_output_rule = '- quizzes: 퀴즈 배열 포함'
        quiz_output_format = """  "quizzes": [
    {"question": "질문 내용", "options": ["선택지0", "선택지1", "선택지2", "선택지3"], "answer_index": 1, "explanation": "정답 해설"}
  ],"""
    else:
        quiz_block = ""
        quiz_output_rule = '- quizzes: 빈 배열 []'
        quiz_output_format = '  "quizzes": [],'

    task_count = "네 가지" if questions_count > 0 else "세 가지"

    return f"""너는 ADHD 학습자를 위한 교육용 게임 콘텐츠 생성 AI다.
입력으로 주어지는 강의 녹취록 챕터(세그먼트 배열 JSON)를 분석하고, 아래 {task_count} 작업을 한 번에 수행하여 JSON으로 반환하라.

강의 맥락 및 고유명사 참고:
{summary}

전체 강의 핵심 키워드 목록 (빈칸 출제 풀):
{keywords_str}

[작업 1] 자막 교정 (diff만 출력)
- STT 음성 인식 오류로 잘못 적힌 단어만 교정하라.
- 원본의 말투, 어조, 뉘앙스는 그대로 유지하라 (반말을 존댓말로 바꾸는 등 의역 금지).
- 문장 구조나 단어 순서를 바꾸지 말고, 확실하지 않으면 원본을 그대로 유지하라.
- 위 "강의 맥락 및 고유명사 참고"의 교정 목록은 발음상 명백한 STT 오인식일 때만 반영하라.
- 지칭 대상 추론/의역/정규화/축약/확장/직함 제거는 절대 하지 마라.
  예: "삼성 회장님 → 이재용", "부산시 → 부산", "울프 → 울프 아저씨", "LCT → 엘시티" 금지.
- ★ 수정이 필요한 세그먼트만 corrections 배열에 포함하라.

{keyword_block}

{quiz_block}

[작업 {"4" if questions_count > 0 else "3"}] 챕터 복습 요약 (chapter_summary)
- 학습자가 영상을 다시 보지 않고도 이 챕터를 복습할 수 있을 정도의 분량으로 요약하라.
- 분량: 3~6문장 또는 3~5개 불릿. 너무 짧으면 복습 가치가 없고, 너무 길면 학습 부담이 됨.
- 포함해야 할 것: 핵심 개념·정의, 등장한 인물·사건·고유명사, 결론·시사점.
- 발화 그대로 옮기지 말고 학습자가 이해하기 쉬운 정돈된 문장으로 재구성하라.
- 추측 금지: 챕터에 없는 내용을 보충하거나 일반 상식을 끼워 넣지 마라.
- 마크다운 사용 가능 (불릿 `-`, **굵게** 등). 헤더(`#`)는 쓰지 마라 — 챕터 제목은 외부에서 붙임.

★ 출력 규칙:
- corrections: 수정이 필요한 세그먼트만 포함
- segment_keywords: 키워드가 있는 세그먼트만 포함
{quiz_output_rule}
- chapter_summary: 반드시 포함 (빈 챕터가 아니라면)

반드시 아래 JSON 형식만 출력하라. 마크다운 코드 펜스(\`\`\`) 금지. 설명 텍스트 금지.
{{
  "corrections": [
    {{"id": 2, "text": "교정된 텍스트 (수정된 세그먼트만)"}}
  ],
  "segment_keywords": [
    {{"id": 2, "keywords": ["키워드1", "키워드2"]}},
    {{"id": 26, "keywords": ["키워드3"]}}
  ],
{quiz_output_format}
  "chapter_summary": "이 챕터의 핵심 내용을 정리한 복습용 요약 (3~6문장 또는 불릿 3~5개)"
}}"""

def process_chapter_unified(chapter_segments, summary, chapter_title, blanks_per_sentence=2, global_keywords=None, questions_count=3):
    """
    하나의 챕터(세그먼트 배열)를 받아 corrections, segment_keywords, quizzes를 반환합니다.

    Args:
        global_keywords: 전체 강의에서 추출한 핵심 키워드 목록. 이 목록 안에서만 빈칸을 선택.

    Returns:
        dict with keys: corrections, segment_keywords, quizzes, chapter_summary
    """
    if not chapter_segments:
        return {"corrections": [], "segment_keywords": [], "quizzes": [], "chapter_summary": ""}

    system_prompt = _build_system_prompt(summary, global_keywords or [], questions_count)

    # 입력으로 줄 세그먼트 구성 (토큰 절약을 위해 최소한의 정보만)
    input_data = []
    for seg in chapter_segments:
        input_data.append({
            "id": seg.get("id", seg.get("segment_id")),
            "text": seg.get("text", "").strip()
        })

    user_prompt = f"챕터 제목: {chapter_title}\n세그먼트 목록:\n" + json.dumps(input_data, ensure_ascii=False)

    max_retries = 3
    base_delay = 2

    for attempt in range(max_retries):
        try:
            if _is_anthropic_model(CHAPTER_MODEL):
                data = _call_anthropic(system_prompt, user_prompt)
            else:
                data = _call_openai(system_prompt, user_prompt)

            # 사용량 로그는 각 provider 함수 내에서 출력
            return data

        except json.JSONDecodeError as e:
            print(f"[TADAC] 통합 처리 JSON 파싱 오류: {e} → 재시도 ({attempt+1}/{max_retries})")
            time.sleep(base_delay)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                delay = base_delay * (2 ** attempt)
                print(f"[TADAC] Rate limit → {delay}초 대기 후 재시도 ({attempt+1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"[TADAC] 통합 처리 오류: {e}")
                time.sleep(base_delay)

    print("[TADAC] 통합 처리 재시도 초과 → 빈 결과 반환")
    return {"corrections": [], "segment_keywords": [], "quizzes": [], "chapter_summary": ""}


def _call_anthropic(system_prompt, user_prompt):
    """Anthropic Claude API 호출 — prompt caching 활성화."""
    client = _get_anthropic_client()

    resp = client.messages.create(
        model      = CHAPTER_MODEL,
        max_tokens = 8192,
        system     = [{
            "type":          "text",
            "text":          system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        messages   = [{"role": "user", "content": user_prompt}],
    )

    # 사용량 로그
    usage = resp.usage
    input_tokens = getattr(usage, 'input_tokens', 0)
    output_tokens = getattr(usage, 'output_tokens', 0)
    cache_read = getattr(usage, 'cache_read_input_tokens', 0)
    cache_create = getattr(usage, 'cache_creation_input_tokens', 0)
    print(f"[TADAC] 토큰 사용량 — input: {input_tokens} (cache_read: {cache_read}, cache_create: {cache_create}), output: {output_tokens}")

    raw = resp.content[0].text.strip()

    # ```json ... ``` 펜스 제거
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

    data = json.loads(raw)

    # 필수 키 보장
    data.setdefault("corrections", [])
    data.setdefault("segment_keywords", [])
    data.setdefault("quizzes", [])
    data.setdefault("chapter_summary", "")

    return data


def _call_openai(system_prompt, user_prompt):
    """OpenAI API 호출 — Structured Outputs로 JSON 강제."""
    client = _get_openai_client()

    json_schema = {
        "name": "chapter_processing",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "corrections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id":   {"type": "integer"},
                            "text": {"type": "string"},
                        },
                        "required": ["id", "text"],
                        "additionalProperties": False,
                    },
                },
                "segment_keywords": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id":       {"type": "integer"},
                            "keywords": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["id", "keywords"],
                        "additionalProperties": False,
                    },
                },
                "quizzes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question":     {"type": "string"},
                            "options":      {"type": "array", "items": {"type": "string"}},
                            "answer_index": {"type": "integer"},
                            "explanation":  {"type": "string"},
                        },
                        "required": ["question", "options", "answer_index", "explanation"],
                        "additionalProperties": False,
                    },
                },
                "chapter_summary": {"type": "string"},
            },
            "required": ["corrections", "segment_keywords", "quizzes", "chapter_summary"],
            "additionalProperties": False,
        },
    }

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        response_format={"type": "json_schema", "json_schema": json_schema},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)

    usage = response.usage
    if usage:
        print(f"[TADAC] 토큰 사용량 — input: {usage.prompt_tokens}, output: {usage.completion_tokens}, total: {usage.total_tokens}")

    return data
