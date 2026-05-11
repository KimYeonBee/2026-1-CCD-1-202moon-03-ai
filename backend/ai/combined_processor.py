# -*- coding: utf-8 -*-
"""
자막 교정 + 키워드 추출 + 퀴즈 생성을 한 번에 수행하는 통합 처리 모듈 (v2 경량화)

- 챕터 단위(수십 개의 세그먼트)로 GPT에 한 번의 API 호출을 수행
- GPT output을 최소화: corrections(diff만) + segment_keywords + quizzes
- 전체 텍스트를 다시 출력하지 않아 output 토큰 ~80% 절감
"""

import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

def _build_system_prompt(summary, global_keywords, questions_count):
    keywords_str = ", ".join(global_keywords) if global_keywords else "(없음)"
    return f"""너는 ADHD 학습자를 위한 교육용 게임 콘텐츠 생성 AI다.
입력으로 주어지는 강의 녹취록 챕터(세그먼트 배열 JSON)를 분석하고, 아래 네 가지 작업을 한 번에 수행하여 JSON으로 반환하라.

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

[작업 2] 빈칸 키워드 매핑 (전역 키워드 기반)
- "전체 강의 핵심 키워드 목록"에서 각 세그먼트 텍스트에 실제로 등장하는 키워드를 골라라.
- ★ 반드시 해당 세그먼트 텍스트 안에 정확히 존재하는 문자열만 선택하라.
- ★ 한 세그먼트당 최대 2개. 해당 구간에서 가장 중요하게 쓰이는 키워드를 우선 선택하라.
- 키워드 목록에 없는 단어는 절대 선택하지 마라.
- 키워드가 하나도 등장하지 않는 세그먼트는 segment_keywords에서 제외하라.
- 같은 키워드가 여러 세그먼트에 반복 등장하면 각 세그먼트마다 모두 선택하라 (반복 학습 효과).

[작업 3] 챕터 퀴즈 생성
- 챕터 전체 내용을 바탕으로 4지선다 퀴즈 {questions_count}개를 생성하라.
- 정답은 반드시 지문에 근거해야 하며, 오답은 그럴듯하게 구성하라.
- 정답 선택 시 피드백은 칭찬과 해설, 오답 선택 시 피드백은 격려와 해설을 포함하라.

[작업 4] 챕터 복습 요약 (chapter_summary)
- 학습자가 영상을 다시 보지 않고도 이 챕터를 복습할 수 있을 정도의 분량으로 요약하라.
- 분량: 3~6문장 또는 3~5개 불릿. 너무 짧으면 복습 가치가 없고, 너무 길면 학습 부담이 됨.
- 포함해야 할 것: 핵심 개념·정의, 등장한 인물·사건·고유명사, 결론·시사점.
- 발화 그대로 옮기지 말고 학습자가 이해하기 쉬운 정돈된 문장으로 재구성하라.
- 추측 금지: 챕터에 없는 내용을 보충하거나 일반 상식을 끼워 넣지 마라.
- 마크다운 사용 가능 (불릿 `-`, **굵게** 등). 헤더(`#`)는 쓰지 마라 — 챕터 제목은 외부에서 붙임.

★ 출력 규칙:
- corrections: 수정이 필요한 세그먼트만 포함
- segment_keywords: 키워드가 있는 세그먼트만 포함
- chapter_summary: 반드시 포함 (빈 챕터가 아니라면)

출력 형식 (반드시 JSON 형식을 지킬 것):
{{
  "corrections": [
    {{"id": 2, "text": "교정된 텍스트 (수정된 세그먼트만)"}}
  ],
  "segment_keywords": [
    {{"id": 2, "keywords": ["키워드1", "키워드2"]}},
    {{"id": 26, "keywords": ["키워드3"]}}
  ],
  "quizzes": [
    {{
      "question": "질문 내용",
      "options": ["선택지0", "선택지1", "선택지2", "선택지3"],
      "answer_index": 1,
      "correct_feedback": "정답 해설 및 칭찬",
      "incorrect_feedback": "오답 해설 및 격려"
    }}
  ],
  "chapter_summary": "이 챕터의 핵심 내용을 정리한 복습용 요약 (3~6문장 또는 불릿 3~5개)"
}}"""

def process_chapter_unified(chapter_segments, summary, chapter_title, blanks_per_sentence=2, global_keywords=None, questions_count=3):
    """
    하나의 챕터(세그먼트 배열)를 받아 corrections, segment_keywords, quizzes를 반환합니다.

    Args:
        global_keywords: 전체 강의에서 추출한 핵심 키워드 목록. 이 목록 안에서만 빈칸을 선택.

    Returns:
        dict with keys: corrections, segment_keywords, quizzes
    """
    if not chapter_segments:
        return {"corrections": [], "segment_keywords": [], "quizzes": [], "chapter_summary": ""}

    system_prompt = _build_system_prompt(summary, global_keywords or [], questions_count)
    
    # 입력으로 줄 세그먼트 구성 (GPT 토큰 절약을 위해 최소한의 정보만)
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
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ]
            )

            raw = response.choices[0].message.content
            data = json.loads(raw)
            
            # 사용량 로그
            usage = response.usage
            if usage:
                print(f"[TADAC] 토큰 사용량 — input: {usage.prompt_tokens}, output: {usage.completion_tokens}, total: {usage.total_tokens}")
            
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
