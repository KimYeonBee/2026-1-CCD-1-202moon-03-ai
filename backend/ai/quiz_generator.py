# -*- coding: utf-8 -*-
"""
퀴즈 생성 모듈 — 챕터별 4지선다 퀴즈 생성

- transcript_refiner에서 분석한 챕터(단원) 정보를 받아서 사용
- 각 챕터마다 4지선다 퀴즈 3문제씩 생성
- trigger_time: 해당 챕터 마지막 세그먼트 종료 시점
  → 프론트엔드가 영상 재생 중 trigger_time 이 되면 퀴즈 팝업
"""

import json
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# OpenAI 클라이언트 초기화 (공식 API 사용)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# 퀴즈 생성 설정
QUESTIONS_PER_CHAPTER = 3   # 챕터당 생성할 문제 수


# ── 퀴즈 프롬프트 & GPT 호출 ──────────────────────────────────────────────────

def _build_quiz_prompt(segments, chapter_title, questions_count):
    lines = [
        f"[챕터: {chapter_title}]",
        f"아래 강의 내용을 읽고 퀴즈 {questions_count}개를 만들어라.\n",
    ]
    for seg in segments:
        seg_id = seg.get("id", seg.get("segment_id", 0))
        text   = seg.get("text", "").strip()
        lines.append(f"[{seg_id}] {text}")
    return "\n".join(lines)


def _get_quiz_system_prompt(questions_count):
    return f"""너는 ADHD 학습자를 위한 퀴즈 출제 전문가다.
아래 강의 내용을 읽고 4지선다 퀴즈 {questions_count}개를 만들어라.

규칙:
- 정답은 반드시 지문에 근거해야 함
- 오답은 그럴듯하지만 명확히 틀린 내용으로 구성
- 단순 암기보다 이해를 묻는 문제 우선
- 각 문제의 options는 반드시 4개
- answer_index는 0~3 사이 정수 (정답 선택지 번호)
- 서로 다른 내용을 묻는 {questions_count}개 문제를 만들어라

출력 형식 (JSON):
{{
  "quizzes": [
    {{
      "question":           "질문 내용",
      "options":            ["선택지0", "선택지1", "선택지2", "선택지3"],
      "answer_index":       1,
      "correct_feedback":   "정답을 맞췄을 때 보여줄 해설 및 칭찬",
      "incorrect_feedback": "오답을 골랐을 때 보여줄 해설 및 격려"
    }}
  ]
}}"""


def _call_gpt_quiz(segments, chapter_title, questions_count):
    """챕터 세그먼트 → GPT → 퀴즈 리스트 반환"""
    user_prompt   = _build_quiz_prompt(segments, chapter_title, questions_count)
    system_prompt = _get_quiz_system_prompt(questions_count)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )

    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
        if "quizzes" in data:
            return data["quizzes"]
        if "question" in data:
            return [data]
        return []
    except json.JSONDecodeError as e:
        print(f"[TADAC] 퀴즈 JSON 파싱 오류: {e}")
        return []


# ── 세그먼트를 챕터에 매핑 ────────────────────────────────────────────────────

def _map_segments_to_chapters(segments, chapters):
    """각 챕터의 시간 범위에 해당하는 세그먼트들을 매핑"""
    chapter_segments = [[] for _ in chapters]

    for seg in segments:
        seg_mid = (seg.get("start", 0.0) + seg.get("end", 0.0)) / 2

        for i, ch in enumerate(chapters):
            if ch["start_sec"] <= seg_mid < ch["end_sec"]:
                chapter_segments[i].append(seg)
                break
        else:
            if chapter_segments:
                chapter_segments[-1].append(seg)

    return chapter_segments


# ── 메인 함수 ─────────────────────────────────────────────────────────────────
# 파이프라인에서 전달받은 챕터 기반으로 퀴즈 생성
# 챕터 감지는 transcript_refiner가 이미 수행 → 여기서는 퀴즈만 생성

def generate_quizzes(segments, chapters=None, questions_per_chapter=QUESTIONS_PER_CHAPTER):
    if not segments:
        print("[TADAC] 세그먼트가 없어 퀴즈 생성 불가")
        return []

    if not chapters:
        print("[TADAC] 챕터 정보 없음 — 전체를 단일 챕터로 처리")
        total_duration = max(seg.get("end", 0.0) for seg in segments)
        chapters = [{"title": "전체 내용", "start_sec": 0.0, "end_sec": total_duration}]

    # 세그먼트를 챕터에 매핑
    chapter_segments = _map_segments_to_chapters(segments, chapters)

    print(f"[TADAC] ── 퀴즈 생성 시작: {len(chapters)}개 챕터 × {questions_per_chapter}문제 ──")

    quizzes = []
    quiz_id_counter = 0

    for ch_idx, (chapter, ch_segs) in enumerate(zip(chapters, chapter_segments)):
        if not ch_segs:
            print(f"[TADAC] 챕터 {ch_idx+1} '{chapter['title']}': 세그먼트 없음 — 건너뜀")
            continue

        first_seg    = ch_segs[0]
        last_seg     = ch_segs[-1]
        seg_id_start = first_seg.get("id", first_seg.get("segment_id", 0))
        seg_id_end   = last_seg.get("id",  last_seg.get("segment_id", 0))
        trigger_time = last_seg.get("end", 0.0)

        print(f"[TADAC] 챕터 {ch_idx+1}/{len(chapters)}: '{chapter['title']}' "
              f"({chapter['start_sec']/60:.0f}분~{chapter['end_sec']/60:.0f}분, "
              f"세그먼트 {len(ch_segs)}개)")

        # GPT 퀴즈 생성 — 챕터 내용으로 3문제
        quiz_list = _call_gpt_quiz(ch_segs, chapter["title"], questions_per_chapter)

        if not quiz_list:
            print(f"[TADAC] 챕터 {ch_idx+1} 퀴즈 생성 실패 — 건너뜀")
            continue

        for q in quiz_list:
            if not isinstance(q, dict) or "question" not in q:
                continue

            print(f"  Q{quiz_id_counter+1}: {q.get('question', '')[:50]}...")

            quizzes.append({
                "ai_quiz_index": quiz_id_counter,
                "chapter_index": ch_idx,
                "chapter_title": chapter["title"],
                "trigger_time":  round(trigger_time, 3),
                "segment_range": [seg_id_start, seg_id_end],
                "question":      q.get("question",    ""),
                "options":       q.get("options",     []),
                "answer_index":  q.get("answer_index", 0),
                "explanation":   q.get("explanation", ""),
            })
            quiz_id_counter += 1

    print(f"[TADAC] 퀴즈 생성 완료: {len(chapters)}개 챕터, 총 {len(quizzes)}개 문제")
    return quizzes
