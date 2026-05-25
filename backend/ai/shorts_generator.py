# -*- coding: utf-8 -*-
"""
숏폼 대본 + 영상 AI 프롬프트 생성 모듈

챕터별 교정 완료된 자막 텍스트를 받아:
1. 가장 임팩트 있는 핵심 포인트를 추출
2. 30초 한국어 숏폼 나레이션 대본 작성
3. Google Flow (Veo 3.1) 용 영문 영상 프롬프트 생성
"""

import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """당신은 복잡한 지식을 대중에게 가장 흥미로운 스토리로 번역해 내는 천재적인 콘텐츠 디렉터입니다.

제시된 [강의 자막]은 시청자가 이미 한 번 시청한 긴 강의의 챕터 중 일부입니다. 이 숏폼의 목적은 '시선을 끄는 것'이 아니라, 시청자가 학습한 내용을 '절대 잊지 못하게 뇌리에 박아주는(복습) 것'입니다.

[Step 1: 핵심 요약 + 뇌 각인용 메타포(비유) 설정]

1. 내용 압축: 이 챕터에서 시청자가 반드시 기억해야 할 핵심 원리나 인과관계를 빠르고 타격감 있게 요약하라.
2. 뇌 각인용 비유(Mnemonic Metaphor) 매칭: 뻔한 설명 대신, 절대 잊을 수 없는 극단적이고 코믹한 비유를 씌워라.
   (예: 화학의 산화-환원 반응 → 지독한 채무자와 사채업자의 관계 / 정치의 삼권분립 → 서로 멱살 잡고 싸우는 세쌍둥이)
3. 훅(Hook) 제거: 어그로를 끄는 서론은 버리고, 곧바로 본론(비유적 상황)으로 돌진하라.

[Step 2: 결과물 출력]

★★ 중요: Google Veo 3.1은 클립당 최대 8초입니다. 따라서 30초 숏폼은 4개의 씬(scene)으로 나눠야 합니다.

위 분석을 바탕으로 아래 JSON 형식으로 출력하라.

{
  "concept": "기획 의도 — 복습을 위해 선택한 극단적 비유와 그 이유를 1~2줄로 설명",
  "opening": "대본의 첫 문장 (헛웃음이 나오는 비유적 상황을 즉각 제시하는 도입부)",
  "script": "30초 숏폼 대본 전문 (한국어, 성우 나레이션 기준 30초 이내, 공백 포함 150~200자). opening으로 시작하여 숨 쉴 틈 없이 빠른 템포로 챕터 내용을 비유를 통해 요약할 것.",
  "scenes": [
    {
      "scene_number": 1,
      "narration": "이 씬의 한국어 나레이션 (15~20자, 약 3~5초 분량)",
      "video_prompt": "Veo 3.1용 영문 프롬프트. 반드시 아래 구조를 따를 것:\n- Visual: 화면에 보이는 장면, 캐릭터, 동작을 구체적으로 묘사\n- Camera: 카메라 앵글과 움직임 (close-up, wide shot, tracking 등)\n- Audio: Warm Korean male/female voiceover says: \\\"(narration을 영어로 번역한 대사)\\\". 배경 효과음도 명시\n- Style: 조명, 색감, 분위기 (cinematic, animated, etc.)"
    },
    {"scene_number": 2, "narration": "...", "video_prompt": "..."},
    {"scene_number": 3, "narration": "...", "video_prompt": "..."},
    {"scene_number": 4, "narration": "...", "video_prompt": "..."}
  ],
  "memory_point": "이 숏폼을 보고 나면 시청자의 머릿속에 남아야 할 단 하나의 핵심 암기 문장"
}

★ 절대 규칙 (하나라도 어기면 실패):
1. 반드시 유효한 JSON만 출력하라.
2. script는 공백 포함 150~200자를 엄수하라.
3. scenes 배열에 반드시 정확히 4개의 씬 객체를 포함하라. 3개 이하거나 5개 이상이면 실패다.
4. 각 씬의 narration을 순서대로 합치면 script 전문과 일치해야 한다.
5. 각 씬의 video_prompt는 반드시 Visual / Camera / Audio / Style 4개 섹션을 포함하라.
6. Audio 섹션에 반드시 says: "영어 대사" 형태로 나레이션을 포함하라. 이것이 없으면 영상에 음성이 생성되지 않는다.
7. says: 뒤의 영어 대사는 15~20단어 이내로 제한하라 (Veo 3.1의 8초 클립 제약).
8. 새로운 지식을 추가하지 말고, 오직 주어진 자막 내용의 '강렬한 요약과 복습'에만 집중하라."""


def generate_shorts_prompt(chapter_title, chapter_text, topic_summary=""):
    """
    챕터 자막 텍스트로부터 숏폼 대본 + 영상 프롬프트를 생성한다.

    Args:
        chapter_title: 챕터 제목
        chapter_text: 교정 완료된 챕터 자막 전문
        topic_summary: 전체 강의 주제 요약 (맥락 제공용)

    Returns:
        dict: {concept, opening, script, video_prompt, memory_point}
        실패 시 None
    """
    if not chapter_text or len(chapter_text.strip()) < 30:
        print(f"[TADAC] 숏폼 생성 스킵: 챕터 '{chapter_title}' 텍스트 부족")
        return None

    context_line = f"강의 주제: {topic_summary}\n" if topic_summary else ""
    user_prompt = f"""{context_line}챕터 제목: {chapter_title}

[강의 자막 전문]
{chapter_text}"""

    max_retries = 3
    base_delay = 2

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.7,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
            )

            raw = response.choices[0].message.content
            data = json.loads(raw)

            usage = response.usage
            if usage:
                print(f"[TADAC] 숏폼 토큰 — input: {usage.prompt_tokens}, output: {usage.completion_tokens}")

            required = {"concept", "opening", "script", "scenes", "memory_point"}
            if not required.issubset(data.keys()):
                missing = required - data.keys()
                print(f"[TADAC] 숏폼 결과 필드 누락: {missing} → 재시도 ({attempt+1}/{max_retries})")
                time.sleep(base_delay)
                continue

            return data

        except json.JSONDecodeError as e:
            print(f"[TADAC] 숏폼 JSON 파싱 오류: {e} → 재시도 ({attempt+1}/{max_retries})")
            time.sleep(base_delay)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                delay = base_delay * (2 ** attempt)
                print(f"[TADAC] Rate limit → {delay}초 대기 ({attempt+1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"[TADAC] 숏폼 생성 오류: {e}")
                time.sleep(base_delay)

    print(f"[TADAC] 숏폼 생성 실패: 챕터 '{chapter_title}'")
    return None


def generate_shorts_for_chapters(chapters, enriched_segments_by_chapter, topic_summary=""):
    """
    전체 챕터에 대해 숏폼 프롬프트를 일괄 생성한다.

    Args:
        chapters: [{"title": ..., "start": ..., "end": ...}, ...]
        enriched_segments_by_chapter: 챕터별 교정 완료 세그먼트 리스트의 리스트
        topic_summary: 전체 강의 주제 요약

    Returns:
        list[dict | None]: 챕터별 숏폼 결과
    """
    results = []
    for ch_idx, (chapter, ch_segs) in enumerate(zip(chapters, enriched_segments_by_chapter)):
        chapter_text = " ".join(seg.get("text", "") for seg in ch_segs)
        print(f"[TADAC] 숏폼 생성 {ch_idx+1}/{len(chapters)}: '{chapter['title']}'")

        result = generate_shorts_prompt(
            chapter_title=chapter["title"],
            chapter_text=chapter_text,
            topic_summary=topic_summary,
        )
        if result:
            result["chapter_index"] = ch_idx
            result["chapter_title"] = chapter["title"]
        results.append(result)

    valid = sum(1 for r in results if r is not None)
    print(f"[TADAC] 숏폼 생성 완료: {valid}/{len(chapters)}개 챕터")
    return results
