# -*- coding: utf-8 -*-
"""
빈칸 자막 + 낙하 이벤트 생성 모듈

- 키워드 위치를 "______"으로 치환한 빈칸 자막 생성
- 낙하 이벤트에는 target_time(키워드 발화 시점)만 포함
  → fall_start_time / fall_duration 은 프론트엔드가 자체 난이도 설정으로 계산

프론트엔드 계산 공식:
    fall_duration   = lead_time / fall_speed          (프론트 설정값)
    fall_start_time = target_time - fall_duration
"""

import re

from transcript_refiner import filter_filler_keywords


# ── 빈칸 밀도 제어 ────────────────────────────────────────────────────────────
# 사용자가 자막을 보고 빈칸을 타이핑할 시간을 확보하기 위한 두 가지 캡:
#   1) 짧은 세그먼트는 빈칸 1개만 — 자막이 너무 빨리 사라져 입력 시간이 부족
#   2) 인접 빈칸 사이 최소 간격 — 키워드가 몰려서 발화되면 따라잡을 수 없음

SHORT_SEG_THRESHOLD = 2.5    # 자막 길이(초) — 이하면 빈칸 1개로 제한
MIN_BLANK_GAP       = 1.8    # 인접 빈칸의 발화 시점 최소 간격(초)
MAX_KEYWORD_LEN     = 5      # 빈칸 키워드 최대 글자 수


def _apply_density_limits(enriched_segments):
    # 0단계 — stopword 안전망: 캡 적용 전에 generic word 먼저 제거
    # (transcript_refiner / keyword_extractor 단에서 이미 필터링되지만,
    #  combined_processor 등 다른 경로에서 새어나올 가능성 차단)
    dropped_stop = 0
    for seg in enriched_segments:
        original = seg.get("keywords", [])
        filtered = filter_filler_keywords(original)
        dropped_stop += len(original) - len(filtered)
        seg["keywords"] = filtered

    # 0.5단계 — 키워드 길이 제한: MAX_KEYWORD_LEN자 초과 키워드 제거
    dropped_long = 0
    for seg in enriched_segments:
        original = seg.get("keywords", [])
        filtered = [kw for kw in original
                    if len(kw["keyword"] if isinstance(kw, dict) else kw) <= MAX_KEYWORD_LEN]
        dropped_long += len(original) - len(filtered)
        seg["keywords"] = filtered

    # 1단계 — 세그먼트별 캡: 짧은 자막은 빈칸 1개만
    dropped_short = 0
    for seg in enriched_segments:
        duration   = seg.get("end", 0.0) - seg.get("start", 0.0)
        max_blanks = 1 if duration < SHORT_SEG_THRESHOLD else 2

        keywords = seg.get("keywords", [])
        if len(keywords) > max_blanks:
            sorted_kws = sorted(keywords, key=lambda k: k.get("start", seg.get("start", 0.0)))
            dropped_short  += len(keywords) - max_blanks
            seg["keywords"] = sorted_kws[:max_blanks]

    # 2단계 — 전역 최소 간격: 인접 빈칸이 너무 가까우면 뒤쪽 drop
    flat = []  # (target_time, seg_idx, kw_idx)
    for seg_idx, seg in enumerate(enriched_segments):
        for kw_idx, kw in enumerate(seg.get("keywords", [])):
            target = kw.get("start", seg.get("start", 0.0))
            flat.append((target, seg_idx, kw_idx))
    flat.sort(key=lambda x: x[0])

    keep      = set()
    last_kept = -float("inf")
    for target, seg_idx, kw_idx in flat:
        if target - last_kept >= MIN_BLANK_GAP:
            keep.add((seg_idx, kw_idx))
            last_kept = target

    dropped_gap = 0
    for seg_idx, seg in enumerate(enriched_segments):
        original = seg.get("keywords", [])
        filtered = [kw for kw_idx, kw in enumerate(original) if (seg_idx, kw_idx) in keep]
        dropped_gap   += len(original) - len(filtered)
        seg["keywords"] = filtered

    if dropped_stop or dropped_long or dropped_short or dropped_gap:
        print(f"[TADAC] 빈칸 밀도 제어: stopword {dropped_stop}개, "
              f"길이초과(>{MAX_KEYWORD_LEN}자) {dropped_long}개, "
              f"짧은 세그먼트 {dropped_short}개, 최소 간격 {dropped_gap}개 제거")

    return enriched_segments


# ── 빈칸 자막 만들기 ──────────────────────────────────────────────────────────
# 키워드 위치를 찾아서 뒤에서부터 "______"으로 교체 (인덱스 밀림 방지)

def _make_blank_text(original_text, keywords):
    text         = original_text
    kw_positions = []  # (텍스트 내 위치, 키워드, 키워드 정보)

    # 각 키워드의 텍스트 내 위치 탐색
    for kw_info in keywords:
        keyword = kw_info["keyword"]
        idx     = text.find(keyword)  # 정확 일치 탐색

        if idx == -1:
            # 정확 일치 실패 → 정규식으로 재탐색
            for match in re.finditer(re.escape(keyword), text):
                idx = match.start()
                break

        if idx != -1:
            kw_positions.append((idx, keyword, kw_info))

    # 뒤에서부터 교체해야 앞 키워드 위치가 밀리지 않음
    kw_positions.sort(key=lambda x: x[0], reverse=True)

    for idx, keyword, _ in kw_positions:
        text = text[:idx] + "______" + text[idx + len(keyword):]

    # 빈칸 메타데이터 — 왼쪽부터 순서대로 기록
    blanks      = []
    blank_count = 0
    for _, keyword, _ in sorted(kw_positions, key=lambda x: x[0]):
        blanks.append({
            "keyword":       keyword,
            "position":      blank_count,   # 왼쪽부터 몇 번째 빈칸인지
            "answer_length": len(keyword),  # 정답 글자 수 (힌트 표시용)
        })
        blank_count += 1

    return text, blanks


# ── 낙하 이벤트 생성 ──────────────────────────────────────────────────────────
# AI는 target_time(발화 시점)과 fall_window(자막 시작~발화 구간)를 제공
# fall_start_time / fall_duration 은 프론트엔드가 난이도 설정으로 계산
#
# fall_window: target_time - segment.start
#   → 자막이 화면에 뜬 시점부터 키워드가 발화되기까지의 시간
#   → 이 구간 안에서만 키워드가 낙하해야 자막 뜨기 전에 도착하는 버그가 없음
#
# 프론트엔드 계산 공식:
#   fall_duration   = fall_window / fall_speed
#   fall_start_time = target_time - fall_duration

def _make_fall_event(keyword, keyword_timestamp, segment_id, segment_start):
    fall_window = round(keyword_timestamp - segment_start, 3)  # 자막 시작부터 발화까지 구간
    return {
        "keyword":     keyword,
        "target_time": round(keyword_timestamp, 3),  # 키워드 발화 시점 = 낙하 목표 지점
        "fall_window": max(fall_window, 0.5),         # 최소 0.5초 보장 (너무 짧으면 낙하 불가)
        "segment_id":  segment_id,
    }


# ── 메인 함수 ─────────────────────────────────────────────────────────────────
# 키워드가 붙은 세그먼트 목록 → 프론트엔드용 game_data 딕셔너리 생성

def build_game_data(enriched_segments, fall_speed=1.0, lead_time=3.0):
    # fall_speed, lead_time 은 하위 호환을 위해 파라미터로 받지만 사용하지 않음
    # (프론트엔드가 자체 계산)

    # 빈칸 밀도 제어 — 짧은 세그먼트 캡 + 전역 최소 간격
    enriched_segments = _apply_density_limits(enriched_segments)

    subtitles    = []
    fall_events  = []
    total_blanks = 0

    for seg in enriched_segments:
        keywords = seg.get("keywords", [])

        # 빈칸 자막 텍스트 + 빈칸 메타데이터 생성
        blank_text, blanks = _make_blank_text(seg["text"], keywords)

        subtitles.append({
            "segment_id":    seg["segment_id"],
            "start":         seg["start"],
            "end":           seg["end"],
            "original_text": seg["text"],    # 정답이 보이는 원본 자막
            "blank_text":    blank_text,      # "______" 처리된 자막
            "blanks":        blanks,          # 빈칸 위치 + 정답 정보 (최대 2개)
        })
        total_blanks += len(blanks)

        # 키워드별 낙하 이벤트 — target_time + fall_window 포함
        for kw_info in keywords:
            target_time = kw_info.get("start", seg["start"])  # 키워드 발화 시점
            event = _make_fall_event(
                keyword           = kw_info["keyword"],
                keyword_timestamp = target_time,
                segment_id        = seg["segment_id"],
                segment_start     = seg["start"],  # 자막 시작 시점 (fall_window 계산용)
            )
            fall_events.append(event)

    # 발화 시점 기준으로 정렬 (프론트엔드가 순서대로 처리)
    fall_events.sort(key=lambda e: e["target_time"])

    game_data = {
        "subtitles":   subtitles,
        "fall_events": fall_events,
        "config": {
            "max_blanks_per_sentence": 2,                       # AI가 생성한 최대 빈칸 수
            "short_seg_threshold":     SHORT_SEG_THRESHOLD,     # 이 이하 자막은 빈칸 1개로 캡
            "min_blank_gap":           MIN_BLANK_GAP,           # 인접 빈칸 최소 간격 (초)
            "total_blanks":            total_blanks,
            "total_segments":          len(subtitles),
        },
    }

    print(f"[TADAC] 게임 데이터 생성 완료: "
          f"자막 {len(subtitles)}개, 낙하 이벤트 {len(fall_events)}개, 빈칸 {total_blanks}개")

    return game_data
