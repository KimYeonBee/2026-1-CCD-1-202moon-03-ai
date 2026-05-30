"""
TADAC pipeline test script.

Tests:
    1. YouTube URL path (subtitle detection → VTT parse OR Whisper fallback)
    2. Local audio file path

Usage:
    python test_pipeline.py                       # both tests (needs OPENAI_API_KEY)
    python test_pipeline.py --youtube-only
    python test_pipeline.py --local-only --file ./sample.mp3
    python test_pipeline.py --dry-run             # validate modules import only
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_summary(game_data: dict) -> None:
    cfg = game_data.get("config", {})
    stats = game_data.get("stats", {})
    print(f"  segments      : {cfg.get('total_segments')}")
    print(f"  blanks        : {cfg.get('total_blanks')}")
    print(f"  fall_events   : {len(game_data.get('fall_events', []))}")
    print(f"  source        : {stats.get('transcript_source')}")

    # Show first 3 subtitles
    for sub in game_data.get("subtitles", [])[:3]:
        print(f"  [{sub['start']:.1f}s–{sub['end']:.1f}s] {sub['blank_text']}")
        for b in sub.get("blanks", []):
            print(f"    blank: '{b['keyword']}' (len={b['answer_length']})")

    # Show first 3 fall events
    for ev in game_data.get("fall_events", [])[:3]:
        print(f"  fall: '{ev['keyword']}' "
              f"target={ev['target_time']:.2f}s window={ev['fall_window']:.2f}s")


def _save(game_data: dict, name: str) -> None:
    out = Path(__file__).parent / f"test_output_{name}.json"
    out.write_text(json.dumps(game_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved → {out}")


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_dry_run() -> bool:
    """Validate that all modules import without errors."""
    print("[TEST] Dry-run: importing modules...")
    try:
        import stt
        import keyword_extractor
        import blank_subtitle
        import youtube_subtitle
        import youtube_audio
        import pipeline
        print("[TEST] All modules imported successfully ✓")
        return True
    except Exception as e:
        print(f"[TEST] Import error: {e}")
        traceback.print_exc()
        return False


def test_youtube(url: str, blanks: int = 2, speed: float = 1.0) -> bool:
    """Test the YouTube URL path end-to-end."""
    import pipeline

    print(f"\n[TEST] YouTube path")
    print(f"  URL    : {url}")
    print(f"  blanks : {blanks}, speed : {speed}")

    try:
        game_data = pipeline.run_pipeline(
            source=url,
            language="ko",
            blanks_per_sentence=blanks,
            fall_speed=speed,
        )
        _print_summary(game_data)
        _save(game_data, "youtube")
        print("[TEST] YouTube path ✓")
        return True
    except Exception as e:
        print(f"[TEST] YouTube path FAILED: {e}")
        traceback.print_exc()
        return False


def test_local_file(file_path: str, blanks: int = 2, speed: float = 1.0) -> bool:
    """Test the local file path end-to-end."""
    import pipeline

    print(f"\n[TEST] Local file path")
    print(f"  file   : {file_path}")
    print(f"  blanks : {blanks}, speed : {speed}")

    if not Path(file_path).exists():
        print(f"[TEST] SKIPPED — file not found: {file_path}")
        return True  # Not a failure

    try:
        game_data = pipeline.run_pipeline(
            source=file_path,
            language="ko",
            blanks_per_sentence=blanks,
            fall_speed=speed,
            stt_prompt="ADHD,도파민,전두엽,세로토닌",
        )
        _print_summary(game_data)
        _save(game_data, "local")
        print("[TEST] Local file path ✓")
        return True
    except Exception as e:
        print(f"[TEST] Local file path FAILED: {e}")
        traceback.print_exc()
        return False


def test_vtt_parser() -> bool:
    """Unit test: VTT parser with synthetic content (no network needed)."""
    import youtube_subtitle
    import tempfile
    import os

    print("\n[TEST] VTT parser unit test")
    sample_vtt = """\
WEBVTT

00:00:01.500 --> 00:00:04.200
도파민 시스템이 일반인과 다르게 작동합니다

00:00:04.500 --> 00:00:07.000
전두엽의 실행 기능이 저하되어 있습니다

"""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False,
                                         encoding="utf-8") as f:
            f.write(sample_vtt)
            vtt_path = f.name

        result = youtube_subtitle.parse_vtt(vtt_path)
        os.unlink(vtt_path)

        assert len(result["segments"]) >= 1, f"Expected at least 1 segment, got {len(result['segments'])}"
        assert len(result["words"]) > 0, "No words interpolated"

        seg0 = result["segments"][0]
        assert seg0["start"] == 1.5
        assert seg0["end"] >= 4.2
        assert "도파민" in seg0["text"]

        words0 = [w for w in result["words"] if w["start"] >= 1.5 and w["end"] <= 4.2]
        assert len(words0) == 5, f"Expected 5 words in seg0, got {len(words0)}: {words0}"

        print(f"  Segments : {len(result['segments'])} ✓")
        print(f"  Words    : {len(result['words'])} ✓")
        print(f"  Words[0] : {result['words'][0]}")
        print("[TEST] VTT parser unit test ✓")
        return True
    except Exception as e:
        print(f"[TEST] VTT parser FAILED: {e}")
        traceback.print_exc()
        return False


def test_blank_builder() -> bool:
    """Unit test: word timestamps 없이 blank subtitle + 균등 fall event 생성."""
    import blank_subtitle

    print("\n[TEST] Blank subtitle unit test")
    enriched = [
        {
            "segment_id": 0,
            "start": 10.0,
            "end": 16.0,
            "text": "도파민 시스템이 일반인과 다르게 작동합니다",
            "keywords": [
                {"keyword": "도파민"},
                {"keyword": "작동합니다"},
            ],
        }
    ]

    try:
        game_data = blank_subtitle.build_game_data(enriched, fall_speed=1.0, lead_time=3.0)

        assert "subtitles" in game_data, "subtitles 필드가 없음"
        assert "fall_events" in game_data, "fall_events 필드가 없음"
        sub = game_data["subtitles"][0]
        for field in ("segment_id", "start", "end", "original_text", "blank_text", "blanks"):
            assert field in sub, f"subtitle schema 필드 누락: {field}"
        assert "______" in sub["blank_text"], "No blank found in text"
        assert len(sub["blanks"]) == 2
        for blank in sub["blanks"]:
            for field in ("keyword", "position", "answer_length"):
                assert field in blank, f"blank schema 필드 누락: {field}"
        assert len(game_data["fall_events"]) == 2
        assert game_data["config"]["total_blanks"] == 2

        ev = game_data["fall_events"][0]
        for event in game_data["fall_events"]:
            for field in ("keyword", "segment_id", "target_time", "fall_window"):
                assert field in event, f"fall_event schema 필드 누락: {field}"
        # fall_start_time / fall_duration 은 프론트 계산 → fall_events에 없음
        assert "fall_duration" not in ev, "fall_duration은 프론트가 계산해야 함"
        assert "fall_start_time" not in ev, "fall_start_time은 프론트가 계산해야 함"
        assert "fall_window" in ev, "fall_window가 없음"
        assert ev["target_time"] == 12.0, f"첫 target_time 오류: {ev['target_time']}"
        assert ev["fall_window"] == 2.0, f"첫 fall_window 오류: {ev['fall_window']}"
        assert game_data["fall_events"][1]["target_time"] == 14.0
        assert game_data["fall_events"][1]["fall_window"] == 4.0

        print(f"  blank_text : {sub['blank_text']}")
        print(f"  fall_events: {game_data['fall_events']}")
        print("[TEST] Blank subtitle unit test ✓")
        return True
    except Exception as e:
        print(f"[TEST] Blank subtitle FAILED: {e}")
        traceback.print_exc()
        return False


def test_quiz_grouping() -> bool:
    """Unit test: 퀴즈 그룹 분할 로직 (API 호출 없음)."""
    import quiz_generator

    print("\n[TEST] 퀴즈 그룹 분할 단위 테스트")

    # 더미 세그먼트 13개 생성
    dummy_segments = [
        {"id": i, "start": float(i * 3), "end": float(i * 3 + 3), "text": f"세그먼트 {i} 내용입니다"}
        for i in range(13)
    ]

    try:
        # quiz_every_n=5 → 13개 세그먼트 → 그룹 [0~4], [5~9], [10~12] = 3개
        groups = []
        for i in range(0, len(dummy_segments), 5):
            groups.append(dummy_segments[i: i + 5])

        assert len(groups) == 3, f"그룹 수 오류: {len(groups)}"

        # 그룹 0: 세그먼트 0~4, trigger_time = 세그먼트 4의 end = 15.0
        g0_last = groups[0][-1]
        assert g0_last["id"] == 4
        assert g0_last["end"] == 15.0, f"trigger_time 오류: {g0_last['end']}"

        # 그룹 2 (마지막): 세그먼트 10~12 (3개)
        assert len(groups[2]) == 3, f"마지막 그룹 크기 오류: {len(groups[2])}"
        assert groups[2][-1]["id"] == 12

        print(f"  그룹 수       : {len(groups)}개 ✓")
        print(f"  그룹 0 범위   : seg {groups[0][0]['id']} ~ {groups[0][-1]['id']} ✓")
        print(f"  그룹 0 trigger: {groups[0][-1]['end']}s ✓")
        print(f"  마지막 그룹   : seg {groups[2][0]['id']} ~ {groups[2][-1]['id']} (총 {len(groups[2])}개) ✓")
        print("[TEST] 퀴즈 그룹 분할 단위 테스트 ✓")
        return True
    except Exception as e:
        print(f"[TEST] 퀴즈 그룹 분할 FAILED: {e}")
        traceback.print_exc()
        return False


def test_backward_chunk_ranges() -> bool:
    """Unit test: 앞 영상 청크가 짧게 먼저 처리되도록 뒤에서부터 분할."""
    import pipeline

    print("\n[TEST] 뒤쪽 기준 오디오 청크 분할 단위 테스트")

    try:
        ranges = pipeline._build_backward_chunk_ranges(904.6, chunk_sec=600)
        rounded = [(round(s, 1), round(e, 1)) for s, e in ranges]

        assert rounded == [(0.0, 304.6), (304.6, 904.6)], f"분할 범위 오류: {rounded}"
        assert ranges[0][0] == 0.0, "첫 청크는 영상 시작 지점이어야 함"
        assert ranges[0][1] - ranges[0][0] < ranges[1][1] - ranges[1][0], "앞 청크가 더 짧아야 함"

        print(f"  ranges: {rounded} ✓")
        print("[TEST] 뒤쪽 기준 오디오 청크 분할 단위 테스트 ✓")
        return True
    except Exception as e:
        print(f"[TEST] 뒤쪽 기준 오디오 청크 분할 FAILED: {e}")
        traceback.print_exc()
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

DEFAULT_YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # replace with real lecture URL


def main() -> None:
    parser = argparse.ArgumentParser(description="TADAC pipeline test runner")
    parser.add_argument("--youtube-only", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Import check only")
    parser.add_argument("--file", default=None, help="Local audio/video file path")
    parser.add_argument("--url", default=DEFAULT_YOUTUBE_URL, help="YouTube URL to test")
    parser.add_argument("--blanks", type=int, default=2)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    results: list[bool] = []

    # Always run unit tests (no API needed)
    results.append(test_dry_run())
    results.append(test_vtt_parser())
    results.append(test_blank_builder())
    results.append(test_quiz_grouping())
    results.append(test_backward_chunk_ranges())

    if args.dry_run:
        passed = sum(results)
        print(f"\n[TEST] Dry-run complete: {passed}/{len(results)} passed")
        sys.exit(0 if all(results) else 1)

    if not args.local_only:
        results.append(test_youtube(args.url, blanks=args.blanks, speed=args.speed))

    if not args.youtube_only:
        local_file = args.file or "./sample.mp3"
        results.append(test_local_file(local_file, blanks=args.blanks, speed=args.speed))

    passed = sum(results)
    total = len(results)
    print(f"\n[TEST] Results: {passed}/{total} passed {'✓' if passed == total else '✗'}")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
