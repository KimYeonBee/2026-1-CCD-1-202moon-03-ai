# -*- coding: utf-8 -*-
"""
fal.ai (Veo 2) + OpenAI TTS + ffmpeg를 이용하여
숏폼 영상(영상 + 나레이션 음성)을 자동 생성하고 병합하는 모듈.

pipeline.py에서 import하여 사용하거나, CLI로 직접 실행 가능.

사전 준비:
pip install fal-client openai requests
.env 파일에 FAL_KEY, OPENAI_API_KEY 추가
시스템에 ffmpeg 설치 필요
"""

import json
import os
import subprocess
import requests
import fal_client
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

FAL_MODEL = "fal-ai/veo2"
FAL_VIDEO_DURATION = os.getenv("FAL_VIDEO_DURATION", "8s")
TTS_MODEL = "tts-1-hd"
TTS_VOICE = "onyx"

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

NO_TEXT_SUFFIX = (
    " CRITICAL INSTRUCTION: The generated video must contain absolutely NO text, "
    "NO words, NO letters, NO numbers, NO signs, NO labels, NO subtitles, NO captions, "
    "NO watermarks, and NO written language in any script (Chinese, Korean, English, Japanese, or any other). "
    "Pure visual storytelling only — zero typography of any kind."
)


def _safe_prompt(prompt):
    return prompt + NO_TEXT_SUFFIX


def _fal_duration(value):
    value = str(value or "8s").strip()
    if value in {"5", "6", "7", "8"}:
        value = f"{value}s"
    if value not in {"5s", "6s", "7s", "8s"}:
        print(f"    [Video] 지원하지 않는 FAL_VIDEO_DURATION={value!r} → 8s 사용")
        return "8s"
    return value


def _get_duration(file_path):
    """ffprobe로 미디어 파일의 길이(초)를 반환."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            file_path,
        ],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return None


def generate_tts(text, output_path):
    """OpenAI TTS API로 한국어 나레이션 음성을 생성."""
    print(f"    [TTS] 음성 생성 중... ({text[:30]}...)")
    try:
        response = openai_client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
        )
        response.stream_to_file(output_path)
        print(f"    [TTS] 생성 완료: {output_path}")
    except Exception as e:
        print(f"    [TTS] 생성 실패: {e}")
        raise e


def generate_video(prompt, output_path):
    """fal.ai로 단일 비디오 클립을 생성하여 output_path에 저장."""
    print(f"    [Video] fal.ai 요청 중... (프롬프트: {prompt[:50]}...)")
    safe = _safe_prompt(prompt)
    try:
        result = fal_client.subscribe(
            FAL_MODEL,
            arguments={
                "prompt": safe,
                "aspect_ratio": "9:16",
                "duration": _fal_duration(FAL_VIDEO_DURATION),
            }
        )
        url = result['video']['url']

        res = requests.get(url, stream=True)
        res.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"    [Video] 생성 완료: {output_path}")
    except Exception as e:
        print(f"    [Video] 생성 실패: {e}")
        raise e


def merge_video_audio(video_path, audio_path, output_path):
    """
    영상 + TTS 나레이션을 합성한다.
    - 오디오가 영상보다 길면: 영상 마지막 프레임을 정지시켜 오디오 끝까지 연장
    - 오디오가 영상보다 짧으면: 오디오를 무음으로 채워 영상 길이를 유지
    - 최종 재생 호환성을 위해 AAC 44.1kHz stereo로 정규화
    """
    audio_dur = _get_duration(audio_path)
    video_dur = _get_duration(video_path)

    if audio_dur is None or video_dur is None:
        # 길이 측정 실패 시 -shortest 폴백
        cmd = [
            "ffmpeg", "-y", "-loglevel", "quiet",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "libx264", "-c:a", "aac",
            "-ar", "44100", "-ac", "2", "-b:a", "128k",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            output_path,
        ]
    elif audio_dur > video_dur:
        # 나레이션이 더 길면 → 영상을 tpad로 마지막 프레임 연장
        cmd = [
            "ffmpeg", "-y", "-loglevel", "quiet",
            "-i", video_path,
            "-i", audio_path,
            "-filter_complex",
            f"[0:v]tpad=stop_mode=clone:stop_duration={audio_dur - video_dur:.2f}[v]",
            "-map", "[v]", "-map", "1:a:0",
            "-c:v", "libx264", "-c:a", "aac",
            "-ar", "44100", "-ac", "2", "-b:a", "128k",
            "-shortest",
            output_path,
        ]
    else:
        # 영상이 더 길면 → 오디오는 무음으로 채우고 영상 길이는 유지
        cmd = [
            "ffmpeg", "-y", "-loglevel", "quiet",
            "-i", video_path,
            "-i", audio_path,
            "-filter_complex", "[1:a]apad[a]",
            "-c:v", "libx264", "-c:a", "aac",
            "-ar", "44100", "-ac", "2", "-b:a", "128k",
            "-map", "0:v:0", "-map", "[a]",
            "-shortest",
            output_path,
        ]

    print(f"    [Merge] 영상+음성 합성 중... (video={video_dur:.1f}s, audio={audio_dur:.1f}s)")
    subprocess.run(cmd, check=True)
    print(f"    [Merge] 합성 완료: {output_path}")


def build_chapter_videos(chapter_data, output_dir="shorts_rendered"):
    """
    단일 챕터의 4개 씬 영상을 생성하고,
    각 씬에 TTS 나레이션을 합성한 뒤 하나로 합본한다.

    Args:
        chapter_data: shorts_generator가 생성한 챕터별 dict
                      (scenes, chapter_index, chapter_title 포함)
        output_dir: 영상 저장 디렉토리

    Returns:
        str: 합본된 최종 영상 파일 경로. 실패 시 None.
    """
    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    ch_idx = chapter_data.get("chapter_index", 0)
    ch_title = chapter_data.get("chapter_title", f"chapter_{ch_idx}")
    scenes = chapter_data.get("scenes", [])

    if not scenes:
        print(f"[TADAC] 챕터 {ch_idx} 씬 없음 → 영상 생성 스킵")
        return None

    print(f"\n[TADAC] 챕터 {ch_idx} 영상+나레이션 생성 시작: '{ch_title}'")

    merged_files = []
    for scene in scenes:
        s_idx = scene["scene_number"]
        vp = scene["video_prompt"]
        narration = scene.get("narration", "")

        raw_video = os.path.join(tmp_dir, f"ch{ch_idx}_s{s_idx}_raw.mp4")
        tts_audio = os.path.join(tmp_dir, f"ch{ch_idx}_s{s_idx}_tts.mp3")
        merged = os.path.join(tmp_dir, f"ch{ch_idx}_s{s_idx}_merged.mp4")

        print(f"  - [씬 {s_idx}/4] 처리 시작")

        # 1. 영상 생성
        generate_video(vp, raw_video)

        # 2. TTS 나레이션 생성
        if narration:
            generate_tts(narration, tts_audio)

        # 3. 영상 + 나레이션 합성
        if narration and os.path.exists(tts_audio):
            merge_video_audio(raw_video, tts_audio, merged)

        final_scene = merged if os.path.exists(merged) else raw_video
        merged_files.append(final_scene)

    # 4. 챕터별 전체 씬 연결 (concat)
    concat_list_path = os.path.join(tmp_dir, f"ch{ch_idx}_concat.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for sf in merged_files:
            f.write(f"file '{os.path.abspath(sf)}'\n")

    final_output = os.path.join(output_dir, f"final_shorts_ch{ch_idx}.mp4")
    print(f"  [Finish] 챕터 {ch_idx} 최종 합본 렌더링 중 -> {final_output}")

    concat_cmd = [
        "ffmpeg", "-y", "-loglevel", "quiet",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-ar", "44100",
        "-ac", "2",
        "-b:a", "128k",
        "-movflags", "+faststart",
        final_output
    ]
    subprocess.run(concat_cmd, check=True)
    print(f"[TADAC] 챕터 {ch_idx} 쇼츠 영상 완료: {final_output}")
    return final_output


def build_all_shorts_videos(shorts_data, output_dir="shorts_rendered"):
    """
    전체 챕터의 숏폼 영상을 일괄 생성한다.

    Args:
        shorts_data: shorts_generator 결과 리스트 (None 항목 포함 가능)
        output_dir: 영상 저장 디렉토리

    Returns:
        list[str | None]: 챕터별 최종 영상 파일 경로
    """
    results = []
    valid_chapters = [ch for ch in shorts_data if ch is not None]

    if not valid_chapters:
        print("[TADAC] 숏폼 영상 생성 스킵: 유효한 챕터 데이터 없음")
        return results

    print(f"[TADAC] 숏폼 영상 일괄 생성 시작: {len(valid_chapters)}개 챕터")

    for chapter in valid_chapters:
        try:
            video_path = build_chapter_videos(chapter, output_dir=output_dir)
            results.append(video_path)
        except Exception as e:
            print(f"[TADAC] 챕터 {chapter.get('chapter_index', '?')} 영상 생성 실패: {e}")
            results.append(None)

    valid = sum(1 for r in results if r is not None)
    print(f"[TADAC] 숏폼 영상 생성 완료: {valid}/{len(valid_chapters)}개")
    return results


def main():
    json_path = "shorts_output.json"
    if not os.path.exists(json_path):
        print(f"파일을 찾을 수 없습니다: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        shorts_data = json.load(f)

    build_all_shorts_videos(shorts_data)


if __name__ == "__main__":
    main()
