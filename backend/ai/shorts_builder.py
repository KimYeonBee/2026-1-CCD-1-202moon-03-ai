# -*- coding: utf-8 -*-
"""
fal.ai 와 OpenAI TTS, ffmpeg를 이용하여 숏폼 영상을 자동 생성하고 병합하는 스크립트.

사전 준비:
pip install fal-client openai requests
.env 파일에 FAL_KEY=... 추가 (fal.ai API key)
시스템에 ffmpeg 설치 필요
"""

import json
import os
import subprocess
import requests
import fal_client
from dotenv import load_dotenv

load_dotenv()

# 영상 생성을 위한 fal.ai 모델 (가장 무난하고 퀄리티 좋은 Kling v1 표준 모델)
FAL_MODEL = "fal-ai/kling-video/v1/standard/text-to-video"

def generate_video(prompt, output_path):
    print(f"    [Video] fal.ai 요청 중... (프롬프트: {prompt[:30]}...)")
    try:
        # fal-client 동기 호출 API
        result = fal_client.subscribe(
            FAL_MODEL,
            arguments={
                "prompt": prompt,
                "aspect_ratio": "9:16",
                "duration": "5"  # 기본 5초 생성 (씬당 길이)
            }
        )
        url = result['video']['url']
        
        # 파일 다운로드
        res = requests.get(url, stream=True)
        res.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        print(f"    [Video] 생성 실패: {e}")
        raise e

def main():
    json_path = "shorts_output.json"
    if not os.path.exists(json_path):
        print(f"파일을 찾을 수 없습니다: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        shorts_data = json.load(f)

    # 출력 폴더 생성 (챕터별 결과물 및 임시 파일)
    output_dir = "shorts_rendered"
    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    for chapter in shorts_data:
        ch_idx = chapter.get("chapter_index", 0)
        ch_title = chapter.get("chapter_title", f"chapter_{ch_idx}")
        print(f"\n▶ 챕터 {ch_idx} 영상 생성 시작: '{ch_title}'")
        
        scenes = chapter.get("scenes", [])
        final_scene_files = []
        
        for scene in scenes:
            s_idx = scene["scene_number"]
            vp = scene["video_prompt"]
            
            # 임시 파일 세팅
            raw_video = os.path.join(tmp_dir, f"ch{ch_idx}_s{s_idx}_raw.mp4")
            
            print(f"  - [씬 {s_idx}/4] 영상 생성 시작")
            
            # 1. 영상 생성
            if not os.path.exists(raw_video):
                generate_video(vp, raw_video)
                
            final_scene_files.append(raw_video)
        
        # 2. 챕터별 4개 씬 연결 (concat)
        concat_list_path = os.path.join(tmp_dir, f"ch{ch_idx}_concat.txt")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for sf in final_scene_files:
                f.write(f"file '{os.path.abspath(sf)}'\n")
                
        final_output = os.path.join(output_dir, f"final_shorts_ch{ch_idx}.mp4")
        print(f"  [🏁 Finish] 챕터 {ch_idx} 최종 합본 렌더링 중 -> {final_output}")
        
        concat_cmd = [
            "ffmpeg", "-y", "-loglevel", "quiet",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            final_output
        ]
        subprocess.run(concat_cmd, check=True)
        print(f"✅ 챕터 {ch_idx} 쇼츠 영상 완료!\n")

if __name__ == "__main__":
    main()
