# TADAC AI Backend Handoff

백엔드가 AI 서버를 붙일 때 필요한 최소 사용 명세입니다. 상세 응답 스키마는 `AI_SPEC.md`를 기준으로 보면 됩니다.

## 실행 환경

- Python 3.10+
- `pip install -r requirements.txt`
- 시스템 의존성: `ffmpeg`, `yt-dlp`
- 환경변수: `OPENAI_API_KEY`

```bash
OPENAI_API_KEY=sk-...
python api.py
```

기본 서버 주소는 `http://localhost:8000`입니다.

## 엔드포인트

### Health Check

```http
GET /api/health
```

성공 예시:

```json
{
  "status": "ok",
  "api_key_configured": true
}
```

### YouTube URL 처리

```http
POST /api/process-url
Content-Type: application/json
```

Request:

```json
{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "language": "ko",
  "stt_prompt": null,
  "refine": true
}
```

Response: `game_data` JSON 전체를 반환합니다. 백엔드는 그대로 DB에 저장하고 프론트에 내려주면 됩니다.

주의:

- `list=...&index=...`가 붙은 YouTube URL도 단일 영상만 처리하도록 AI 쪽에서 `--no-playlist` 처리했습니다.
- YouTube 수동 자막만 직접 사용합니다. 자동 자막만 있으면 품질 이슈로 Whisper STT를 사용합니다.
- 긴 영상은 수 분 이상 걸릴 수 있으므로 백엔드 타임아웃을 넉넉하게 잡거나 비동기 job 처리 권장.
- `refine=false`는 GPT 교정을 생략해서 빠르지만 품질이 떨어질 수 있습니다.

### 파일 업로드 처리

```http
POST /api/process
Content-Type: multipart/form-data
```

Form fields:

| field | required | default | description |
|---|---:|---|---|
| `file` | yes | - | 영상 파일: `.mp4`, `.webm` |
| `language` | no | `ko` | STT 언어 코드 |
| `stt_prompt` | no | `null` | Whisper 용어 힌트 |
| `refine` | no | `true` | GPT 교정 여부 |

현재 백엔드 연동 범위를 줄이기 위해 업로드 API는 영상 파일만 허용합니다.
오디오 파일 처리 로직은 파이프라인 내부에 남아 있지만 HTTP API에서는 받지 않습니다.

## game_data 저장 단위

백엔드는 아래 최상위 필드를 저장하면 됩니다.

```json
{
  "subtitles": [],
  "fall_events": [],
  "quizzes": [],
  "config": {},
  "stats": {}
}
```

필수 게임 필드:

- `subtitles`: 자막 구간과 빈칸 텍스트
- `fall_events`: 키워드 낙하 타이밍
- `quizzes`: 중간 퀴즈
- `config`: 총 segment/blank 수

`stats`는 디버그/로깅용입니다. DB에 저장해도 되고, 프론트 응답에서 제외해도 됩니다.

## 프론트 계산 책임

AI는 난이도를 계산하지 않습니다. 프론트가 `fall_events[].fall_window`와 `target_time`으로 낙하 타이밍을 계산합니다.

```js
fall_duration = event.fall_window / fallSpeed
fall_start_time = event.target_time - fall_duration
```

빈칸 난이도도 프론트가 `subtitles[].blanks` 중 앞 N개만 활성화하는 방식으로 처리합니다.

## 최근 반영된 주의사항

- YouTube playlist URL 오처리 방지: `--no-playlist` 적용.
- GPT 고유명사 교정 과잉 방지:
  - `부산시 -> 부산`
  - `삼성 회장님 -> 이재용`
  - `울프 -> 울프 아저씨`
  - `LCT -> 엘시티`
  같은 의미 추론/정규화/축약/확장은 금지했습니다.
- 실제 STT 오인식처럼 발음과 길이가 가까운 교정만 허용합니다.

## 오류 처리

| status | meaning |
|---:|---|
| `400` | 지원하지 않는 파일 형식 또는 YouTube URL 아님 |
| `422` | 세그먼트 추출 실패 |
| `500` | OpenAI, yt-dlp, ffmpeg 등 처리 실패 |

백엔드는 실패 시 `detail` 메시지를 로깅하면 됩니다.
