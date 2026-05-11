# TADAC AI 명세서

> **대상**: 백엔드 파트
> **AI 서버 주소**: `http://localhost:8000` (백엔드가 내부에서 호출, 프론트엔드 직접 접근 불가)
> **공통 Content-Type**: `application/json` (파일 업로드는 `multipart/form-data`)

---

## 역할 분리 한눈에 보기

```
[프론트엔드]
     │  영상 URL / 파일 업로드
     ▼
[백엔드 서버]  ← 프론트는 백엔드만 호출
     │  내부에서 AI 서버 호출
     ▼
[AI 서버 :8000]  ← 이 문서가 설명하는 영역
     │  game_data JSON 반환
     ▼
[백엔드 서버]
     │  DB 저장 후 프론트에 응답
     ▼
[프론트엔드]
     │  game_data 기반으로 게임 렌더링
     │  난이도 파라미터 자체 관리 (실시간)
```

| 항목 | AI | 백엔드 | 프론트엔드 |
|---|---|---|---|
| 음성 → 텍스트 변환 (Whisper) | ✅ | ❌ | ❌ |
| 키워드 추출 (GPT) | ✅ | ❌ | ❌ |
| game_data JSON 생성 | ✅ | ❌ | ❌ |
| game_data DB 저장·조회 | ❌ | ✅ | ❌ |
| 콘텐츠 메타데이터 관리 | ❌ | ✅ | ❌ |
| 사용자 인증 / 세션 | ❌ | ✅ | ❌ |
| 학습 이력·점수 저장 | ❌ | ✅ | ❌ |
| 난이도 조절 (빈칸 수, 낙하 속도) | ❌ | ❌ | ✅ |

### 난이도 파라미터를 프론트엔드가 관리하는 이유

AI는 콘텐츠를 **한 번만 처리**해서 DB에 저장한다.
난이도는 학습자가 플레이하는 동안 **실시간**으로 바뀌어야 하기 때문에,
매번 AI를 재호출하는 건 비효율적이다.

```
AI 생성 (한 번, DB 저장)
  blanks: ["도파민", "전두엽"]  ← 세그먼트당 최대 2개 생성
  target_time: 각 키워드의 발화 시점

프론트엔드 (실시간, DB 재호출 없음)
  초급: blanks 중 1개만 활성화, fall_speed=0.7
  중급: 2개 활성화,              fall_speed=1.0
  고급: 2개 전부 활성화,         fall_speed=2.0

  fall_start_time = target_time - (lead_time / fall_speed)  ← 프론트가 직접 계산
```

---
---

# 1. 영상 파일 업로드 처리

## 기능 설명

백엔드가 영상 파일을 전달하면, AI가 오디오를 추출한 뒤 음성 인식 → 키워드 추출 → 빈칸 자막 게임 데이터를 생성해서 반환한다.

## AI 구현 주의사항

- 파일 크기 **25MB 초과** 시 자동으로 10분 단위 분할 처리 후 병합 (백엔드 별도 처리 불필요)
- 처리 시간이 길 수 있으므로 백엔드에서 **비동기 또는 타임아웃 설정** 권장 (영상 길이에 따라 1~5분 소요)
- `refine=false` 옵션으로 GPT 교정 단계를 스킵하면 처리 속도 단축 가능
- 난이도 파라미터(`blanks_per_sentence`, `fall_speed`, `lead_time`)는 요청 시 전달 불필요
  - AI는 세그먼트당 최대 빈칸 2개를 생성
  - `fall_speed`, `lead_time`은 프론트엔드가 자체 난이도 계산에 사용

---

## Request (요청)

### Headers

```json
{
  "Content-Type": "multipart/form-data"
}
```

### Request Body (Form-data)

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|
| `file` | File | ✅ | — | 영상 파일 |
| `language` | string | ❌ | `"ko"` | STT 언어 코드 |
| `stt_prompt` | string | ❌ | `null` | Whisper 전문 용어 힌트 |
| `refine` | boolean | ❌ | `true` | Whisper 결과 GPT 교정 여부 |

**지원 파일 형식**: `.mp4` `.webm`

### Request Body 예시

```bash
curl -X POST http://localhost:8000/api/process \
  -F "file=@lecture.mp4" \
  -F "language=ko" \
  -F "refine=true"
```

---

## Response (응답)

### 성공 응답

- **Status**: `200 OK`

### Response Body (JSON)

```json
{
  "ai_summary": "이 강의는 ADHD의 신경생리학적 기제를 다룬다.\n\n## 챕터 1 제목\n- 핵심 개념 1\n- 핵심 개념 2\n\n## 챕터 2 제목\n핵심 내용을 정리한 3~6문장 분량의 복습 노트.",
  "subtitles": [
    {
      "segment_id": 0,
      "start": 0.0,
      "end": 5.2,
      "original_text": "도파민 시스템이 일반인과 다르게 작동합니다",
      "blank_text": "______ 시스템이 일반인과 다르게 ______",
      "blanks": [
        { "keyword": "도파민",    "position": 0, "answer_length": 3 },
        { "keyword": "작동합니다","position": 1, "answer_length": 5 }
      ]
    }
  ],
  "fall_events": [
    {
      "keyword":     "도파민",
      "target_time": 3.5,
      "fall_window": 3.5,
      "segment_id":  0
    }
  ],
  "quizzes": [
    {
      "quiz_id":       0,
      "trigger_time":  14.5,
      "segment_range": [0, 4],
      "question":      "구석기 시대 사람들의 생활 방식으로 옳은 것은?",
      "options": [
        "정착 생활을 하며 농사를 지었다",
        "이동 생활을 하며 사냥과 채집을 했다",
        "빗살무늬 토기를 만들어 사용했다",
        "철제 도구를 사용하여 농사를 지었다"
      ],
      "answer_index": 1,
      "correct_feedback": "정답입니다! 구석기 시대는 이동 생활을 하며 사냥과 채집으로 식량을 구했습니다.",
      "incorrect_feedback": "아쉽지만 틀렸습니다. 구석기 시대는 이동 생활을 하며 사냥과 채집으로 식량을 구했습니다."
    }
  ],
  "config": {
    "max_blanks_per_sentence": 2,
    "total_blanks":            42,
    "total_segments":          15
  },
  "stats": {
    "transcript_source": "whisper",
    "total_words":       312,
    "language":          "ko",
    "gpt_refined":       true,
    "total_quizzes":     4
  }
}
```

### Response Body 필드 설명

**`ai_summary`** — 영상 복습용 마크다운 요약 (string)

| 필드 | 타입 | 설명 |
|---|---|---|
| `ai_summary` | string | 강의 전체 한 줄 주제 + 챕터별 복습 요약을 마크다운으로 합친 문자열. 챕터 헤더는 `## {챕터 제목}` 형식. 챕터 요약은 챕터당 3~6문장 또는 3~5개 불릿. `refine=false` 또는 수동 자막 경로에서는 한 줄 주제만 들어감. 영상 길이/내용에 따라 비어 있을 수 있음(`""`). |

> 프론트엔드는 학습 완료 후 "복습 노트" 패널에 그대로 렌더링하면 됨.
> 백엔드는 그대로 DB에 저장 권장 (재생성 비용이 비쌈).

**`subtitles[]`** — 자막 세그먼트 목록

| 필드 | 타입 | 설명 |
|---|---|---|
| `segment_id` | int | 세그먼트 번호 (0부터 순서대로) |
| `start` | float | 시작 시간(초) |
| `end` | float | 종료 시간(초) |
| `original_text` | string | 정답이 보이는 원본 자막 |
| `blank_text` | string | `______` 처리된 빈칸 자막 |
| `blanks[]` | array | 빈칸 목록 **(최대 2개)** |
| `blanks[].keyword` | string | 정답 단어 |
| `blanks[].position` | int | 왼쪽부터 몇 번째 빈칸인지 (0-indexed) |
| `blanks[].answer_length` | int | 정답 글자 수 (힌트 표시용) |

> 프론트엔드는 `blanks` 배열에서 **난이도에 따라 앞 N개만 활성화**하고 나머지는 채워진 상태로 보여준다.

**`fall_events[]`** — 낙하 이벤트 타임라인 (`target_time` 오름차순 정렬)

| 필드 | 타입 | 설명 |
|---|---|---|
| `keyword` | string | 낙하하는 키워드 텍스트 |
| `target_time` | float | 키워드 발화 시점(초) = 낙하 목표 도달 시점 |
| `fall_window` | float | 자막 시작 ~ 키워드 발화까지의 구간(초), 최솟값 0.5 |
| `segment_id` | int | 연결된 자막 세그먼트 ID |

> `fall_start_time`과 `fall_duration`은 **프론트엔드가 직접 계산**한다.
>
> `fall_window`를 기준으로 계산해야 **자막이 화면에 뜨기 전에 키워드가 낙하하는 버그**가 없다.
> ```js
> // 프론트엔드 계산 공식 (fall_window 기반)
> fall_duration   = event.fall_window / fallSpeed   // 자막 구간 안에서 낙하 완료
> fall_start_time = event.target_time - fall_duration
> ```

**`quizzes[]`** — 구간별 4지선다 퀴즈 목록

| 필드 | 타입 | 설명 |
|---|---|---|
| `quiz_id` | int | 퀴즈 번호 (0부터 순서대로) |
| `trigger_time` | float | 퀴즈 팝업 시점(초) = 해당 구간 마지막 세그먼트 종료 시점 |
| `segment_range` | [int, int] | 퀴즈가 커버하는 세그먼트 범위 `[시작 id, 끝 id]` |
| `question` | string | 질문 내용 |
| `options` | string[] | 선택지 4개 (0~3번 인덱스) |
| `answer_index` | int | 정답 선택지 인덱스 (0~3) |
| `correct_feedback` | string | 정답 선택 시 피드백/해설 |
| `incorrect_feedback` | string | 오답 선택 시 피드백/해설 |

> 프론트엔드는 영상 재생 중 `trigger_time`이 되면 퀴즈 팝업을 띄우고,
> 학습자가 답을 선택하면 `answer_index`와 비교해서 정오 판정.
> 결과(정답 여부, 선택지)는 백엔드가 저장.

**`config`**

| 필드 | 타입 | 설명 |
|---|---|---|
| `max_blanks_per_sentence` | int | AI가 생성한 최대 빈칸 수 (항상 `2`) |
| `total_blanks` | int | 전체 빈칸 개수 (max 기준) |
| `total_segments` | int | 전체 자막 세그먼트 수 |

**`stats`** — 디버그·로깅용

| 필드 | 타입 | 설명 |
|---|---|---|
| `transcript_source` | string | `"whisper"` / `"youtube_manual"` |
| `total_words` | int | 인식된 전체 단어 수 |
| `language` | string | 처리 언어 |
| `gpt_refined` | boolean | GPT 교정 실행 여부 |
| `total_quizzes` | int | 생성된 퀴즈 총 개수 |

### 오류 응답

| Status | 발생 조건 | 응답 예시 |
|---|---|---|
| `400 Bad Request` | 지원하지 않는 파일 형식 | `{"detail": "지원하지 않는 파일 형식: '.txt'"}` |
| `422 Unprocessable` | 세그먼트 추출 실패 (무음 파일 등) | `{"detail": "세그먼트가 없어 게임 데이터를 만들 수 없음"}` |
| `500 Internal Error` | OpenAI API / ffmpeg 오류 | `{"detail": "Whisper API 호출 실패: ..."}` |

---

## 기타

### 비고

- 백엔드는 AI로부터 받은 `game_data` JSON을 DB에 저장하고 프론트엔드에 전달
- `stats` 필드는 DB 저장 전 제거해도 무방 (순수 로깅용)
- `stats.transcript_source`는 로깅·분석용으로 DB에 저장해두길 권장

---
---

# 2. YouTube URL 처리

## 기능 설명

백엔드가 YouTube URL을 전달하면, AI가 자막 유무를 확인한다.
수동 자막이 있으면 해당 자막을 사용하고, 자동 자막만 있거나 자막이 없으면 Whisper STT를 사용해 game_data를 생성해 반환한다.

## AI 구현 주의사항

- 자막 존재 여부는 AI가 내부적으로 판단하므로 백엔드 별도 확인 불필요
- 수동 자막 있으면 Whisper 미사용 → 처리 속도 빠름 (10~30초)
- 자동 자막만 있으면 품질 이슈로 자동 자막을 쓰지 않고 Whisper STT로 전환
- 자막 없으면 오디오 추출 → Whisper → GPT 교정 순으로 진행 (1~5분 소요)

---

## Request (요청)

### Headers

```json
{
  "Content-Type": "application/json"
}
```

### Request Body (JSON)

```json
{
  "url":        "string",
  "language":   "string",
  "stt_prompt": "string | null",
  "refine":     "boolean"
}
```

### 유효성 검사 규칙

```json
{
  "url":        "필수, YouTube URL만 허용 (youtu.be 단축 URL 포함)",
  "language":   "선택, 기본값 'ko'",
  "stt_prompt": "선택, 전문 용어 힌트 쉼표 구분 문자열 (예: 'ADHD,도파민,전두엽')",
  "refine":     "선택, 기본값 true"
}
```

### Request Body 예시

```json
{
  "url": "https://www.youtube.com/watch?v=SBgQos4iAdw",
  "language": "ko",
  "stt_prompt": "구석기,신석기,삼국시대",
  "refine": true
}
```

---

## Response (응답)

### 성공 응답

- **Status**: `200 OK`
- **Response Body**: [1. 파일 업로드 처리 Response Body와 동일한 구조](#response-body-json)

### Response Body 예시

```json
{
  "ai_summary": "이 강의는 한국사 선사~삼국시대 흐름을 정리한다.\n\n## 구석기 시대\n이동 생활, 사냥·채집, 뗀석기 사용 등 핵심 특징.\n\n## 신석기 시대\n정착·농경 시작, 빗살무늬토기, 간석기 사용.",
  "subtitles": [
    {
      "segment_id": 0,
      "start": 10.0,
      "end": 14.5,
      "original_text": "구석기 시대는 약 70만 년 전부터 시작됩니다",
      "blank_text": "______ 시대는 약 70만 년 전부터 시작됩니다",
      "blanks": [
        { "keyword": "구석기", "position": 0, "answer_length": 3 }
      ]
    }
  ],
  "fall_events": [
    {
      "keyword":     "구석기",
      "target_time": 10.0,
      "fall_window": 0.5,
      "segment_id":  0
    }
  ],
  "config": {
    "max_blanks_per_sentence": 2,
    "total_blanks":            38,
    "total_segments":          20
  },
  "stats": {
    "transcript_source": "whisper",
    "total_words":       280,
    "language":          "ko",
    "gpt_refined":       false
  }
}
```

### 오류 응답

| Status | 발생 조건 | 응답 예시 |
|---|---|---|
| `400 Bad Request` | YouTube가 아닌 URL | `{"detail": "YouTube URL만 지원합니다"}` |
| `422 Unprocessable` | 세그먼트 추출 실패 | `{"detail": "세그먼트가 없어 게임 데이터를 만들 수 없음"}` |
| `500 Internal Error` | yt-dlp / OpenAI API 오류 | `{"detail": "yt-dlp 오디오 추출 실패: ..."}` |

---

## 기타

### 자막 처리 경로

| 상황 | 처리 경로 | `transcript_source` | 처리 시간 |
|---|---|---|---|
| 수동 자막 있음 | VTT 다운로드 → 파싱 | `"youtube_manual"` | ~10초 |
| 자동 자막만 있음 | 오디오 추출 → Whisper | `"whisper"` | 1~5분 |
| 자막 없음 | 오디오 추출 → Whisper | `"whisper"` | 1~5분 |

---
---

# 3. 서버 상태 확인

## 기능 설명

AI 서버가 정상 기동 중인지, OpenAI API 키가 설정되어 있는지 확인한다.
백엔드 서버 시작 시 헬스체크 용도로 사용 권장.

---

## Request (요청)

요청 바디 없음.

---

## Response (응답)

### 성공 응답

- **Status**: `200 OK`

### Response Body 예시

```json
{
  "status": "ok",
  "api_key_configured": true
}
```

### 오류 응답

AI 서버 자체가 다운된 경우 `Connection refused` (HTTP 응답 없음).

---

## 기타

### 비고

- `api_key_configured: false` 이면 이후 처리 API가 모두 실패하므로 백엔드에서 얼리 리턴 처리 권장

---
---

# 부록 A. 프론트엔드 난이도 계산 공식

AI가 제공하는 `target_time`을 기준으로 프론트엔드가 직접 계산.

```js
// 난이도 설정 예시
const DIFFICULTY = {
  easy:   { activeBlanks: 1, fallSpeed: 0.7, leadTime: 5.0 },
  normal: { activeBlanks: 2, fallSpeed: 1.0, leadTime: 3.0 },
  hard:   { activeBlanks: 4, fallSpeed: 2.0, leadTime: 2.0 },
}

// 빈칸 활성화 — blanks 배열에서 앞 N개만 사용
const activeBlanks = subtitle.blanks.slice(0, difficulty.activeBlanks)

// 낙하 시점 계산 — fall_window 기준으로 역산 (자막 뜨기 전 낙하 버그 방지)
// fall_window = AI가 계산한 "자막 시작 ~ 키워드 발화" 구간(초)
const fallDuration   = event.fall_window / difficulty.fallSpeed
const fallStartTime  = event.target_time - fallDuration
```

---

# 부록 B. 환경 변수

AI 서버가 필요로 하는 환경 변수. **백엔드 `.env`와 별도 파일로 관리.**

| 변수명 | 필수 | 설명 |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | OpenAI API 키 (`sk-...`) |

```bash
# backend/ai/.env
OPENAI_API_KEY=sk-...
```

---

# 부록 C. AI 파일 구조

```
backend/ai/
├── api.py                  ← FastAPI 서버 (엔드포인트 진입점)
├── pipeline.py             ← 파이프라인 오케스트레이터
├── combined_processor.py   ← 챕터 단위 GPT 통합 처리 (교정 + 키워드 + 퀴즈)
├── stt.py                  ← Whisper STT
├── transcript_refiner.py   ← 전체 내용 분석 + 챕터 분할 + 교정 맥락 생성
├── keyword_extractor.py    ← 키워드 타임스탬프 매핑 helper / 교정 스킵 경로용 추출
├── blank_subtitle.py       ← 빈칸 자막 + 낙하 이벤트 생성
├── quiz_generator.py       ← 챕터-세그먼트 매핑 helper / 교정 스킵 경로용 퀴즈 생성
├── youtube_subtitle.py     ← YouTube VTT 처리
├── youtube_audio.py        ← YouTube 오디오 추출
├── BACKEND_HANDOFF.md      ← 백엔드 전달용 요약 명세
├── requirements.txt
└── .env
```
