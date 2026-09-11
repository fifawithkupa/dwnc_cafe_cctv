# 박스 → Supabase 전송 + 손님 지도 데이터 — 설계

> 2026-09-08. 결정 과정은 `문서/다음할일.md` 2번 "판정 결과를 보여줄 서버·앱"에서 왔다.
> 손님 앱은 다른 사람이 만든다. 이 문서의 범위는 **박스가 Supabase 에 쓰는 것과,
> 앱이 읽을 표의 모양, 지도 데이터의 첫 초안**까지다. 앱 화면은 범위 밖이다.

## 1. 목표

손님 앱이 두 화면을 그릴 수 있어야 한다.

1. **카페 목록:** 카페마다 "사용중 3 / 전체 8" (테이블 기준).
2. **카페 지도:** 카페를 누르면 자리마다 네모 칸이 있고, 칸 색으로 찼는지·비었는지·모름을 보여준다.

박스는 판정 하나가 끝날 때마다(15초) 지금 값을 Supabase 에 쓴다. 지난 값은 안 올린다
(사용자 결정: "지금 값만"). 판정 기록 파일은 박스 안에 그대로 남는다.

## 2. 전체 그림

```
박스 (카페 안)                          Supabase                       손님 앱
engine.seatnow 판정 루프 ──15초마다──▶ cafe_live (카페당 1줄, 덮어씀) ──실시간──▶ 목록 숫자 · 지도 색
                        ──시작 때·파일 바뀔 때──▶ cafe_maps (카페당 1줄, 지도)  ──▶ 지도 칸 위치
사람 (대시보드에서 1회)  ───────────────▶ cafes (카페 이름 등 고정 정보)   ──▶ 목록
```

- 박스는 Supabase 에 **직접** 쓴다. 중간 서버 없음 (사용자 결정 1번).
- 박스마다 Supabase 로그인 계정이 하나 있고, 그 계정은 **자기 카페 줄만** 고칠 수 있다
  (행 단위 권한, RLS). 카페에 놓인 박스가 뜯겨도 다른 카페 데이터는 못 건드린다.
- 앱은 로그인 없이(anon 키) 세 표를 읽기만 한다. 셋 다 공개 정보다.

## 3. Supabase 표

SQL 한 파일(`deploy/supabase/schema.sql`)로 만든다. 대시보드 SQL 편집기에 붙여넣고 실행하면 끝.

### `cafes` — 카페 고정 정보 (사람이 대시보드에서 넣는다)

| 열 | 타입 | 뜻 |
|---|---|---|
| `id` | text, PK | 사람이 읽을 수 있는 짧은 이름. 예 `dwnc`. 박스 설정과 같아야 한다 |
| `name` | text | 손님에게 보이는 카페 이름 |
| `address` | text, null | 주소 |
| `created_at` | timestamptz | 기본 now() |

### `boxes` — 어떤 박스가 어떤 카페인가 (사람이 1회)

| 열 | 타입 | 뜻 |
|---|---|---|
| `auth_user_id` | uuid, PK | Supabase 인증 사용자의 id (박스 계정) |
| `cafe_id` | text, FK→cafes | 이 박스가 쓸 수 있는 카페 |
| `label` | text | 예 "uhho (i3-6100T)" |

RLS 정책의 근거가 되는 표다. 앱은 안 읽는다 (anon 읽기 없음).

### `cafe_live` — 카페당 한 줄, 박스가 15초마다 덮어쓴다

| 열 | 타입 | 뜻 |
|---|---|---|
| `cafe_id` | text, PK, FK→cafes | |
| `status` | text | `live` (정상) 또는 `gap` (카메라 끊김 — 모든 자리 unknown) |
| `total_tables` | int | 전체 (테이블 기준. 바 칸 하나 = 1) |
| `occupied_tables` | int | 사용중 |
| `free_tables` | int | **확실히 빈 것만** |
| `unknown_tables` | int | 모름 |
| `seats` | jsonb | 자리 목록 (아래) |
| `tick_at` | timestamptz | 박스가 판정한 시각 (박스 시계) |
| `updated_at` | timestamptz | **서버가 트리거로 찍는다.** 앱은 이걸 본다 |
| `box_version` | text | 박스 코드의 git 짧은 해시. 문제 추적용 |
| `schema_version` | int | 이 줄의 모양 버전. 지금 1 |

`seats` 는 자리마다 하나:

```json
[
  {"seat_id": "T1",     "kind": "table",    "zone": null,   "state": "occupied", "reason_code": null},
  {"seat_id": "T2",     "kind": "table",    "zone": null,   "state": "empty",    "reason_code": null},
  {"seat_id": "BAR7-1", "kind": "bar_seat", "zone": "BAR7", "state": "unknown",  "reason_code": "compact_occluded_pose"}
]
```

- `state` 는 `occupied` / `empty` / `unknown` 셋 중 하나. 애매한 것을 `empty` 로 반올림하지 않는다 (`CLAUDE.md`).
- `reason_code` 는 `unknown` 일 때만 값이 있다. 어휘는 `engine/seatnow_report.py` 의 것 그대로.
- 바 자리는 **칸마다 한 줄**이다. 지금 앱용 요약(`seat_report`)은 바를 구역으로 묶어 개수만
  내는데, 지도는 칸마다 색을 칠해야 하므로 여기서는 판정 단위(`record["tables"]`)를 그대로 편다.
  합계 네 숫자는 `seat_report.totals` 와 같다 (바 칸도 각각 1로 센다).
- 화각 밖이라 판정 안 하는 자리(`ignore`)는 목록에 **없다.** 지도에는 칸이 있는데 상태가
  없으면 앱은 회색 "확인 불가"로 그린다.

**`status = gap` 일 때:** `seats` 의 모든 `state` 가 `unknown`, `reason_code` 는 `no_fresh_frames`,
`occupied = free = 0`, `unknown = total`. 박스가 카메라를 놓친 동안 "옛 값 유지"가 아니라
"모름"이 되게 하기 위해서다.

### `cafe_maps` — 카페당 한 줄, 지도

| 열 | 타입 | 뜻 |
|---|---|---|
| `cafe_id` | text, PK, FK→cafes | |
| `floorplan` | jsonb | `layouts/<카페>.floorplan.json` 파일 내용 그대로 |
| `version` | int | 올릴 때마다 +1. 앱은 값이 바뀌면 다시 받는다 |
| `updated_at` | timestamptz | 서버 트리거 |

`floorplan` 은 이미 있는 평면도 형식(`install/floorplan.py`, schema 2)이다. 앱이 쓰는 부분만 적으면:

```json
{
  "schema_version": 2,
  "extent": {"width": 1000, "height": 562},
  "seats": [
    {"seat_id": "T1", "kind": "table", "x": 358.9, "y": 482.1, "w": 150, "h": 105,
     "angle": 0, "shape": "rect", "needs_review": true, "image_anchor": [1145.4, 556.7]}
  ],
  "chairs": [
    {"seat_id": "T1", "x": 300.0, "y": 470.0, "w": 46, "h": 46, "angle": 0,
     "capacity": 1, "hidden": false, "needs_review": true, "image_anchor": [1131.5, 564.9]}
  ],
  "landmarks": [
    {"kind": "door", "label": "입구", "x": 20, "y": 500, "w": 80, "h": 20, "angle": 0}
  ],
  "counters": [],
  "walls": [[0, 0], [1000, 0], [1000, 562], [0, 562]]
}
```

- `seats[].x, y` 는 칸의 **왼쪽 위**, `w, h` 는 크기, `angle` 은 시계 방향 회전(도).
- `chairs[].seat_id` 는 그 의자가 속한 자리. `hidden: true` 면 안 그린다.
- `landmarks[].kind` 는 `wall` `door` `counter` `window` `sofa` 중 하나. `walls` 는 벽 꼭짓점 목록.
- `image_anchor` 와 `needs_review` 는 편집 도구용이다. 앱은 무시한다.

좌표는 미터가 아니라 그림 단위다. 앱은 `extent` 를 화면에 맞춰 비율대로 줄이고, `seats` 를
네모로 그린 뒤 `cafe_live.seats` 와 `seat_id` 로 짝지어 색을 칠한다. `chairs` 와 `landmarks` 는
있으면 그리고 없으면 건너뛴다.

### 권한 (RLS)

| 표 | anon (앱) | 박스 계정 |
|---|---|---|
| `cafes` | select | select |
| `boxes` | 없음 | 자기 줄 select |
| `cafe_live` | select | 자기 카페 줄 insert/update (`cafe_id in (select cafe_id from boxes where auth_user_id = auth.uid())`) |
| `cafe_maps` | select | 자기 카페 줄 insert/update (같은 조건) |

`cafe_live` 와 `cafe_maps` 는 실시간 발행(`supabase_realtime` publication)에 넣는다. 앱은
`cafe_live` 의 자기 카페 줄을 구독하면 15초마다 새 값을 받는다.

## 4. 박스 쪽 코드

### 새 모듈 `edge/publish.py` — `SupabasePublisher`

한 가지 일만 한다: **"지금 값"을 Supabase 에 쓴다. 판정 루프를 절대 기다리게 하지 않는다.**

```
SupabasePublisher(url, anon_key, email, password, cafe_id, box_version)
  .start()                     # 백그라운드 스레드 시작, 로그인
  .publish_live(payload: dict) # 최신 값 하나만 보관(queue 크기 1, 새 것이 옛 것을 밀어냄)
  .publish_map(floorplan: dict)
  .stats() -> {"sent", "failed", "last_error", "last_ok_at", "logged_in"}
  .stop()
```

- **HTTP:** `requests` (엣지 요구사항에 이미 있음). PostgREST upsert:
  `POST {url}/rest/v1/cafe_live` + `Prefer: resolution=merge-duplicates`. 타임아웃 5초.
- **로그인:** `POST {url}/auth/v1/token?grant_type=password` → access token. 401 이 오면 한 번
  다시 로그인하고 재시도. refresh token 은 쓰지 않고 만료 전(기본 1시간)에 다시 password 로그인.
  단순함이 우선이다.
- **실패:** 버리고 다음 틱에 최신 값을 보낸다. 실패가 이어지면 백오프(5초→10→20→최대 60초)로
  시도 간격만 늘린다. 앱 쪽은 45초 규칙으로 "확인 중"이 된다. 박스 로그에는 첫 실패와
  10번마다 한 줄, 복구 때 한 줄.
- **인터넷이 없어도** 판정은 그대로 돈다. 스레드가 죽어도 판정은 돈다 (예외를 삼키고 카운트).

### 판정 루프 연결 (`engine/seatnow.py` `process_live`)

세 군데만 건드린다.

1. 시작할 때: 환경변수가 다 있으면 publisher 를 만들고 `start()`. 없으면 `None` 이고 아무 일도
   안 한다 (파일 입력·벤치·테스트는 지금과 같다). 시작 화면에 `Supabase 전송: 켜짐 (dwnc)` 또는
   `꺼짐 (SEATNOW_SUPABASE_URL 없음)` 한 줄.
2. 판정 한 줄을 파일에 쓴 직후: `publisher.publish_live(live_payload(record, cafe_id, box_version))`.
3. `gap` 줄을 쓴 직후: `publisher.publish_live(gap_payload(...))`.
4. 매 틱, 평면도 파일의 수정 시각이 바뀌었으면 `publish_map(...)`. 시작 때도 한 번.
   평면도 경로는 레이아웃 경로에서 `.json` → `.floorplan.json`. 파일이 없으면 지도는 안 올리고
   시작 화면에 한 줄 알린다.
5. 종료 요약(`last_run_summary.json`)에 `publisher.stats()` 를 넣는다. `edge.live_report` 가
   "전송 성공/실패 횟수"를 같이 보여준다.

payload 를 만드는 함수(`live_payload`, `gap_payload`)는 순수 함수로 `edge/publish.py` 에 둔다.
입력은 JSONL 한 줄(dict)이라 기존 기록 파일로 그대로 테스트된다.

### 설정 (`deploy/seatnow.env`)

```
SEATNOW_CAFE_ID=dwnc
SEATNOW_SUPABASE_URL=https://xxxx.supabase.co
SEATNOW_SUPABASE_ANON_KEY=eyJ...
SEATNOW_SUPABASE_EMAIL=box-dwnc@seatnow.local
SEATNOW_SUPABASE_PASSWORD=...
```

`seatnow.env` 는 이미 git 에서 제외돼 있다(카메라 비밀번호가 있어서). 서비스 파일은 안 바꾼다 —
`EnvironmentFile` 로 이미 다 들어간다. `seatnow.env.example` 에 다섯 줄을 설명과 함께 추가.

`box_version` 은 시작 때 `git rev-parse --short HEAD` (실패하면 `unknown`).

## 5. 지도 첫 초안 — 바닥 네 점 없이

지금 `install/floorplan.build_draft` 는 바닥 기준점(네 점)이 없으면 거부한다. 사용자 결정은
"카메라 화면 위치 그대로, 나중에 편집기로 고친다"이므로 **네 점이 없을 때의 길**을 추가한다:

- 각 판정 단위의 화면 상자 중심을 그대로 캔버스에 놓는다 (화면 1920×1080 → 캔버스 긴 변 1000,
  여백 8%). 크기는 기본값(`DEFAULT_SIZES`). 의자도 화면 위치 그대로.
- 전부 `needs_review: true`. 편집기가 이미 이 표시를 보여준다.
- 겹침은 기존 `separate_overlaps` 로 푼다.
- 네 점이 **있으면** 지금처럼 투영한다. 두 길의 출력 형식은 같다.

새 명령 `python -m install.floorplan --layout layouts/<카페>.json` 은 평면도 파일이 없을 때만
초안을 만들어 저장한다 (있으면 건드리지 않는다 — 사람이 고친 것을 덮지 않기 위해).
`문서/카페설치당일.md` 6-3 "저장했어. 박스에 올려줘" 단계에서 Claude 가 이 명령을 돌리고 레이아웃과
평면도를 같이 박스에 복사한다. 그러면 박스가 다음 틱에 `cafe_maps` 를 올린다.

고치고 싶으면 노트북에서 기존 편집기(`python -m install.floorplan_editor --layout ...`)로 옮기고
저장 → 박스에 복사 → 자동으로 올라간다. 편집기 자체는 손대지 않는다.

## 6. 앱 개발자에게 주는 문서 `docs/앱연동.md`

이 설계의 3장(표), `seats` 와 `floorplan` 예시 JSON, 그리고 규칙:

1. **45초 규칙.** `cafe_live.updated_at` 이 지금보다 45초 이상 오래됐으면 숫자와 지도를
   "확인 중"으로 바꾼다. `status = gap` 도 같다. 이걸 안 지키면 박스가 꺼졌을 때 어제 값이
   영원히 보인다.
2. 빈 자리는 `free_tables` 만 쓴다. `total - occupied` 로 계산하지 않는다 (모름이 빈 자리가 된다).
3. 지도 칸은 `seat_id` 로 짝짓는다. `cafe_live.seats` 에 없는 칸은 회색.
4. 실시간 구독 예시 (supabase-js `channel().on('postgres_changes', ...)`) 와 처음 한 번 읽는 예시.
5. 앱이 쓰는 키는 anon 키. 쓰기는 안 된다.

## 7. 실패와 경계

| 상황 | 박스 | 앱이 보는 것 |
|---|---|---|
| 카페 인터넷 끊김 | 판정 계속, 전송 실패 카운트, 복구 시 최신 값부터 | 45초 뒤 "확인 중" |
| 카메라 끊김 | `gap` 줄 → `status=gap` 전송 | 즉시 모두 "모름" |
| 박스 꺼짐 | — | 45초 뒤 "확인 중" |
| Supabase 비밀번호 틀림 | 시작 화면에 크게 경고, 판정은 계속, 60초마다 재로그인 시도 | "확인 중" |
| 평면도 파일 없음 | 시작 화면 한 줄, `cafe_live` 는 정상 | 숫자는 보이고 지도는 "준비 중" |
| 레이아웃 바뀜 (자리 추가·삭제) | 새 `seats` 목록이 그대로 올라감 | 지도 칸과 안 맞는 자리는 회색 → 평면도도 다시 올린다 |

## 8. 테스트

- `tests/test_publish.py`
  - `live_payload`: 실제 JSONL 한 줄(fixtures 에 있는 것)로 → 자리 수·합계·바 칸 펼침·`ignore` 제외·`reason_code` 가 unknown 일 때만.
  - `gap_payload`: 전부 unknown, 합계 규칙.
  - `SupabasePublisher`: 가짜 HTTP 서버(`http.server` 스레드)로 — 로그인 → upsert 헤더·본문, 401 → 재로그인 1회, 타임아웃/연결 실패 → 실패 카운트만 늘고 예외 없음, queue 크기 1(3개 넣으면 마지막 것만 감), `stop()` 이 3초 안에 끝남.
- `tests/test_floorplan.py` 에 추가: 바닥 네 점 없는 레이아웃 → 초안이 나오고 전부 `needs_review`, 좌표가 캔버스 안, 네 점 있는 경우는 기존 결과 그대로.
- `tests/test_live_logdir.py` 근처: 환경변수 없을 때 publisher 가 `None` 이고 루프 출력이 지금과 같음.
- 손으로 1회: 집 박스에 실제 Supabase 값을 넣고 10분 돌려 대시보드에서 `cafe_live` 가 15초마다 바뀌는지, 인젝터를 뽑아 `gap` 이 오는지, 랜선을 뽑았다 꽂아 복구되는지.

## 9. 사람이 할 일 (Supabase 대시보드, 1회)

1. SQL 편집기에서 `deploy/supabase/schema.sql` 실행.
2. Authentication → 사용자 추가: 이메일 `box-dwnc@seatnow.local`, 비밀번호 정하기 (자동 확인 켬).
3. Table editor 에서 `cafes` 에 한 줄 (`id=dwnc`, `name=...`), `boxes` 에 한 줄 (방금 만든 사용자의 uuid, `dwnc`).
4. Claude 에게 프로젝트 URL · anon 키 · 박스 이메일 · 비밀번호를 준다 → 박스 `seatnow.env` 에 넣고 재시작.

## 10. 범위 밖

- 앱 화면 (다른 사람).
- 지난 값 저장·통계 (사용자 결정: 지금 값만).
- 중간 서버, Edge Function.
- 원격에서 박스에 붙기 (별도 항목).
- 평면도 편집기 개선.
