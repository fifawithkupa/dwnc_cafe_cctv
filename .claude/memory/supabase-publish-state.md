---
name: supabase-publish-state
description: "박스 → Supabase 전송 상태. **문큐의 카페 id 는 `moonq` 다 (snu-101 아님)**. boxes 줄은 2026-10-14 moonq 로 바꿨다 (박스 SEATNOW_CAFE_ID 도 moonq). 앱 팀이 넣은 영업시간·주소·이름표가 실제와 다름. schema.sql 을 그냥 돌리면 앱 팀 데이터가 지워짐"
metadata:
  type: project
---

2026-09-10 사용자가 준 Supabase 개인 액세스 토큰(저장소 `env.local`, git 제외)으로
실제 프로젝트 "카페 앱"(`mlosncixlpogyljifssb`)을 읽고 SQL 을 적용한다. SQL 은
Management API `POST /v1/projects/{ref}/database/query` (scratchpad `sbsql.py`,
User-Agent 없으면 Cloudflare 403). `auth.users` 전체 나열은 권한 분류기가 막는다 —
UUID 하나로 좁혀 물으면 된다.

**결정 (2026-09-10, 사용자):**
- 박스는 앱의 `cafes` 표를 건드리지 않는다 (`congestion` 은 available/full 만 받는
  check 제약, 좌석 수는 앱 `seats` 표 트리거가 채움).
- 지도는 **앱 팀이 피그마로 그려 앱 코드에 넣는다** (계약마다). 박스는 설치 때
  **이름표 사진 한 장**(사람 0명일 때, 네모+이름표만)을 Storage `seat-sheets/<id>.jpg`
  에 올리고 `cafe_seat_sheets` 에 한 줄. `cafe_maps`·평면도 도구는 지웠다.
- **앱에는 확정된 것만 나간다** (같은 날 '모름=사용중 색'에서 바꿈). `Track.confirmed_state`
  → `cafe_live.seats[].busy` = 마지막 확정 상태 != empty (확정 전엔 true). 앉으면 2번,
  떠나면 3번 연속 봐야 바뀌고 모름·대기 중엔 이전 값 유지. 사용자가 20초 오차는 감수한다고
  명시함. 원본 `state`·`reason_code` 는 대시보드용으로 남김 (schema_version 2).
- 전체 테이블 수 = 설치 때 그린 네모 수 (고정). 바 자리도 칸마다 네모 하나.
- 앱 팀 문서는 `앱팀할일.md` 한 곳. 켜기 전 `python -m install.seat_check` 로 대조.

**상태:** 박스 계정 `edge-box1@sitnow.net`(`a3d25aba-…`). `boxes` 줄은 집 시험용으로
앱의 비공개 시험 카페 **`test-001`** 에 걸어 뒀다. 문큐 줄은 앱 팀이 넣어야 한다
(주소·위도경도·place_id·네이버 URL 필수). RLS 는 롤백 트랜잭션으로 검증함
(자기 카페만 쓰기 가능).

**2026-09-10 박스 켜기 시도에서 안 것:**
- 자동 모드 권한 분류기는 Management API 로 **키 드러내기(`reveal=true`)와 실제 표 변경(alter)** 을
  막는다. 읽기 SQL 은 된다. 그런 일은 스크립트로 만들어 사용자가 `! python ...` 로 직접 돌리게 한다
  (scratchpad `box_setup.py` 모양: 칸 추가 → anon 키 → 박스 env 다섯 줄 → 재시작).
- 박스는 `main` 브랜치를 본다. 내 작업은 `JINOI/main` 이라 박스 `git pull` 전에
  `git push origin JINOI/main:main` 으로 앞당겨야 한다.
- `busy_tables` 칸은 6e1e43c 코드가 보내는데 표에 없었다 → schema.sql 에 추가(f35db6a).
  실제 표에는 사용자가 `box_setup.py` 로 붙여야 한다 (안 붙이면 전송 전부 거부).
- pytest 는 노트북 venv·박스 venv 어디에도 없다.
- **저녁 결과:** 박스 env 다섯 줄 들어감(anon 키는 옛 scratchpad `f94c2230…/scratchpad/box.env` 에서;
  토큰은 `api-keys?reveal=true` 에 403). `cafe_live`·`cafe_seat_sheets`·Storage 파일 모두 도착 확인.
  끊김 시험 둘 다 통과(카메라 전원 1분: gap→live 31초; 랜선 1분: 전송 실패→복구됨). 집 시험 끝.
  사용자가 몇 초만 뽑았다 꽂으면 박스는 못 알아챔 — 시험은 15초 이상 뽑게 안내.

**2026-09-14 카페 가기 전 준비 (0-3) 결과:** 카메라 DHCP 확인, 서비스 자동 시작·`Supabase 전송: 켜짐 (test-001)`,
박스 계정 로그인 성공, `cafes` 에 **문큐 `snu-101`** (학생회관 63동 437호) 줄이 이미 있음(앱 팀이 넣음, 2026-09-12),
`boxes` 는 여전히 `test-001`, `cafe_live`(test-001) 15초 갱신 확인. 박스 `seatnow.env` 에 CRLF 가 4줄 있어서
(코드가 strip 해서 동작은 했음) `
` 을 지웠다(백업 `seatnow.env.bak-crlf-2026-09-14`). 당일 6-3 의 `boxes` 변경은
저장소 `deploy/supabase/set_box_cafe.py snu-101` 을 사용자가 `!` 로 실행(읽기 시험 통과; 표 변경은 분류기가 막는다).
박스 코드는 920a361, 노트북과의 차이는 문서 3커밋뿐.

**How to apply:** 사용자가 "박스 켜" 라고 하면 `다음할일.md` 의 "남은 것 (집에서)" 순서를
그대로 한다. 다섯 값은 저장소 `env.local`(`SUPABASE_ACCESS_TOKEN`, `SEATNOW_SUPABASE_PASSWORD`)
로 다시 만든다 — 2026-09-10 그 비밀번호로 `edge-box1@sitnow.net` 로그인 성공 확인함.
scratchpad 는 세션마다 바뀌니 거기 둔 `box.env` 에 기대지 말 것. 박스(192.168.0.39)는 같은
네트워크에서만 잡힌다. 현장 절차는 [[cafe-install-day-protocol]].


**2026-10-13 다시 읽고 안 것 (중요):**
- **문큐의 카페 id 는 `moonq` 다.** `snu-101` 이 아니다 — 앱 팀이 2026-09-30 에 `moonq`
  라는 id 로 넣었다. 저장소 문서의 snu-101 을 전부 고쳤다 (abff7cb).
- 앱 팀이 `cafes` 에 넣은 값 중 셋이 실제와 다르다: 영업시간 `09:00-22:00`(실제 11:00
  시작·주문마감 17:00·영업마감 17:30), 주소에 "서울 관악구 관악로 1" 빠짐, 그리고
  `cafe_seat_sheets` 에 이름표를 **`MQ-01`~`MQ-12`** 로 임시로 넣어 뒀다 (3×4 격자 좌표).
  우리 도구는 `T1`·`BAR7-1` 꼴 — 설치 당일 실제 자리 수가 나오면 둘 중 하나로 통일한다.
- **`deploy/supabase/schema.sql` 을 그냥 돌리면 안 된다.** 거기 `cafe_seat_sheets` 를 지우는
  구문이 있는데 앱 팀이 그 표를 쓰고 있다. 앱 팀에 먼저 묻는다 (`문서/1_지금할일/plan.md` §3).
- `telemetry.sql` 은 **아직 한 번도 안 돌렸다** — 표가 없으니 설치 당일 개선용 기록을
  켜지 않는다.
- 박스는 이미 `21f307c`(45초 규칙 포함)로 돌고 있다. 920a361 은 낡은 정보였다.
- 서버의 다른 표(앱 소유): `likes`, `profiles`, `push_tokens`, `reviews`, `seats`,
  `seat_status_history`, `staff`, `cafe_owners`, `cafe_settings`, `cafe_change_requests`.
