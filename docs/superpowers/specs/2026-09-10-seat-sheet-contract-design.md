# 이름표 사진 계약 — 피그마 지도 + `busy` 한 칸 (설계)

2026-09-10. 사용자와 대화로 정한 것. 구현은 같은 날.

## 배경

앱 팀은 카페 배치도를 **피그마로 그려서 앱 코드에 넣는다** (카페 계약할 때마다 하나).
그래서 박스가 만들던 자동 지도(`cafe_maps`, `install/floorplan*`)는 필요 없어졌다.
대신 앱 팀이 피그마 네모마다 이름표(`seat_id`)를 붙일 수 있게 **근거 사진 한 장**이 필요하다.

## 결정 (사용자)

| 물음 | 답 |
|---|---|
| 전체 테이블 수 | 설치 때 그린 네모 수 = 고정. 카메라가 못 보는 테이블은 없게 설치한다 |
| 모름(unknown) 표시 | **사용중으로 칠한다.** 대시보드는 나중에 |
| 바 자리 | 칸마다 네모 하나 (UI 표현은 앱 팀 몫) |
| 매칭이 사는 곳 | 앱 코드. 사진은 근거 자료 |
| 사진 보관 | Supabase Storage, 카페당 한 장, 덮어쓰기 |

## 박스가 보내는 것

- `cafe_live` (기존) + `busy_tables`(= occupied + unknown), `seats[].busy`(= state != empty).
  `schema_version = 2`. `state`·`reason_code` 는 원본으로 남긴다 (대시보드용).
- `cafe_seat_sheets` (새, 카페당 한 줄): `seat_ids`, `image_path`, `taken_at`, `box_version`.
- Storage `seat-sheets/<cafe_id>.jpg` (비공개 버킷).

## 이름표 사진을 찍는 규칙

- 좌석 파일(`layouts/<카페>.json`)의 내용 해시가 마커 파일(`<카페>.seat_sheet.sha`)과 다르면
  "찍어야 함". 마커는 올리기에 성공했을 때만 쓴다 → 재시작해도 다시 안 찍고, 실패했으면 다시 시도.
- **화면에 사람이 0명일 때만** 찍는다 (`record["poses"]` 가 비어 있을 때). 손님 얼굴이 안 들어간다.
  사람이 있으면 5분에 한 번만 "대기 중 (사람 N명)" 로그.
- 그림: 카메라 원본 위에 판정 단위(테이블·바 칸)마다 네모 + 이름표. 상태 색 없음, 한 가지 색.

## 검사

`python -m install.seat_check --layout layouts/<카페>.json --app-ids <앱이 보낸 목록.txt>`
→ 개수·이름 대조. 통과해야 켠다.

## 지우는 것

`cafe_maps` 표·트리거·발행, `publish_map`, `FloorplanWatcher`, `install/floorplan.py`,
`install/floorplan_editor.py`, `install/floor_projection.py`, `install/static`, 그 테스트,
`layouts/*.floorplan.json`. git 이력에 남는다.

## 앱 팀 문서

`앱팀할일.md` 한 곳. `docs/앱연동.md` 는 거기를 가리키기만 한다.
