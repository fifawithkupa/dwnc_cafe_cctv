# 레이아웃 존 실시간 트래킹 설계

**Goal:** 최초 1회 캘리브레이션 후, 운영 중 가구가 움직이면 존(테이블·의자 ROI)이 YOLO 탐지를 따라 이동한다. 탐지가 없는 존은 마지막 위치에 고정된다. 존 개수·ID·의자-테이블 연결은 절대 변하지 않는다.

## 결정 사항 (사용자 확정)

- 존은 매칭된 탐지 박스의 **위치+크기**를 모두 따라간다 (EMA 스무딩).
- 장면 전환 시 **마지막 위치 유지** (원위치 복귀 없음). `reset()` API는 제공하되 파이프라인은 호출하지 않는다.

## 컴포넌트: `LayoutZoneTracker` (seatnow_core.py)

순수 로직 클래스 — 모델 없이 유닛테스트 가능.

- 상태: 테이블 존 박스 리스트, 의자 존 박스 리스트 (초기값 = 스케일된 레이아웃).
- `update(table_detections, chair_detections)` — 프레임마다 호출.
- 매칭 규칙 (보수적):
  1. 탐지는 현재 존과 **IoU ≥ 0.30**인 것만 후보.
  2. 한 탐지가 두 존과 비슷하게 겹치면(1등·2등 IoU 차 < 0.10) 그 탐지는 버린다 — 애매하면 정지.
  3. IoU 내림차순 greedy 배정, 존·탐지 각각 1회만.
- 이동: 좌표 4개를 `alpha=0.35` EMA로 블렌드.
- 입력 게이트: 테이블 탐지 conf ≥ 0.25, 의자류 conf ≥ 0.30.

## 통합

- `SeatNowAnalyzer.analyze()` 레이아웃 분기에서 고정 레이아웃 박스 대신 트래커의 현재 박스 사용. 트래커는 첫 프레임에 lazy 초기화 (해상도 스케일 이후 보장).
- `AnalyzerConfig.layout_tracking: bool = True`, CLI `--no-layout-track`.
- run.config에 `layout_tracking` 기록.

## 포함 버그 수정

레이아웃 존은 `is_severely_border_cropped` 무시 규칙에서 제외 — 화면 경계에 수동으로 그린 존(T4)이 영구 ignore되던 문제.

## 검증

- 유닛: 매칭 이동 / 무탐지 정지 / 저 IoU 거부 / 애매성 가드 / greedy 1:1 / reset.
- 무회귀: 기존 72개 테스트.
- E2E: 좌석착석 영상 + seminar_room.json — T4가 ignore에서 벗어나고 존 상태 정상.
