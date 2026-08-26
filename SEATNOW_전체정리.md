# SeatNow — CCTV 좌석 점유 감지 시스템 전체 정리

> SSATIS 팀 (4인) / YOLOv8 기반 카페 좌석 점유 감지
> 이 문서는 지금까지의 논의 + 구현 내용 + 앞으로 할 일을 한 곳에 정리한 것
> 최종 갱신: **2026-08-26** — `main` = `0297d23` 기준으로 **코드 구조 · 기본 로직 전면 재정리**
> (직전 갱신 2026-07-11 이후 추가된 버스트 다수결 / 적응형 케이던스 / 레이아웃 변경 감지 반영,
>  그리고 §9에 **현재 코드에서 발견된 회귀** 기록)

---

## 1. 프로젝트 개요

카페 CCTV 영상을 분석해서 좌석(테이블) 점유 여부를 실시간으로 판단하고, 손님용 앱에 "몇 자리 남음"을 보여주는 시스템.

- **참고 논문**: Seatify (IEEE ICARC 2026, DOI: 10.1109/ICARC68737.2026.11453649) — 스리랑카 대학 졸업작품, 상업 제품 아님 → 방법론 자유롭게 구현 가능
- **개발 환경**: MacBook Pro 2018 (Intel i7, 16GB RAM, GPU 없음), Python 3.9.6, 가상환경(`venv`)
- 의존성 핀: `ultralytics==8.4.82`, `opencv-python==4.10.0.84`, `numpy<2`, `torch==2.2.2`
  (numpy 2.x ↔ PyTorch 충돌 / opencv 4.11+ ↔ numpy<2 충돌 때문에 둘 다 고정)
- **영상 I/O는 OpenCV가 아니라 ffmpeg 바이너리** — 타깃 Intel Mac의 OpenCV 휠에 비디오 백엔드가 없음

---

## 2. 코드 구조 (현재 `main` 기준)

### 2.1 파일 맵

| 파일 | 줄수 | 역할 |
|------|-----:|------|
| `seatnow_core.py` | 3,194 | **본체**. 아래 2.2의 6개 계층을 한 모듈에 담음 |
| `seatnow.py` | 596 | CLI 진입점. 인자 파싱 → 샘플링 루프 → JSONL/MP4 출력 |
| `verify_seatnow.py` | 196 | 결과 JSONL을 수동 라벨 정답(`tests/fixtures/test_video_expectations.json`)과 대조 |
| `seatnow_layout.py` | 162 | 수동 좌석 레이아웃 JSON 로드/검증/해상도 스케일 (`SeatLayout`) |
| `calibrate.py` | 304 | 클릭 캘리브레이션 도구 (테이블·의자 등록 → `layouts/*.json`) |
| `occupancy_mvp.py`, `pose_judge.py` | 107 / 124 | 초기 PoC, 참고용으로만 유지 (파이프라인에서 import 안 됨) |
| `tests/` | 2,177 | 유닛 테스트 **117개** (모델 없이 순수 함수로 검증) |
| `docs/superpowers/` | — | 설계 스펙 3건 (수동 레이아웃, 존 트래킹) |
| `layouts/` | — | 캘리브레이션 결과 샘플 2건 |

> `README.md`는 아직 "테스트 54개"라고 적혀 있음 — 실제 117개. §9 참조.

### 2.2 `seatnow_core.py`의 6개 계층

한 파일이지만 아래 순서로 층이 뚜렷하게 나뉘어 있고, **아래층은 위층을 모른다**(순수 함수 우선).

```
① 타입/설정        L29–280    PoseState, OccupancyState, Detection, PoseObservation,
                              TableObservation, FrameAnalysis, AnalyzerConfig,
                              Track, TrackerUpdate, LayoutChangeCandidate
② 기하 유틸        L281–390   box_iou, expand_box, overlap_over_smaller,
                              table_surface_box, blend_boxes, center_distance_ratio …
③ 판정 순수함수    L390–970   classify_pose(포즈), select_table_candidates(테이블 선별),
                              associate_*(증거 연결), filter_carried_objects(들고있는 짐 제외),
                              occupancy_state_from_evidence(OR 규칙),
                              cluster_people / has_seat_support(가려진 좌석 추론)
④ 프레임 분석기    L972–1562  LayoutZoneTracker(존 드리프트) + SeatNowAnalyzer.analyze()
                              ← 유일하게 YOLO 모델을 잡고 있는 곳
⑤ 시간 안정화      L1563–2482 aggregate_burst_observations(버스트 다수결),
                              AdaptiveCadenceController(주기 제어),
                              TableTracker(ID 추적 + 디바운싱 + 레이아웃 변경 감지)
⑥ I/O·렌더         L2483–3194 scene_change 감지, FFmpegSampleReader/BurstReader/VideoWriter,
                              render_frame(주석 영상), track_to_dict / frame_log_record(JSONL)
```

**설계 원칙**: ③층(판정 로직)은 모델·영상 없이 좌표만으로 테스트 가능해야 한다. 그래서 유닛 테스트 117개가 GPU도 모델 가중치도 없이 돌아간다. 판정 로직을 고칠 때는 **실패 사례의 실제 좌표**로 회귀 테스트를 추가하는 것이 이 저장소의 관례.

---

## 3. 실행 파이프라인

`python seatnow.py <영상.mp4>` 한 번 = 아래 흐름.

```
영상 파일
  │  probe_video()            해상도·fps·길이·코덱 파악
  ▼
[샘플링]  FFmpegBurstReader    기본 15초마다 "샘플 시점" 하나를 잡고,
  │                           그 시점 ±2 네이티브 프레임 = 총 5장을 뽑는다
  ▼
[분석]    SeatNowAnalyzer.analyze()  ← 5장 각각에 대해 1회씩 (§4)
  │       · 사이드 4장: update_temporal=False (시간 상태 안 건드림, 투표만)
  │       · 센터 1장  : 마지막에 실행, 포즈 히스토리·존 드리프트를 여기서만 갱신
  ▼
[다수결]  aggregate_burst_observations()   좌석별 raw_state 5표 다수결 → 센터 관측에 반영
  ▼
[추적]    TableTracker.update()            ID 유지 + 비대칭 디바운싱 + 레이아웃 변경 감지 (§5,§7)
  ▼
[출력]    frame_log_record() → JSONL 1줄   /   render_frame() → 주석 MP4 1프레임
  ▼
[주기결정] AdaptiveCadenceController        다음 샘플까지 15초? 5초? (§5.2)
```

**두 가지 실행 모드**
- **버스트 모드(기본)**: 위 그림 그대로. `--median-frames 2 --fast-sample-seconds 5 --fast-cycles 3`
- **레거시 모드**: `--median-frames 0 --no-adaptive` → 고정 주기 + 단일 프레임 (`FFmpegSampleReader`).
  구버전 동작을 그대로 재현하며 JSONL에 `vote_counts` 필드가 붙지 않는다. 회귀 비교용.

---

## 4. 기본 로직 ① — 한 프레임 안에서의 좌석 판정

`SeatNowAnalyzer.analyze()` (`seatnow_core.py:1210`). 프레임 1장 → `TableObservation` 리스트.

### 4-1. 두 번의 추론
1. **탐지 모델**(YOLOv8 detect) 1회 → `dining table` 후보 / 손님 물체 후보로 분류
2. **포즈 모델**(YOLOv8 pose) 1회 → 사람마다 17 keypoint

`--table-crops`(기본 켬)이면 테이블 주변만 고해상도로 잘라 물체 탐지를 한 번 더 돌린다(최대 4곳).

### 4-2. 테이블 후보 선별 — `select_table_candidates()`
YOLO의 `dining table` 탐지를 그대로 쓰지 않는다.

| 규칙 | 내용 |
|------|------|
| 통과 | 화면 면적 6% 이하(soft cap) + conf ≥ 0.20 |
| 대형 구제 | 6% 초과라도 conf ≥ 0.30 **또는** 의자 2개 이상이 구조적으로 붙어 있으면 통과 (전경 대형 테이블) |
| 약탐 구제 | conf 0.12~0.20 도 의자 2개 이상이 지지하면 통과 (뒷줄 테이블 미탐 해결) |
| hard cap | 화면 40% 초과는 무조건 거부 |
| 병합 박스 거부 | 승인된 테이블 2개 이상을 포함하거나, 더 작은 승인 테이블과 30% 이상 겹치는 대형 박스는 제거 (작은 테이블의 증거를 빼앗는 문제) |

이후 `deduplicate_tables()`로 IoU 0.65 이상 중복 제거.

### 4-3. 포즈 판정 (앉음/서있음) — `classify_pose()`
두 테스트를 **OR**:
- **Test 1** 엉덩이–무릎–발목 각도 < 110° → 앉음
- **Test 2** 어깨–엉덩이–무릎 각도 < 110° → 앉음

Seatify 논문의 3번째 테스트(Hip-Knee 수직 거리)는 **의도적으로 제외** — 서서 몸을 굽히는 동작에서 오탐이 났다.

추가 안전장치:
- **상반신만 보이는 압축 박스**(`compact_occluded_pose`)는 무조건 `UNKNOWN` — 카운터 뒤 직원이 앉은 것으로 잡히던 실제 실패 사례
- **이동 중인 사람 강등**(`_filter_moving_people`): 프레임 간 이동 속도가 임계(0.025)를 넘으면 약한 seated → standing.
  단, "이 규칙이 스스로 만든 standing"은 다음 프레임의 seated 증거를 막지 못한다(한 번 걸으면 영원히 standing으로 굳던 버그 수정)
- **약한 seated 충돌 억제**(`suppress_conflicting_weak_seated_poses`)

### 4-4. 증거 연결
- `filter_carried_objects()` — **앉지 않은 사람**의 박스와 60% 이상 겹치는 물체는 증거에서 제외.
  짐을 들고 지나가는 손님, 사람 몸 위에 뜬 환각 객체(banana 오탐 등) 차단
- `associate_objects()` — 물체를 **가장 그럴듯한 테이블 1곳**에만 배정 (테이블 상판 박스 겹침 + 바닥중심 근접도)
- `associate_people()` — 앉은 사람을 테이블에 배정 (앵커 거리 + 겹침, 테이블보다 한참 위에 뜬 사람은 배제)

**물체로 안 치는 클래스**(`EXCLUDED_OBJECT_CLASSES`): 가구(chair/couch/dining table/bench/bed), 고정 설비(tv/refrigerator/sink/oven/microwave/toaster/toilet/clock/potted plant), 동물 전체(cat/dog/bird/horse/…), 실외 오탐(car/bus/truck/traffic light/stop sign/…). 여기 없는 휴대 가능한 물체만 "손님 짐"으로 센다.

### 4-5. 최종 OR 규칙 — `occupancy_state_from_evidence()`
테이블 ROI 안에 아래 중 **하나라도** 있으면 `OCCUPIED`:
1. **손님 물체**가 있음 (컵, 노트북, 가방 등)
2. **앉아있는 사람**이 배정됨
3. ~~**연결된 의자가 점유됨**~~ → **현재 비활성. §9 참조**

- 앉았는지 알 수 없는 사람만 근처에 있으면 `UNKNOWN`
- 아무 증거 없으면 `EMPTY`
- 화면 경계에 심하게 잘린 테이블은 `IGNORE` (레이아웃 모드에서 수동으로 그린 존은 예외)
- **서있는 사람은 점유 근거가 아니다** — 의도된 설계

### 4-6. provisional 플래그
물체 증거 없이 "약한 seated"만으로 점유가 된 경우 `provisional=True`로 표시해 로그에서 구분 가능하게 남긴다.

---

## 5. 기본 로직 ② — 샘플 사이의 시간 안정화

### 5.1 버스트 다수결 — `aggregate_burst_observations()`
한 샘플 시점의 5장(`±2` 프레임)을 각각 판정한 뒤, **좌석별로** raw_state를 다수결.
- 센터 프레임의 관측이 출력의 기준(박스·레이아웃 ID)이고, 사이드 프레임은 표만 던진다
- 동률이거나 occupied/empty 과반이 없으면 **센터 프레임 값 그대로** (단일 프레임 의미 보존)
- 센터가 `IGNORE`면 투표 무시하고 통과
- 결과는 JSONL의 `vote_counts`에 `{"occupied": 3, "empty": 2}` 형태로 남는다

목적: 한 프레임의 탐지 깜빡임(손이 컵을 가림, 포즈 한 프레임 실패)이 상태를 흔들지 않게 함.

### 5.2 적응형 케이던스 — `AdaptiveCadenceController`
- **1차 판단**: 기본 15초 주기
- **2차 판단**: 비어있던(`stable EMPTY`) 좌석에 점유 증거가 뜨면, 다음 **3회를 5초 주기로 무조건** 실행
  - 예: 15초에 감지 → 20/25/30초 재판단 → 45초에 base 복귀
  - "무조건"이 핵심 — 중간에 확정/번복되어도 3회 시리즈는 끝까지 돈다. 카운트다운 중 새 트리거가 오면 시리즈를 다시 채운다
- `--no-adaptive`로 고정 주기

### 5.3 비대칭 디바운싱 — `TableTracker`
- **점유 확정 2샘플 / 빈자리 확정 3샘플** — 빈자리로 되돌리는 쪽을 더 보수적으로
  (앱에 "빈자리"라고 잘못 띄우는 것이 반대 오류보다 나쁘다)
- **트랙 TTL**: 탐지가 3샘플까지 사라져도 트랙을 유지하고 속도로 위치를 예측(`predicted=true`) — 잠깐의 가림이 좌석을 없애지 않게
- **`visible_state` vs `stable_state` 분리**: 이번 프레임이 `IGNORE`/`UNKNOWN`이어도, 이전에 확정된 점유 상태를 `EMPTY`로 뒤집지 않는다
- **장면 전환 리셋**: `is_scene_change()`가 컷을 감지하면 트래커를 새로 만들고(ID 번호는 이어감) 그 샘플의 관측을 전부 `IGNORE`로 처리

---

## 6. 기본 로직 ③ — 레이아웃 모드 (`--layout`)

`calibrate.py`로 등록한 `layouts/*.json`의 존이 **좌석의 정답**이 되는 운영 모드.

- 좌석 수·ID·이름이 고정 → 앱 표시가 안정적 (`L001` 형태 라벨)
- 존 **밖**의 탐지는 완전 무시, `inferred-seat`도 비활성
- 사람·물체 증거만 자동으로 채운다
- 경량 모델(yolov8n)과 병용을 전제로 함

**존 실시간 트래킹**(기본 켬, `--no-layout-track`으로 끔) — `LayoutZoneTracker`:
- 존과 IoU ≥ 0.30 으로 겹치는 탐지 박스를 EMA(α=0.35)로 따라감
- 탐지가 없는 존(모델이 못 보는 가구의 수동 ROI)은 **마지막 위치에 고정**
- 한 탐지가 두 존과 비슷하게(차이 < 0.10) 겹치면 **아무 존도 움직이지 않는다** — 잘못 따라가는 것이 안 움직이는 것보다 나쁘다
- 버스트의 사이드 프레임에서는 드리프트를 커밋하지 않는다 (α가 샘플당 5번 곱해지는 것 방지)

---

## 7. 기본 로직 ④ — 테이블 레이아웃 변경 감지 (신규, `0297d23`)

가구 배치 자체가 바뀌는 것을 추적하는 층. `TableTracker` 안에 있으며 `layout_version`으로 관리된다.

| 변경 | 조건 | 확정 샘플 수 |
|------|------|-----:|
| `ADDED` | 기존 트랙과 매칭 안 되는 테이블이 같은 자리에서 반복 관측 | 3 (`--table-layout-add-confirm`) |
| `MOVED` | 중심 이동 ≥ 4% **그리고** ≤ 30%, 크기 유사도 ≥ 0.55 | 3 (`--table-layout-move-confirm`) |
| `REMOVED` | 반복적으로 관측 실패 | 3 (`--table-layout-remove-confirm`) |

- 후보는 `LayoutChangeCandidate`로 모아두고 확정 전까지 `pending_changes`로만 로그에 노출 (`layout_state = MOVE_PENDING`)
- 후보 박스는 EMA(α=0.25)로 다듬어짐
- 확정되면 `layout_version`이 1 올라가고 `layout_change` 이벤트가 JSONL에 기록됨
- 좌석 점유 이력을 이어붙일 수 있도록 각 트랙에 `occupancy_history_key = [layout_version, track_id]`를 남긴다

---

## 8. 출력 포맷

### JSONL (샘플 1개 = 1줄) — `frame_log_record()`
```
frame_index, timestamp, inference_ms, scene_change, scene_metrics, scene_id
summary { visible, occupied, empty, unknown, ignore, inferred_seats,
          occupied_chairs, seated/standing/unknown_poses }
layout  { active_layout_version, layout_state, raw_table_count,
          stable_table_count, pending_changes[], committed_changes[] }
tables[] { id, label, source, layout_id/name/version/state, box, predicted,
           state, persistent_state, raw_state, confidence, reason, provisional,
           objects[], seated_people, connected_chairs[], occupied_chairs,
           pending_state, pending_count, missing_count,
           occupancy_history_key, vote_counts? }
poses[]  { state, box, anchor, confidence, reason, angles }
events[] { state_change | state_resolved | scene_change | layout_change }
run      { profile, input(sha256 포함), models(sha256 포함), config 전체 }
```
`run`에 입력 영상·모델 가중치의 sha256과 모든 설정값이 박혀 있어서 **결과 JSONL만 있으면 재현 조건이 특정된다.**

### 주석 MP4 — `render_frame()`
좌석 박스(상태별 색), 라벨, 증거 요약. `--debug`면 포즈 keypoint·각도·테이블 상판 박스까지.

---

## 9. ⚠️ 현재 코드 상태 점검 — 확인 필요한 회귀

**2026-07-18 커밋 2개 사이에서 의자(chair) 연결 로직이 통째로 빠진 채로 남아 있다.**

| 커밋 | 한 일 |
|------|------|
| `f1f41d5` "테이블 roi가 너무 큼 조정 필요" | 확장 ROI(`table_occupancy_roi`, x±45% / y±60%)를 도입하고, **그걸로 의자를 대체한다**는 판단 하에 의자 연결 함수 호출 4개를 전부 제거 |
| `70a86bc` "restore: 테이블 ROI 크기 원복" | 확장 ROI를 **되돌림**. 그런데 **의자 연결은 복구하지 않음** |

결과적으로 지금 `main`은 *"의자 연결도 없고, 넓은 ROI도 없는"* 중간 상태다.
`seatnow_core.py:1282`, `:1301`에서 `seat_detections = []`로 고정되어 있어서:

1. **판정 규칙 3번(의자→테이블 점유 전파)이 동작하지 않는다.** `occupancy_state_from_evidence()`는 `occupied_chairs` 인자를 `_ = occupied_chairs`(`seatnow_core.py:850`)로 명시적으로 버린다.
   → 의자에만 가방을 둔 손님, 테이블에서 조금 떨어져 앉은 손님을 놓칠 수 있음
2. **`inferred-seat`(테이블이 사람에 완전히 가려진 경우의 구제)가 사실상 죽어 있다** — 조건인 `has_seat_support()`(`seatnow_core.py:1518`)가 항상 `False`
3. **애매한 포즈의 의자 겹침 구제도 죽어 있다** — `seat_support_score()`(`seatnow_core.py:1331`)가 항상 `0.0`
4. **레이아웃 모드에서 캘리브레이션한 의자 존이 소비되지 않는다** — `LayoutZoneTracker`에 빈 의자 리스트를 넘기고(`seatnow_core.py:1259`), `SeatLayout.chair_boxes()` / `chair_assignments()`를 아무도 호출하지 않는다. `calibrate.py`로 의자를 찍는 작업이 현재는 결과에 영향을 주지 않음
5. **죽은 코드 4개**: `associate_chairs_to_tables`, `filter_strong_chair_links`, `associate_objects_to_chairs`, `associate_seated_people_to_chairs` — **테스트에서만 호출된다.** 그래서 테스트는 전부 통과하고 회귀가 잡히지 않았다

**선택지 (셋 중 하나로 정리 필요)**
- (a) 의자 연결을 복구한다 — `f1f41d5`의 해당 부분만 revert. 기존 §4-5의 3번 규칙이 되살아남
- (b) 확장 ROI를 다시 도입하되 마진을 줄인다 — `f1f41d5`의 x0.45/y0.60이 너무 컸던 것이 원복 사유였으므로, 예컨대 x0.20/y0.30부터 재튜닝
- (c) 현 상태가 의도라면 — 죽은 함수 4개와 그 테스트를 삭제하고, `TableObservation.connected_chairs`/`occupied_chairs` 필드와 JSONL 스키마, 이 문서의 3번 규칙을 함께 정리

> 판단 근거가 될 데이터: `sample_raw/`의 "의자 위 가방", "짐 옮김" 시나리오 영상. (a)/(b) 각각 돌려서 verify 결과를 비교하면 결론이 난다.

### 그 밖의 문서·코드 불일치
- `README.md`: "유닛 테스트 54개" → 실제 **117개** (`test_seatnow_core` 48, `cadence` 13, `burst` 12, `layout_tracker` 8, `calibrate_state` 7, `seatnow_layout` 10, `table_layout_state` 6, `video_io` 10, `verify` 3)
- `README.md`의 "현재 상태(2026-07-12)"는 레이아웃 기능을 "구현 중"이라 적고 있으나 이미 구현·머지됨

### 알려진 모델 한계 (회귀가 아니라 원래 약점)
- 비스듬한 각도에서 나란한 긴 테이블 2개가 한 박스로 병합됨 → fine-tuning으로 해결 예정
- 팬(이동) 촬영 영상은 프레임별 정확 집계가 어려움 — fixture 검증 5/20 (고정 CCTV가 제품 타깃이므로 허용, 전이 횟수 검증은 통과)

---

## 10. 테스트

```bash
./venv/bin/python -m unittest discover tests      # 117개, 모델 가중치 없이 동작
./venv/bin/python verify_seatnow.py <결과.jsonl>   # 수동 라벨 정답과 대조
```

| 파일 | 개수 | 커버 범위 |
|------|-----:|-----------|
| `test_seatnow_core.py` | 48 | 판정 로직 전반 + 실제 실패 사례 좌표 회귀 테스트 |
| `test_cadence_controller.py` | 13 | 적응형 주기(2차 판단 3회 보장 포함) |
| `test_burst_aggregation.py` | 12 | 버스트 다수결·동률 처리·매칭 |
| `test_video_io.py` | 10 | ffmpeg 리더/라이터 (ffmpeg 없으면 친절히 실패) |
| `test_seatnow_layout.py` | 10 | 레이아웃 JSON 파싱·스케일 |
| `test_layout_tracker.py` | 8 | 존 드리프트·모호성 정지 |
| `test_calibrate_state.py` | 7 | 캘리브레이션 도구 상태 |
| `test_table_layout_state.py` | 6 | ADD/MOVE/REMOVE 확정 로직 |
| `test_verify_seatnow.py` | 3 | 검증 스크립트 |

---

## 11. 개인정보보호법(PIPA) 대응

- **실시간 추론 (영상 미저장)**: 2024년 개정 예외 조항(개인정보보호법 제25조1항6호, 시행령 제22조1항) — 통계처리 목적으로 영상을 저장하지 않으면 예외 적용. 이 경우 얼굴 모자이크 불필요
- **학습용 실제 카페 영상 사용 시**: 얼굴 블러링/익명화 필수 + 사장님과의 계약에 "AI 학습 목적 데이터 활용 동의" 조항 + 원본은 일정 기간 후 파기
- 포즈 추정은 관절 좌표만 쓰므로 얼굴을 블러해도 알고리즘에 영향 없음
- **주의**: 실제 실행 전 변호사 확인 권장 (Claude가 법률 자문을 대체하지 않음)

### 운영 3원칙
1. 영상 자체는 저장하지 않음 (상태값만 전송)
2. 개인 식별은 하지 않음 (점유/비점유만 판단)
3. 카페 사장님이 운영 주체

---

## 12. 모델 전략 · 배포 · 기술 스택

### 모델 라이프사이클
```
YOLOv8n (nano) pretrained
        ↓ fine-tuning
카페 데이터로 학습된 fp32 모델 (원본, 계속 재학습 가능하도록 보존)
        ↓ 배포 시점에만
INT8 양자화 → 엣지 디바이스에 배포되는 스냅샷
```
원본(fp32)과 배포용 양자화본은 분리 관리 — 양자화는 "배포용 스냅샷 뜨기"다.

**Fine-tuning 데이터**: Roboflow Universe 공개 데이터셋 우선 (Seat Analysis / Deep Vachhani emptyoccupied / Person-on-Chair / chair detection 2 / chair+with object) → 이후 파일럿 카페의 익명화된 실제 프레임 추가. 학습 환경은 Google Colab (T4).

> 코드 기본값은 이미 경량 모델(`yolov8n.pt` / `yolov8n-pose.pt`)이다. 정확도 검증·데모는 `--det-model yolov8x.pt --pose-model yolov8x-pose.pt`.

### 배포 구조
- **MVP**: 카페 기존 PC에 소프트웨어 설치 (하드웨어 비용 0)
- **확장**: 대여형 미니PC 또는 라즈베리파이/Jetson Nano → RTSP로 IP카메라 수신

**CCTV 호환성** — ✅ IP카메라(RTSP/ONVIF), NVR+IP카메라 / ⚠️ 일부 클라우드형·아날로그+DVR(변환 시) / ❌ 순수 아날로그 구형, 완전 폐쇄형 클라우드.
국내 주요 브랜드(한화비전, 아이디스, 하이크비전, 다후아) 대부분 RTSP 지원.

```
[카페 안]                          [클라우드]      [손님]
CCTV → NVR → 엣지 디바이스     →   Supabase   →  SeatNow 앱
             (실시간 분석, 영상 폐기)   (상태값만)     (빈자리 확인)
```

### 기술 스택
| 역할 | 선택 |
|------|------|
| CV 추론 | Python + Ultralytics YOLOv8 (기본 yolov8n, 정확도 우선 시 yolov8x) + OpenCV |
| 영상 I/O | ffmpeg 바이너리 (파이프) |
| 모델 서빙 | FastAPI (또는 Next.js API Route) |
| 학습 환경 | Google Colab (T4 GPU) |
| 실시간 DB | Supabase Realtime |
| 프레임 큐 | Redis (확장 시) |
| 엣지 | 라즈베리파이 / Jetson Nano |
| 카메라 연결 | RTSP / ONVIF |

---

## 13. 로드맵 & 다음 작업

**현재 위치**: 영상 처리 + 시간 안정화 + 레이아웃/변경감지까지 완료 — **"작동하는 데모" 단계, 파일럿 카페 섭외 가능**

```
✅ 1. 물체 기반 점유 판정
✅ 2. 포즈 기반 앉음/서있음 판정
✅ 3. 둘 통합
✅ 4. [Stage 1] 영상 처리 엔진 (ffmpeg 파이프, 주석 영상 + JSONL)
✅ 5. [Stage 2] 시간 안정화 (비대칭 디바운싱, TableTracker, 장면 전환 리셋)
✅ 5b. 버스트 다수결 + 적응형 2차 판단 주기 (2026-07-18)
✅ 4b. [Stage 4 선행] 좌석 캘리브레이션 도구 + 레이아웃 모드 + 존 트래킹 + 레이아웃 변경 감지

   ── ✅ 여기까지 완성 ──

⚠️ 0. **§9의 의자 연결 회귀 정리** ← 파일럿 나가기 전에 결론 내야 함
   6. [Stage 3] Fine-tuning — 파일럿 카페 실 데이터 + Roboflow로 YOLOv8n 학습, 목표 92%
   7. [Stage 4] 캘리브레이션 어드민 UI (현재는 CLI 도구)
   8. [Stage 5] 백엔드 연동 — 상태값만 Supabase 전송
   9. [Stage 6] 손님용 앱 화면 — "몇 자리 남음" + 시팅맵
  10. [Stage 7] 사장님 배포 패키지 — 설치 패키징, RTSP 가이드, 개인정보 처리방침 문구
  11. (더 나중) 수요 예측 대시보드 (XGBoost 등)
```

### 지금 당장 다음 작업
1. **§9 의자 연결 회귀 결론** — `sample_raw/`의 "의자 위 가방"·"짐 옮김" 시나리오로 (a)/(b) 비교 후 택1
2. `README.md` 테스트 개수·현재 상태 문단 갱신
3. 파일럿 카페 섭외 (데모 자료: `sample_results/v5_*.mp4`)
4. 확보 후 실제 CCTV 프레임으로 YOLOv8n fine-tuning — 병합 테이블·저신뢰 테이블 문제의 근본 해결책

```bash
# 실행 예시
./venv/bin/python seatnow.py sample_raw/영상.mp4 --debug
# 짧은 클립은 주기를 줄여서
./venv/bin/python seatnow.py sample_raw/영상.mp4 --debug --sample-seconds 5 --fast-sample-seconds 2
# 구버전 동작 재현
./venv/bin/python seatnow.py sample_raw/영상.mp4 --sample-seconds 1 --median-frames 0 --no-adaptive
```

---

## 14. 관련 문서

- `README.md` — 팀원용 셋업/실행 (일부 내용 갱신 필요, §9)
- `ONBOARDING.md` — 셋업 시 함정 2가지 (ffmpeg 필수 / numpy<2 유지)
- `docs/superpowers/specs/2026-07-11-manual-seat-layout-design.md` — 수동 레이아웃 설계
- `docs/superpowers/specs/2026-07-13-layout-zone-tracking-design.md` — 존 트래킹 설계
- `docs/superpowers/plans/2026-07-11-manual-seat-layout.md` — 구현 계획
- (과거) `SEATNOW_HANDOFF.md`, `SEATNOW_STRATEGY.md` — 이 문서가 그 둘의 통합·최신본
