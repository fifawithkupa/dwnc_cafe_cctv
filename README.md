# SeatNow — 카페 CCTV 좌석 점유 감지 (dwnc_cafe_cctv)

카페 CCTV 영상을 분석해 테이블별 점유 여부를 판정하고, 손님 앱에 "몇 자리 남음"을
보여주기 위한 CV 파이프라인. YOLOv8(탐지+포즈) 기반.

**전체 배경/로드맵/판정 로직은 [`SEATNOW_전체정리.md`](SEATNOW_전체정리.md) 를 먼저 읽어주세요.**

## 저장소 구성

| 파일 | 역할 |
|------|------|
| `seatnow_core.py` | 본체: 추론·점유 판정·의자/물체/사람 연결·추적(디바운싱)·FFmpeg 영상 I/O·렌더링 |
| `seatnow.py` | CLI 진입점 (이미지/영상 → 주석 영상 + JSONL 로그) |
| `seatnow_layout.py`, `calibrate.py` | 수동 좌석 레이아웃(테이블·의자 존) 정의·로드 |
| `verify_seatnow.py` | 결과 JSONL을 수동 라벨 정답과 대조 검증 (영상 여러 개 동시 채점) |
| `make_labels.py` | 라벨링용 대조표 프레임 추출 + fixture 스켈레톤 생성·검사 |
| `export.py` | `.pt` → OpenVINO FP32/INT8 익스포트 (엣지 배포용) |
| `bench.py` | 추론 latency 측정 → tick 예산 산출 |
| `bench_sweep.py` | 파라미터 그리드 스윕 → 정확도 × tick 비용 표 |
| `rtsp_republish.py` | 샘플 영상을 로컬 RTSP로 재송출 (카메라 없이 라이브 검증) |
| `tests/` | 유닛 테스트 171개 (모델 없이 순수 로직 검증) |
| `docs/superpowers/` | 설계 스펙·구현 계획 |
| `plan.md` | 코드 작업 플랜 (T1~T12) |
| `occupancy_mvp.py`, `pose_judge.py` | 초기 PoC (참고용) |

## 환경 설정 (팀원용)

요구사항: **Python 3.9~3.11**, **ffmpeg** (필수 — 영상 입출력이 ffmpeg 바이너리 사용)

```bash
git clone https://github.com/fifawithkupa/dwnc_cafe_cctv.git
cd dwnc_cafe_cctv

# 1. ffmpeg 설치 (없다면)
#    macOS: brew install ffmpeg   /  Windows: winget install ffmpeg  /  Ubuntu: apt install ffmpeg

# 2. 가상환경 + 의존성
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
#    Windows: venv\Scripts\pip install -r requirements.txt
#    Apple Silicon/Windows에서 torch 버전 충돌 시: torch 핀을 지우고 재설치 (numpy<2 는 유지)

# 3. 테스트로 환경 확인 (모델 다운로드 없이 돌아감)
./venv/bin/python -m unittest discover tests
# 기대: Ran 171 tests ... OK
```

**모델 가중치는 저장소에 없습니다.** 첫 실행 때 ultralytics가 자동 다운로드합니다
(기본: `yolov8n.pt`/`yolov8n-pose.pt` 경량 모델. 정확도 검증·데모용은
`--det-model yolov8x.pt --pose-model yolov8x-pose.pt` — 약 270MB, 역시 자동 다운로드).

**샘플 영상(`sample_raw/`)과 결과(`sample_results/`)도 저장소에 없습니다** (용량·개인정보).
팀 드라이브로 공유 — 받은 뒤 프로젝트 루트에 같은 이름의 폴더로 두면 됩니다.

## 실행

```bash
# 영상 분석 → 주석 영상 + JSONL (sample_results/에 저장)
./venv/bin/python seatnow.py sample_raw/cafe_sample_1.mp4 --debug

# 정확도 우선(느림, 프레임당 ~8초 on CPU)
./venv/bin/python seatnow.py sample_raw/cafe_sample_1.mp4 --debug \
  --det-model yolov8x.pt --pose-model yolov8x-pose.pt

# 수동 라벨 정답과 대조 (fixture 영상 결과에 대해)
./venv/bin/python verify_seatnow.py sample_results/<결과>.jsonl

# 영상 여러 개를 한 번에 채점 (fixture는 영상 sha256으로 자동 매칭)
./venv/bin/python verify_seatnow.py sample_results/*.jsonl --expectations tests/fixtures
```

### 평가·벤치 도구

```bash
# 1. 새 영상 라벨링: 대조표 프레임 + fixture 스켈레톤 생성
./venv/bin/python make_labels.py sample_raw/cafe_1h.mp4 --interval 30 \
  --contact-sheet labels/cafe_1h --layout layouts/cafe.json
#    → labels/cafe_1h/*.jpg 를 보며 occupied/empty/ignore 를 손으로 채운 뒤
./venv/bin/python make_labels.py x --validate tests/fixtures/cafe_1h_expectations.json

# 2. 엣지 배포용 익스포트 + 추론 latency/tick 예산 측정
./venv/bin/python export.py --imgsz 640 960 1280
./venv/bin/python bench.py --frames sample_raw/cafe_sample_1.mp4 --label macbook

# 3. 파라미터 스윕 (라벨된 평가셋 필요)
./venv/bin/python bench_sweep.py sample_raw/*.mp4 --dry-run

# 4. 카메라 없이 RTSP 파이프라인 검증
./venv/bin/python rtsp_republish.py sample_raw/cafe_sample_1.mp4
```

진단이 필요할 때는 `--log-detections` 를 붙이면 JSONL에 detect 원본 출력과
테이블 후보 탈락 사유가 함께 남습니다 — "모델이 못 봤나" vs "코드가 버렸나"를
로그만으로 구분할 수 있습니다.

주요 옵션: `--sample-seconds`(1차 판단 주기, 기본 15초), `--fast-sample-seconds`(2차 판단
주기, 기본 5초 — empty 좌석에 착석/물건 증거가 뜨면 `--fast-cycles`(기본 3)회 재판단을
무조건 수행: 예. 15초 감지 → 20/25/30초 재판단 → 45초 base 복귀),
`--median-frames N`(샘플 시점 ±N 연속 프레임 다수결, 기본 2 → 5장), `--no-adaptive`(주기
고정), `--max-samples N`(스모크 테스트), `--no-video`(로그만). 전체는 `--help`.

> 짧은 샘플 클립(대부분 15초 미만)은 기본 15초 주기로는 샘플이 1–2개뿐이므로
> 데모 시 `--sample-seconds 5 --fast-sample-seconds 2`처럼 줄여서 실행하세요.
> 기존 단일 프레임·고정 주기 동작은 `--sample-seconds 1 --median-frames 0 --no-adaptive`.

## 현재 상태 (2026-08-26)

- ✅ 이미지/영상 점유 판정 + 시간 안정화(디바운싱·추적) — 시나리오 영상 7종 검증 통과
- ✅ 점유판정 개선: 테이블 선별 규칙(의자 구조 기반 구제), 들고 있는 짐 오탐 차단, 의자 위 짐 점유 인식
- ✅ **수동 좌석 캘리브레이션(레이아웃) 머지 완료** — 설계/계획:
  - 스펙: `docs/superpowers/specs/2026-07-11-manual-seat-layout-design.md`
  - 구현 계획(태스크 단위): `docs/superpowers/plans/2026-07-11-manual-seat-layout.md`
- ✅ 의자→테이블 점유 전파 회귀 복구 (plan.md T1). `f1f41d5`가 제거하고
  `70a86bc`가 되살리지 않아 `seat_detections`가 항상 비어 있던 문제
- ✅ 진단 로깅(T2), 다중 영상 평가셋·커버리지 집계(T3), OpenVINO 익스포트·벤치(T4),
  RTSP 재송출 하네스(T5), 파라미터 스윕 하네스(T7 코드)
- 🚧 다음: 보유 영상 라벨링(T3 데이터) → 엣지 박스 벤치(T6) → RTSP 리더(T8) →
  Quick Sync(T9) → 배포 프로파일(T10). 파인튜닝(T11)은 조건부

작업 순서와 근거는 [`plan.md`](plan.md) 참조.

## 개발 규칙

- 판정 로직 변경 시 반드시 유닛 테스트 추가 (`tests/test_seatnow_core.py` — 실패 사례의 실제 좌표로 회귀 테스트 작성하는 관례)
- **`analyze()`의 배선을 바꿨다면 `tests/test_analyze_pipeline.py`에도 추가.**
  헬퍼만 직접 호출하는 테스트는 "호출이 사라지는" 회귀를 못 잡는다 (T1이 그렇게 새어나갔다)
- 커밋 전 `./venv/bin/python -m unittest discover tests` 통과 확인
- 시나리오 영상 재실행 결과는 `sample_results/v5_*` 네이밍 사용
