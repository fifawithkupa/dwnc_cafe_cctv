# SeatNow — 카페 CCTV 좌석 점유 감지 (dwnc_cafe_cctv)

카페 CCTV 영상을 분석해 테이블별 점유 여부를 판정하고, 손님 앱에 "몇 자리 남음"을
보여주기 위한 CV 파이프라인. YOLOv8(탐지+포즈) 기반.

**전체 배경/로드맵/판정 로직은 [`SEATNOW_전체정리.md`](SEATNOW_전체정리.md) 를 먼저 읽어주세요.**

## 저장소 구성

| 파일 | 역할 |
|------|------|
| `seatnow_core.py` | 본체: 추론·점유 판정·의자/물체/사람 연결·추적(디바운싱)·FFmpeg 영상 I/O·렌더링 |
| `seatnow.py` | CLI 진입점 (이미지/영상 → 주석 영상 + JSONL 로그) |
| `verify_seatnow.py` | 결과 JSONL을 수동 라벨 정답과 대조 검증 |
| `tests/` | 유닛 테스트 54개 (모델 없이 순수 로직 검증) |
| `docs/superpowers/` | 설계 스펙·구현 계획 (진행 중: 수동 좌석 레이아웃/캘리브레이션) |
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
# 기대: Ran 54 tests ... OK
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
```

주요 옵션: `--sample-seconds`(1차 판단 주기, 기본 15초), `--fast-sample-seconds`(2차 판단
주기, 기본 5초 — empty 좌석에 착석/물건 증거가 뜨면 `--fast-cycles`(기본 3)회 재판단을
무조건 수행: 예. 15초 감지 → 20/25/30초 재판단 → 45초 base 복귀),
`--median-frames N`(샘플 시점 ±N 연속 프레임 다수결, 기본 2 → 5장), `--no-adaptive`(주기
고정), `--max-samples N`(스모크 테스트), `--no-video`(로그만). 전체는 `--help`.

> 짧은 샘플 클립(대부분 15초 미만)은 기본 15초 주기로는 샘플이 1–2개뿐이므로
> 데모 시 `--sample-seconds 5 --fast-sample-seconds 2`처럼 줄여서 실행하세요.
> 기존 단일 프레임·고정 주기 동작은 `--sample-seconds 1 --median-frames 0 --no-adaptive`.

## 현재 상태 (2026-07-12)

- ✅ 이미지/영상 점유 판정 + 시간 안정화(디바운싱·추적) 완료 — 시나리오 영상 7종 검증 통과
- ✅ 점유판정 개선: 테이블 선별 규칙(의자 구조 기반 구제), 들고 있는 짐 오탐 차단, 의자 위 짐 점유 인식
- 🚧 **수동 좌석 캘리브레이션(레이아웃) 기능 구현 중** — 설계/계획:
  - 스펙: `docs/superpowers/specs/2026-07-11-manual-seat-layout-design.md`
  - 구현 계획(태스크 단위): `docs/superpowers/plans/2026-07-11-manual-seat-layout.md`
- 다음: 레이아웃 완성 → 파일럿 카페 섭외 → YOLOv8n fine-tuning (Colab T4)

## 개발 규칙

- 판정 로직 변경 시 반드시 유닛 테스트 추가 (`tests/test_seatnow_core.py` — 실패 사례의 실제 좌표로 회귀 테스트 작성하는 관례)
- 커밋 전 `./venv/bin/python -m unittest discover tests` 통과 확인
- 시나리오 영상 재실행 결과는 `sample_results/v5_*` 네이밍 사용
