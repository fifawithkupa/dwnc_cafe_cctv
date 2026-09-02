# SeatNow — 카페 CCTV 좌석 점유 감지 (dwnc_cafe_cctv)

카페 CCTV 영상을 분석해 테이블별 점유 여부를 판정하고, 손님 앱에 "몇 자리 남음"을
보여주기 위한 CV 파이프라인. YOLOv8(탐지+포즈) 기반.

**전체 배경/로드맵/판정 로직은 [`SEATNOW_전체정리.md`](SEATNOW_전체정리.md) 를 먼저 읽어주세요.**

## 저장소 구성

폴더는 **무엇을 할 때 쓰는지**로 나뉜다.

| 폴더 | 언제 쓰나 |
|---|---|
| `engine/` | **판정.** 영상을 보고 자리가 찼는지 정하는 코드. 여기만 매장에서 돌아간다 |
| `install/` | **설치 당일 한 번.** 좌석을 그리고(캘리브레이션) 손님용 2D 평면도를 만든다 |
| `checks/` | **우리가 잘 하고 있는지 채점.** 판독표·정답 대조·라벨링 |
| `edge/` | **엣지 박스 성능 재기.** 벤치·익스포트·RTSP 재송출 |
| `results/` | **결과는 전부 여기.** 판독표·사진·로그. 아래 참조 |
| `layouts/` | 매장별 좌석 도면 (판정의 입력) |

실행은 저장소 최상위에서 `python -m <폴더>.<파일>` 형태다.
예: `python -m engine.seatnow ...`, `python -m install.calibrate ...`

### 결과 보는 곳 — `results/`

한 번 돌린 결과는 **폴더 하나에 다 들어 있다.**

```
results/
  angle1/          ← 영상 하나를 돌린 결과
    report.md        판독표 — 사람이 읽는 것. 여기부터 보면 된다
    log.jsonl        원시 판정 로그 (tick마다 한 줄)
    clean/           주석 없는 원본 사진 (세는 용도)
    marked/          판정을 그려 넣은 사진 (진단 용도)
    judge/           Codex가 매긴 정답지
  angle1_layout/   ← 사람이 그린 평면도로 다시 돌린 결과 (지금 쓰는 것)
    review/          사람 눈으로 한 장씩 넘겨보는 폴더
  preseed/         캘리브레이션 자동 초안 미리보기
  edge/            벤치 결과 (bench_report.json, decode_report.json, clips/)
```

저장소에 올라가는 건 **`report.md`와 `judge/`**다. 판독표는 팀이 같이 봐야 하고,
Codex 정답지는 다시 만들려면 Codex를 또 돌려야 하기 때문이다. 사진·로그·영상은
용량과 개인정보 때문에 각자 컴퓨터에만 남는다.

| 파일 | 역할 |
|------|------|
| `engine/seatnow_core.py` | 본체: 추론·점유 판정·의자/물체/사람 연결·추적(디바운싱)·FFmpeg 영상 I/O·렌더링 |
| `engine/seatnow.py` | CLI 진입점 (이미지/영상 → 주석 영상 + JSONL 로그) |
| `engine/seatnow_layout.py`, `install/calibrate.py` | 수동 좌석 레이아웃(테이블·의자 존, 바 구역·자리 칸) 정의·로드 |
| `engine/seatnow_report.py` | 앱용 좌석 가용성 계약(`seat_report`) 생성 + UNKNOWN 사유 코드 |
| `checks/verify_seatnow.py` | 결과 JSONL을 수동 라벨 정답과 대조 검증 (영상 여러 개 동시 채점) |
| `checks/make_labels.py` | 라벨링용 대조표 프레임 추출 + fixture 스켈레톤 생성·검사 |
| `edge/export.py` | `.pt` → OpenVINO FP32/INT8 익스포트 (엣지 배포용) |
| `edge/bench.py` | 추론 latency 측정 → tick 예산 산출 |
| `edge/bench_sweep.py` | 파라미터 그리드 스윕 → 정확도 × tick 비용 표 |
| `edge/rtsp_republish.py` | 샘플 영상을 로컬 RTSP로 재송출 (카메라 없이 라이브 검증) |
| `engine/frame_dump.py` | 판정한 tick마다 사진 두 장 저장 (`clean/` 세는 용도, `marked/` 진단 용도) |
| `checks/judge_frames.py` | 깨끗한 사진마다 Codex를 새로 불러 "사람 몇 명"을 세게 함 (눈가림 채점) |
| `checks/inspect_run.py` | 검출·포즈·좌석 세 층을 한 줄에 놓은 판독표 + 층별 재현율 |
| `checks/make_review.py` | 사람이 한 장씩 넘겨보는 `review/` 폴더 — 자리 이름·판정·근거 글자만 남긴 사진 |
| `checks/judge_schema.json` | Codex가 답해야 하는 JSON 모양 |
| `tests/` | 유닛 테스트 546개 (모델 없이 순수 로직 검증) |
| `docs/superpowers/` | 설계 스펙·구현 계획 |
| `plan.md` | 코드 작업 플랜 (T1~T12) |

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
# 기대: Ran 514 tests ... OK
```

**모델 가중치는 저장소에 없습니다.** 첫 실행 때 ultralytics가 자동 다운로드합니다
(기본: `yolov8n.pt`/`yolov8n-pose.pt` 경량 모델. 정확도 검증·데모용은
`--det-model yolov8x.pt --pose-model yolov8x-pose.pt` — 약 270MB, 역시 자동 다운로드).

**샘플 영상(`sample_raw/`)은 저장소에 없습니다** (용량·개인정보).
`results/`는 판독표(`report.md`)만 올라가고 사진·로그·영상은 빠집니다.
팀 구글 드라이브로 공유 — 받는 방법과 **"오프라인 사용 가능" 고정이 왜 필수인지**는
[`ONBOARDING.md` §2](ONBOARDING.md) 참조.

## 실행

```bash
# 영상 분석 → 주석 영상 + JSONL (results/에 저장)
./venv/bin/python -m engine.seatnow sample_raw/cafe_sample_1.mp4 --debug

# 정확도 우선(느림, 프레임당 ~8초 on CPU)
./venv/bin/python -m engine.seatnow sample_raw/cafe_sample_1.mp4 --debug \
  --det-model yolov8x.pt --pose-model yolov8x-pose.pt

# 수동 라벨 정답과 대조 (fixture 영상 결과에 대해)
./venv/bin/python -m checks.verify_seatnow results/<결과>/log.jsonl

# 영상 여러 개를 한 번에 채점 (fixture는 영상 sha256으로 자동 매칭)
./venv/bin/python -m checks.verify_seatnow results/*/log.jsonl --expectations tests/fixtures
```

### 평가·벤치 도구

```bash
# 0. 설치 시 좌석 캘리브레이션 (1회)
#    yolov8x가 테이블·의자를 미리 잡아주고, 사람은 잘못 잡힌 것만 지우고
#    못 잡은 것만 추가한다. 일자형·벽 책상은 모델이 절대 못 잡으므로
#    [z]로 바 구역을 치고 [x]로 자리마다 칸을 긋는다 (칸 수 = 자리 수).
./venv/bin/python -m install.calibrate sample_raw/cafe_sample_angle1.mov \
  --output layouts/cafe_angle1.json
#    키: [t]able [c]hair [z]one seat[x] [g]en-seats [f]loor [m]ove [d]elete [u]ndo [s]ave [q]uit
#    [f] 바닥에서 실제로 직사각형인 것의 네 귀퉁이를 시계방향 클릭 (2D 평면도용)
#    [m] 의자 클릭 -> m -> 옮길 테이블/바 클릭. 지웠다 다시 그리지 않고 소속만 바꾼다
#    [g] 바 구역을 선택하고 누르면 붙어 있는 의자에서 자리 칸을 그대로 만든다

# 1. 새 영상 라벨링: 대조표 프레임 + fixture 스켈레톤 생성
./venv/bin/python -m checks.make_labels sample_raw/cafe_1h.mp4 --interval 30 \
  --contact-sheet labels/cafe_1h --layout layouts/cafe.json
#    → labels/cafe_1h/*.jpg 를 보며 occupied/empty/ignore 를 손으로 채운 뒤
./venv/bin/python -m checks.make_labels x --validate tests/fixtures/cafe_1h_expectations.json

# 2. 엣지 배포용 익스포트 + 추론 latency/tick 예산 측정
./venv/bin/python -m edge.export --imgsz 640 960 1280
./venv/bin/python -m edge.bench --frames sample_raw/cafe_sample_1.mp4 --label macbook

# 3. 파라미터 스윕 (라벨된 평가셋 필요)
./venv/bin/python -m edge.bench_sweep sample_raw/*.mp4 --dry-run

# 4. 카메라 없이 RTSP 파이프라인 검증
./venv/bin/python -m edge.rtsp_republish sample_raw/cafe_sample_1.mp4

# 5. 엣지 박스 검수 (새 박스에서 제일 먼저)
./venv/bin/python -m edge.check_edge

# 6. 디코딩 비용 측정 → 살 카메라의 해상도·코덱 결정
./venv/bin/python -m edge.bench_decode --source sample_raw/cafe_sample_angle1.mov

# 7. 검출 검사 하네스 — "모델이 이 카페를 제대로 보는가"
#    라벨(T16) 없이 지금 돌릴 수 있다. 레이아웃도 주지 않는다 —
#    사람이 그려준 정답을 빼고 모델만 놓고 봐야 답이 나오기 때문이다.
./venv/bin/python -m engine.seatnow sample_raw/cafe_sample_angle1.mov \
  --no-video --log-detections \
  --frame-dir results/angle1 --log results/angle1/log.jsonl

#    깨끗한 사진마다 Codex가 사람 수를 센다. 우리 답은 안 보여준다
./venv/bin/python -m checks.judge_frames results/angle1

#    세 층을 한 줄에 놓고 층별 재현율을 낸다
./venv/bin/python -m checks.inspect_run results/angle1/log.jsonl \
  --judge results/angle1/judge --output results/angle1/report.md

#    사람이 한 장씩 넘겨보는 검수 폴더 (review/) 를 만든다
./venv/bin/python -m checks.make_review results/angle1 --title angle1
```

> `checks/judge_frames.py`를 안 돌려도 판독표는 나온다. `실제` 칸이 `___`로 비어
> 있을 뿐이고, 사람이 사진을 보며 손으로 채워도 같은 표가 된다.

### 엣지 박스 세팅

새로 산 미니PC를 켜서 "이 박스로 어떤 카메라를 살 수 있나"를 재는 데까지의 절차는
[`docs/edge-setup.md`](docs/edge-setup.md)에 있습니다 (Windows·Linux 양쪽).

 `edge/bench.py`가 재는 **추론**은 프레임을 `imgsz`로 줄여서 넣기 때문에 카메라 해상도와
거의 무관합니다. 카메라 해상도가 실제로 잡아먹는 것은 24시간 도는 **디코딩**이고,
그것을 재는 것이 `edge/bench_decode.py`입니다. 두 도구의 결과가 합쳐져야 카메라를 고를
수 있습니다.

디코딩은 `--hwaccel`(기본 `auto`)로 하드웨어 가속을 씁니다. **켜졌는지 꺼졌는지가
항상 화면에 찍힙니다** — ffmpeg는 하드웨어 디코딩에 실패해도 조용히 소프트웨어로
넘어가므로, 후보마다 실제로 프레임이 나오는지 확인하고 나온 것만 채택합니다.
이 노트북 실측으로 하드웨어 디코딩은 비용을 3~6배 줄입니다.

진단이 필요할 때는 `--log-detections` 를 붙이면 JSONL에 detect 원본 출력과
테이블 후보 탈락 사유가 함께 남습니다 — "모델이 못 봤나" vs "코드가 버렸나"를
로그만으로 구분할 수 있습니다.

주요 옵션: `--sample-seconds`(판단 주기, 기본 15초 고정),
`--median-frames N`(샘플 시점 ±N 연속 프레임 다수결, 기본 2 → 5장),
`--max-samples N`(스모크 테스트), `--no-video`(로그만). 전체는 `--help`.

> 짧은 샘플 클립(대부분 15초 미만)은 기본 15초 주기로는 샘플이 1–2개뿐이므로
> 데모 시 `--sample-seconds 5`처럼 줄여서 실행하세요.
> 기존 단일 프레임 동작은 `--sample-seconds 1 --median-frames 0`.

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
- ✅ 좌석 리포트 출력 계약 `seat_report` + 바형 좌석 칸 (T13)
- ✅ Quick Sync 하드웨어 디코딩(T9), 디코딩 벤치 `edge/bench_decode.py`,
  엣지 박스 검수 `edge/check_edge.py`, 설치 절차 `docs/edge-setup.md`
- 🚧 다음: 엣지 박스 도착 → `edge/check_edge.py` → `edge/bench.py` → `edge/bench_decode.py`
  → 카메라 확정 → RTSP 리더(T8) → 배포 프로파일(T10). 파인튜닝(T11)은 조건부

작업 순서와 근거는 [`plan.md`](plan.md) 참조.

## 개발 규칙

- 판정 로직 변경 시 반드시 유닛 테스트 추가 (`tests/test_seatnow_core.py` — 실패 사례의 실제 좌표로 회귀 테스트 작성하는 관례)
- **`analyze()`의 배선을 바꿨다면 `tests/test_analyze_pipeline.py`에도 추가.**
  헬퍼만 직접 호출하는 테스트는 "호출이 사라지는" 회귀를 못 잡는다 (T1이 그렇게 새어나갔다)
- 커밋 전 `./venv/bin/python -m unittest discover tests` 통과 확인
- 시나리오 영상 재실행 결과는 `results/v5_*/` 네이밍 사용
