# dwnc_cafe_cctv (SeatNow) 온보딩 — 주의점 3가지

```bash
git clone https://github.com/fifawithkupa/dwnc_cafe_cctv.git
```

셋업 전에 아래 세 가지만 꼭 확인하세요. 나머지 상세 절차는 저장소의 `README.md`를 따르면 됩니다.

## 1. ffmpeg 필수

이 프로젝트는 영상 읽기/쓰기를 OpenCV가 아니라 **ffmpeg 바이너리**로 처리합니다.
ffmpeg이 설치되어 있지 않으면 영상 처리가 전혀 동작하지 않습니다 (테스트가 친절하게 실패하니 테스트로 먼저 확인하세요).

```bash
# macOS
brew install ffmpeg
# Windows
winget install ffmpeg
# Ubuntu
sudo apt install ffmpeg
```

설치 확인:

```bash
python -m unittest discover tests
# 기대: Ran 229 tests ... OK
```

## 2. 샘플 영상은 구글 드라이브에 있다 — 반드시 "오프라인 사용 가능"으로 고정할 것

영상(`sample_raw/`)과 결과(`results/`)는 용량·개인정보 때문에 저장소에 없고
팀 구글 드라이브로 공유됩니다.

**윈도우/맥은 [Google Drive for Desktop](https://www.google.com/drive/download/)을 쓰세요.**
rclone 같은 CLI 도구는 필요 없습니다.

1. 앱 설치 후 로그인 → 드라이브가 마운트됩니다
   - Windows: `G:\내 드라이브\...`
   - macOS: `~/Library/CloudStorage/GoogleDrive-<계정>/내 드라이브/...`
2. **영상 폴더에서 우클릭 → "오프라인 사용 가능"** 으로 고정합니다
3. 프로젝트 루트에 링크를 걸거나, 실행 시 경로를 직접 넘깁니다

```bash
# macOS 예시 — sample_raw 를 드라이브 폴더로 연결
ln -s "$HOME/Library/CloudStorage/GoogleDrive-<계정>/내 드라이브/SeatNow/sample_raw" sample_raw
```

> ⚠️ **고정(pin)을 건너뛰지 마세요.** 스트리밍 상태로 두면 실행할 때마다 영상 전체가
> 네트워크로 다시 내려옵니다. `engine/seatnow.py`는 실행 메타데이터에 남길 sha256을 매번
> 입력 영상 전체를 읽어 계산하는데(`engine/seatnow.py:287`), 정작 분석은 15초마다 몇 프레임만
> 읽습니다. 특히 `edge/bench_sweep.py`는 그리드 1점당 영상을 1회씩 실행하므로
> 12점 그리드면 전체 파일을 12번 내려받게 됩니다.

리눅스(엣지 박스)에는 공식 앱이 없으니 그쪽만 `rclone`을 씁니다:

```bash
rclone config                                   # drive 리모트 생성 (브라우저 OAuth)
rclone copy gdrive:SeatNow/sample_raw sample_raw --include "*.mp4" -P
```

## 3. numpy는 반드시 2 미만

`requirements.txt`에 `numpy<2`로 고정되어 있습니다. **임의로 numpy를 2.x로 업그레이드하지 마세요** — PyTorch와 충돌합니다.

Apple Silicon이나 Windows에서 torch 설치가 꼬이면:

1. `requirements.txt`에서 **torch 핀(`torch==2.2.2`)만 삭제**
2. 다시 `pip install -r requirements.txt`
3. **`numpy<2`는 그대로 유지**
