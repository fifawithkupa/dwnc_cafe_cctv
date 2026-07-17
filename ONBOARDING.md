# dwnc_cafe_cctv (SeatNow) 온보딩 — 주의점 2가지

```bash
git clone https://github.com/fifawithkupa/dwnc_cafe_cctv.git
```

셋업 전에 아래 두 가지만 꼭 확인하세요. 나머지 상세 절차는 저장소의 `README.md`를 따르면 됩니다.

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
# 기대: Ran 54 tests ... OK
```

## 2. numpy는 반드시 2 미만

`requirements.txt`에 `numpy<2`로 고정되어 있습니다. **임의로 numpy를 2.x로 업그레이드하지 마세요** — PyTorch와 충돌합니다.

Apple Silicon이나 Windows에서 torch 설치가 꼬이면:

1. `requirements.txt`에서 **torch 핀(`torch==2.2.2`)만 삭제**
2. 다시 `pip install -r requirements.txt`
3. **`numpy<2`는 그대로 유지**
