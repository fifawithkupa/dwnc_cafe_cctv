# 엣지 박스 설치·검수 절차

> 작성: 2026-08-31
> 관련: `docs/superpowers/specs/2026-08-31-edge-decode-bench-design.md`, `plan.md` T6·T9

> ⚠️ **엣지 박스 첫 실행이라면 `docs/edge-first-run.md` 를 먼저 본다.**
> 이 문서는 "어떤 카메라를 살 수 있나"까지 가는 긴 절차라 첫날 순서와 다르다.
> 지금 박스는 **OptiPlex 7040 Micro / i3-6100T / RAM 4GB / Linux** 이고,
> 그 문서가 이 장비에 맞춰 다시 쓰여 있다.

## 이 문서로 하는 일

새로 산 미니PC를 켜서 **"이 박스로 어떤 카메라를 살 수 있나"를 재는 데까지** 간다.
카메라는 아직 없어도 된다. 보유한 카페 영상을 실제 카메라 해상도·화질로 바꿔서
재기 때문이다.

끝나면 이런 표가 나온다.

```
| 클립     | 디코딩 방식 | 점유 코어 | 전체 대비 | 판정 |
| 4mp_h265 | none        | 0.38      | 5%        | PASS |
| 4mp_h265 | d3d11va     | 0.07      | 1%        | PASS |
```

`FAIL`이 뜬 해상도·코덱의 카메라는 **사면 안 된다.**

---

## 1. OS를 무엇으로 할 것인가

**결론: 측정할 때는 박스에 깔려 온 OS 그대로 쓰고, 실제 매장에 놓을 때 리눅스로 바꾼다.**

| 단계 | 권장 | 이유 |
|---|---|---|
| 지금 (측정) | 깔려 온 것 그대로 | 추론 속도는 Windows/Linux 차이가 거의 없다. 리눅스 설치에 반나절 쓸 이유가 없다 |
| 나중 (24시간 운영) | Ubuntu Server | 성능이 아니라 **사고 위험** 때문이다 |

리눅스를 권하는 이유는 속도가 아니다. **Windows는 새벽에 강제 업데이트로 재부팅하고
로그인 화면에서 멈춘다.** 무인으로 놔둔 박스가 그러면 아침에 좌석 정보가 죽어 있고,
매장에서는 "앱이 고장났다"로 보인다. 리눅스는 그런 일이 없고, 프로그램이 죽어도
자동으로 다시 띄우는 장치(systemd)가 OS에 들어 있다.

지금 안 정해도 된다. 하드웨어 디코딩 부분이 OS별로 갈아끼울 수 있게 만들어져 있어서
나중에 옮겨도 코드를 다시 쓰지 않는다.

---

## 2-A. Windows 길

### 1) Python 설치

[python.org](https://www.python.org/downloads/) 에서 **3.11** 을 받는다.
설치 화면에서 **"Add python.exe to PATH"** 를 반드시 체크한다.

확인:
```
python --version
```
`Python 3.11.x` 가 나와야 한다.

### 2) ffmpeg 설치

영상을 푸는 프로그램이다. 이게 없으면 아무것도 안 된다.

1. https://www.gyan.dev/ffmpeg/builds/ 에서 **release full build** (zip)을 받는다
2. `C:\ffmpeg` 에 압축을 푼다 (`C:\ffmpeg\bin\ffmpeg.exe` 가 있어야 한다)
3. 시작 메뉴에 "환경 변수"를 검색 → **시스템 환경 변수 편집** → **환경 변수** →
   아래쪽 **시스템 변수**의 `Path` 선택 → **편집** → **새로 만들기** →
   `C:\ffmpeg\bin` 입력 → 확인
4. **명령 프롬프트를 새로 연다** (기존 창은 바뀐 PATH를 모른다)

확인:
```
ffmpeg -version
```

### 3) 인텔 그래픽 드라이버

하드웨어 디코딩(Quick Sync)을 켜는 데 필요하다. 이게 없으면 영상 푸는 데만
CPU를 **3~6배** 더 쓴다.

[인텔 드라이버 지원 도우미](https://www.intel.co.kr/content/www/kr/ko/support/detect.html)
를 받아 실행하고, 그래픽 드라이버를 최신으로 올린다.

### 4) 저장소와 파이썬 환경

```
cd C:\
git clone <저장소 주소> seatnow
cd seatnow
python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 5) 샘플 영상 넣기

`sample_raw/` 폴더를 만들고 카페 영상을 넣는다 (`ONBOARDING.md` §2 참조).
없으면 디코딩 벤치를 돌릴 원본이 없다.

---

## 2-B. Linux 길 (Ubuntu Server)

### 1) 설치 USB 만들기

1. [Ubuntu Server LTS](https://ubuntu.com/download/server) ISO를 받는다
2. [Rufus](https://rufus.ie/) (Windows) 또는 [balenaEtcher](https://etcher.balena.io/)
   를 받는다
3. USB(8GB 이상)를 꽂고, 프로그램에서 ISO와 USB를 골라 **굽기**
4. 미니PC에 USB를 꽂고 켜면서 `F2`·`F10`·`Del` 중 하나를 연타해 부팅 메뉴로 들어가
   USB를 고른다 (제조사마다 키가 다르다)
5. 설치 화면에서 기본값으로 진행하되, **OpenSSH server 설치**는 체크한다
   (나중에 원격으로 들어가려면 필요하다)

### 2) 필요한 것 설치

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git ffmpeg \
    intel-media-va-driver-non-free vainfo
```

`intel-media-va-driver-non-free` 가 Quick Sync를 켜는 드라이버다. **`-non-free`가
붙은 쪽을 설치해야 한다** — 이름만 비슷한 `intel-media-va-driver`는 기능이 빠져 있다.

확인:
```bash
vainfo
```
`VAProfileH264...` 같은 줄들이 나오면 된 것이다.

### 3) 저장소와 파이썬 환경

```bash
cd ~
git clone <저장소 주소> seatnow
cd seatnow
python3 -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements-edge.txt
```

> 리눅스 박스에서는 `requirements.txt` 가 아니라 **`requirements-edge.txt`** 를
> 쓴다. 노트북과 같은 버전을 고정하고(판정이 같은지 비교할 수 있어야 한다),
> torch 를 CPU 판으로 받는다 — 그냥 받으면 GPU가 없는 박스에 엔비디아용
> 파일로 2~3GB를 더 쓴다. `docs/edge-first-run.md` 1단계 참고.

### 4) 샘플 영상 넣기

`sample_raw/` 에 카페 영상을 넣는다 (`ONBOARDING.md` §2).

---

## 3. 검수와 측정 — 세 줄

Windows는 `venv\Scripts\python.exe`, Linux는 `./venv/bin/python` 으로 바꿔 읽는다.
아래는 Windows 기준이다.

```
venv\Scripts\python.exe -m edge.check_edge
venv\Scripts\python.exe -m edge.bench --frames sample_raw\cafe_sample_angle1.mov --label edge-box
venv\Scripts\python.exe -m edge.bench_decode --source sample_raw\cafe_sample_angle1.mov
```

**순서가 중요하다.** `edge/bench.py`를 먼저 돌려야 `edge/bench_decode.py`가 추론까지 합친
표를 낼 수 있다.

### `edge/check_edge.py` — 이 박스가 돌릴 수 있는가

항목마다 합격/불합격이 나오고, 불합격이면 무엇을 하면 되는지가 `→` 줄에 붙는다.

```
[합격] CPU 코어 수: 8개
[불합격] 하드웨어 디코딩: 하드웨어 디코딩 꺼짐 — 소프트웨어로 돌립니다. 시도한 것: qsv, vaapi
         → 그래픽 드라이버를 설치한다. Linux는 intel-media-va-driver, ...
```

**하드웨어 디코딩이 "꺼짐"으로 나오면 여기서 멈추고 드라이버부터 잡는다.**
꺼진 채로 측정하면 실제보다 훨씬 비관적인 숫자가 나와서, 살 수 있는 카메라를
못 산다고 잘못 판단하게 된다.

### `edge/bench.py` — 판정이 15초 안에 끝나는가

프로파일마다 `PASS`/`CONDITIONAL`/`FAIL` 이 나온다. 합격선은 **tick의 50% 이내
(7.5초)** 다. 여유 50%를 요구하는 이유는 24시간 도는 박스에 스트림 재연결·디코더
리셋·로그 정리가 언제든 끼어들기 때문이다.

`--backends pt` 만 나오고 `ov-fp32`·`ov-int8` 이 `skip` 이면 아직 익스포트를 안 한
것이다. `python -m edge.export` 를 먼저 돌리면 OpenVINO 백엔드까지 잰다 — **실제 배포는
OpenVINO로 하므로 이 숫자가 진짜 숫자다.**

`skip ... (고정 크기 N 로 익스포트된 모델이다)` 가 뜨면 `--static` 으로 뽑은
모델이라 그 크기 말고는 잴 수 없다는 뜻이다. 크기를 박아둔 모델은 **다른
크기로 요청받아도 오류 없이 자기 크기로 돌기 때문에**, 그대로 재면 표가
실제보다 빠른 쪽으로 거짓말을 한다. `python -m edge.export` 를 기본값(어떤
크기든 받는 모델)으로 다시 돌린다.

### `edge/bench_decode.py` — 어떤 카메라를 살 수 있는가

처음 돌릴 때는 벤치용 클립 6개를 만드느라 몇 분 걸린다. 만든 클립은
`results/edge/clips/` 에 남아 다음부터는 재사용된다.

---

## 4. 결과 읽는 법

### 디코딩 표

```
| 클립     | 디코딩 방식 | 점유 코어 | 전체 대비 | 판정 |
| 4mp_h265 | none        | 0.38      | 5%        | PASS |
| 4mp_h265 | d3d11va     | 0.07      | 1%        | PASS |
```

- **점유 코어** = 영상 1초를 푸는 데 드는 CPU 초. 24시간 내내 이만큼을 쓴다
- `none` 은 소프트웨어(하드웨어 안 씀), 나머지는 하드웨어 디코딩
- **`none` 줄과 하드웨어 줄의 차이가 곧 드라이버의 값어치다**

| 판정 | 뜻 |
|---|---|
| `PASS` | 전체 코어의 25% 이내. 나머지를 판정과 OS가 쓸 수 있다 |
| `CONDITIONAL` | 25~50%. 돌긴 하는데 여유가 없다 |
| `FAIL` | 50% 초과. **이 박스로 이 카메라는 못 쓴다** |

### 합산 표

디코딩과 추론을 더한 것이다. **최종 구매 근거는 이 표다.**

⚠️ **이 표는 근사치다.** 추론이 도는 동안 코어를 전부 쓴다고 가정했다. 카메라를
탈락시키는 근거로는 충분하지만, 3% 차이를 다투는 데는 못 쓴다.

### 카메라 고르는 법

1. 합산 표에서 `PASS` 인 것 중 **해상도가 가장 높은 것**을 고른다
2. 같은 해상도면 **H.265** 를 고른다 — 같은 화질에 대역폭을 덜 쓴다.
   단 하드웨어 디코딩이 꺼져 있으면 H.265는 H.264보다 40%쯤 비싸므로,
   드라이버를 못 잡는 박스라면 H.264로 간다
3. 고른 카메라가 **RTSP를 열어주는지 반드시 확인한다.** 하이크비전·다후아 계열은
   확정이고, 국내 저가 브랜드는 검증이 필요하다 (`plan.md` §2)

---

## 5. 자주 막히는 곳

| 증상 | 원인과 해결 |
|---|---|
| `ffmpeg를 찾지 못했다` | PATH에 안 잡혔다. 명령 프롬프트를 **새로 열어** 본다. 그래도 안 되면 2-A의 3번을 다시 한다 |
| 하드웨어 디코딩이 계속 "꺼짐" | 그래픽 드라이버 문제다. Linux는 `vainfo` 가 뭐라고 하는지 본다. Windows는 인텔 드라이버를 다시 설치한다. **가상머신 안에서는 원래 안 된다** |
| `qsv` 는 실패하고 `d3d11va` 는 되는데 괜찮나 | 괜찮다. 둘 다 하드웨어 디코딩이고, 자동으로 되는 쪽을 고른다. 성능 차이는 작다 |
| 클립 만드는 게 너무 느리다 | `--duration 10` 을 붙여 클립을 짧게 만든다. 측정값은 길이에 비례하지 않으므로 결과는 같다 |
| `edge/bench.py` 가 전부 `skip` | 모델 파일이 없다. `python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8n-pose.pt')"` 로 받는다 |
| 8MP 클립 만들 때 메모리가 모자란다 | `--clips 2mp_h264 2mp_h265 4mp_h264 4mp_h265` 로 8MP를 뺀다. 4MP까지만 봐도 대부분의 결정은 난다 |
| 테스트가 깨진다 | `python -m unittest discover tests` 로 확인. numpy가 2.x로 올라갔을 가능성이 크다 (`pip install "numpy<2"`) |

---

## 6. 다음 단계

측정이 끝나 카메라를 고르고 나면:

1. 카메라 구매 (**RTSP 개방 확인 필수**)
2. `plan.md` T8 — 라이브 스트림 읽기. 지금 코드는 **파일 전용**이라 카메라를 꽂아도
   안 돈다. tick마다 영상의 특정 시각으로 되감는 구조인데, 라이브는 되감기가 안 된다
3. `plan.md` T10 — 24시간 무인 운영 (자동 재시작, 로그 정리, **영상 미저장 기본값**)

2번과 3번은 **카메라가 있어야 검증되므로** 지금 만들지 않았다.
