# 엣지 박스 설치·검수 절차

> 작성: 2026-08-31
> 관련: `docs/superpowers/specs/2026-08-31-edge-decode-bench-design.md`, `plan.md` T6·T9

> ⚠️ **엣지 박스 첫 실행이라면 `docs/edge-first-run.md` 를 먼저 본다.**
> 이 문서는 "어떤 카메라를 살 수 있나"까지 가는 긴 절차라 첫날 순서와 다르다.
> **다만 §0의 확인 목록은 첫날에도 그대로 쓴다.**
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

## 0. 박스에서 확인할 것 — 순서대로 (2026-09-03)

**이 표만 위에서 아래로 통과하면 나머지는 절차일 뿐이다.**
개발 노트북에서 미리 다 돌려보고 만든 목록이라, 각 줄의 "나와야 하는 값"은
추측이 아니라 노트북에서 실제로 나온 값이다.

앞의 넷은 설치가 제대로 됐는지, 뒤의 넷은 **판정이 노트북과 같고 시간 안에
끝나는지**를 본다. 순서가 중요하다 — 위가 틀린 채로 아래를 재면 그 숫자는
버려야 한다.

> **1~5번은 설치 직후 바로 볼 수 있다. 6~8번은 아니다** — 6번은
> `edge.export` 를 한 번 돌린 뒤에, 7·8번은 영상을 한 번 실행한 뒤에 생기는
> 것을 본다 (`docs/edge-first-run.md` 4·5단계). 명령의 `results/edge_angle1`
> 은 그때 만들어지는 폴더 이름이다.

| | 확인 | 명령 | 나와야 하는 값 | 아니면 |
|---|---|---|---|---|
| 1 | 파이썬 | `python3 --version` | **3.12** (24.04 기본) | 24.04 라면 그대로 진행한다 — 고정한 34개가 전부 설치되는 것을 확인했다. **22.04(3.10)라면 이 목록 그대로는 안 된다** (`contourpy`·`matplotlib` 에 3.10용 파일이 없어 소스 빌드로 넘어가고 컴파일러가 없어 멈춘다). 우분투에 3.11을 따로 넣지는 말 것 — 22.04 저장소의 것은 rc 다 |
| 2 | 그래픽 권한 | `groups` | 목록에 **`render`** | `sudo usermod -aG render,video $USER` 후 **재로그인**. 이게 없으면 드라이버를 깔아도 하드웨어 디코딩이 안 켜진다 |
| 3 | 영상 디코딩 | `vainfo` | `VAProfileH264...` 줄들 | `intel-media-va-driver-**non-free**` 를 깔았는지 확인 (2-B의 2번) |
| 4 | torch 종류 | `./venv/bin/python -c "import torch; print(torch.__version__)"` | **`2.2.2+cpu`** | `+cpu` 가 없으면 엔비디아용을 받은 것이다. `requirements-edge.txt` 로 다시 깐다 |
| 5 | **OpenVINO가 내장 그래픽을 보는가** (저장소 없이 먼저 볼 수 있다 → `docs/clinfo.md`) | `./venv/bin/python -c "import openvino as ov; print(ov.Core().available_devices)"` | **`['CPU', 'GPU']`** | ⬅️ **오늘 가장 중요한 줄.** 아래 설명 |
| 6 | 익스포트가 크기에 안 묶였는가 | `grep -A7 "^args:" yolov8n_openvino_model/metadata.yaml` | **`dynamic: true`** | `false` 면 크기를 박은 모델이다. `./venv/bin/python -m edge.export --precision fp32 --imgsz 1280` 로 다시 뽑는다 |
| 7 | 판정이 노트북과 같은가 | `./venv/bin/python -m checks.score_answers results/edge_angle1 --answers results/angle1_layout/angle_answer.md` | **`accepted=70 wrong=2`** | 속도를 논하기 전에 이것부터. 보통 라이브러리 버전 문제다 |
| 8 | 판단 한 번에 몇 초인가 | 실행 로그의 `inference=` (**두 번째 줄부터**) | **7500ms 이하** | `docs/edge-first-run.md` 5단계의 카드 |

### 5번이 왜 가장 중요한가

노트북에서 재보니 **OpenVINO는 가만히 두면 알아서 내장 그래픽으로 넘어간다**
(실제 실행 장치를 물어보니 `GPU.0` 이었다). 그리고 그것이 있느냐 없느냐가
전부를 가른다 — 코어 개수가 아니다.

| 노트북 실측 (같은 영상, 같은 판정) | 틱당 추론 |
|---|---:|
| OpenVINO, 내장 그래픽 사용 | **1.13 ~ 1.20초** |
| OpenVINO, 코어 2개로 묶음 (그래픽은 사용) | 1.29 ~ 1.59초 |
| ⚠️ OpenVINO, **CPU만** | 6.0 ~ 7.6초 |
| ⚠️ OpenVINO, **CPU만 + 코어 2개** | 9.5 ~ 11.0초 |
| (참고) 파이토치, 코어 2개 | 12.2 ~ 13.9초 |

### `['CPU']` 만 나올 때 — 드라이버가 두 종류라서 그렇다 (2026-09-03 확인)

**HD 530이 낡아서가 아니다.** OpenVINO의 GPU 지원 범위는 Gen9~Gen12이고
HD 530(Skylake, Gen9)은 그 안에 든다. 문제는 **드라이버가 두 종류**인데
예전 설치 목록에 한쪽만 있었다는 것이다.

| 드라이버 | 하는 일 | 확인 |
|---|---|---|
| `intel-media-va-driver-non-free` | **영상을 푼다** (디코딩) | `vainfo` |
| `intel-opencl-icd` | **내장 그래픽에 AI 연산을 시킨다** | `clinfo` |

둘 다 2-B의 설치 목록에 들어 있다. 이 계산용 드라이버는 **OpenVINO 패키지
안에 없다** — 그래서 pip 로 openvino 를 깔아도 `GPU` 가 안 보일 수 있다.

`clinfo` 에 HD Graphics 530 이 안 보이면 **우분투 버전을 본다.**
인텔이 `24.35.30872.22` 부터 **Gen8·Gen9·Gen11을 별도 패키지로 분리**했기
때문이다.

| 우분투 | `intel-opencl-icd` 버전 | HD 530(Gen9) |
|---|---|---|
| 22.04 LTS | 22.14 | ✅ 된다 |
| 24.04 LTS | 23.43 | ✅ 된다 |
| 25.10 이상 | 25.31 이상 | ❌ **`sudo apt install intel-opencl-icd-legacy`** 를 대신 깐다 |

> ⚠️ 인텔이 배포하는 `.deb` 파일 이름은 `intel-opencl-icd-**legacy1**` 인데,
> **우분투 패키지 이름에는 `1` 이 없다** (`intel-opencl-icd-legacy`).
> 없는 이름을 넣으면 `apt install` 이 통째로 실패해서 같은 줄의 다른 것도
> 안 깔린다. 헷갈리면 `apt search intel-opencl-icd` 로 먼저 확인한다.
> 배포판에 아예 없으면 <https://github.com/intel/compute-runtime/releases>
> 의 legacy1 릴리스에서 `.deb` 를 받는다.

그렇게 해도 `GPU` 가 안 보이면 **CPU만으로 가야 한다.** 위 표의 아래쪽이고
박스 코어는 노트북보다 2배쯤 느리므로, `docs/edge-first-run.md` 5단계의
①(3프레임)·④(크롭 개수)를 같이 써야 7.5초에 든다.

### 시간을 잴 때 주의할 것 두 가지

**① 첫 줄은 버린다.** OpenVINO는 처음 한 번 모델을 그 기계에 맞게 컴파일한다.
노트북에서 첫 틱이 **30~83초**였고 두 번째부터 1초대였다. 박스에서는 더
걸릴 수 있다. 고장이 아니다.

**② `inference=` 는 영상 푸는 시간을 안 센다.** 실제로 24시간 도는 박스는
디코딩도 같이 한다. 맨 끝에 찍히는 `Completed 6 samples in Xs` 를 6으로
나눈 값이 **진짜 틱 시간**이다. 두 숫자를 같이 적어둔다.

### `check_edge` 의 "합격"을 곧이곧대로 믿지 않는다

- **"CPU 코어 수: 4개 합격"은 논리 코어다.** i3-6100T의 실제 연산 코어는 2개다
- **RAM 4GB도 "합격"이 뜬다** (합격선이 정확히 4.0GB라서). 틱 시간이 들쭉날쭉하면
  (어떤 틱은 4초, 어떤 틱은 20초) CPU가 아니라 **메모리가 모자란 것**이다

---

## 1. OS를 무엇으로 할 것인가

> **지금 박스는 이미 리눅스로 깔았다.** 그러니 2-A(Windows)는 건너뛰고
> **2-B로 간다.** 아래는 왜 그렇게 정했는지의 기록이다.

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
git clone https://github.com/fifawithkupa/dwnc_cafe_cctv.git seatnow
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
    intel-media-va-driver-non-free vainfo \
    intel-opencl-icd clinfo
```

> **우분투 24.04 기준으로 여덟 개 전부 있는 것을 확인했다** (2026-09-03).
> `intel-media-va-driver-non-free` 24.1.0 은 multiverse, `intel-opencl-icd`
> 23.43 과 `clinfo` 는 universe 에 있고, 우분투 기본 설치는 두 저장소를 다
> 켜 둔다. `Unable to locate package` 가 뜨면 저장소가 꺼진 것이니
> `sudo add-apt-repository multiverse universe && sudo apt update` 후 다시.
>
> 우분투 **25.10 이상**이라면 마지막 줄을 `intel-opencl-icd-legacy clinfo` 로
> 바꾼다. 그 버전부터 기본 패키지가 HD 530(Gen9)을 빼고 나오기 때문이다
> (§0의 표). 22.04·24.04 는 위 그대로면 된다.

`intel-media-va-driver-non-free` 가 Quick Sync를 켜는 드라이버다. **`-non-free`가
붙은 쪽을 설치해야 한다** — 이름만 비슷한 `intel-media-va-driver`는 기능이 빠져 있다.

**드라이버는 두 종류이고 둘 다 필요하다.** 앞의 것은 **영상을 푸는** 드라이버고,
`intel-opencl-icd` 는 **내장 그래픽에 AI 연산을 시키는** 드라이버다.
후자가 없으면 추론이 전부 CPU로 떨어져 **5~6배** 느려진다 (§0의 5번).

`/dev/dri` 에 들어갈 권한도 필요하다. **이 명령 뒤에는 로그아웃했다가 다시
로그인해야 적용된다.**

```bash
sudo usermod -aG render,video $USER
```

확인:
```bash
vainfo                          # VAProfileH264... 줄들 = 영상 디코딩 OK
clinfo | grep -i "Device Name"  # HD Graphics 530 = AI 연산 OK
```

### 3) 저장소와 파이썬 환경

```bash
cd ~
git clone https://github.com/fifawithkupa/dwnc_cafe_cctv.git seatnow
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

리눅스 박스(지금 우리 것) 기준이다.

```bash
./venv/bin/python -m edge.check_edge
./venv/bin/python -m edge.bench --frames sample_raw/cafe_sample_angle1.mov --label edge-box
./venv/bin/python -m edge.bench_decode --source sample_raw/cafe_sample_angle1.mov
```

Windows에서 돌린다면 `./venv/bin/python` 을 `venv\Scripts\python.exe` 로,
경로의 `/` 를 `\` 로 바꾼다.

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

`p95` 가 몇 초로 크게 나오는 줄이 있어도 놀라지 않는다 — **OpenVINO의 첫
추론은 모델 컴파일이라 원래 오래 걸린다.** 판단은 `median` 으로 한다.

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
**`백엔드` 칸을 반드시 본다** — 실제 배포는 `ov-fp32`(OpenVINO)로 하므로
`pt` 줄이 FAIL 이어도 그것만으로 카메라를 탈락시키지 않는다.

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
| `apt: Unable to locate package intel-opencl-icd-legacy1` | 우분투에는 그 이름이 없다 (`1` 없는 `intel-opencl-icd-legacy` 이고, 그것도 25.10부터다). §0의 표대로 고른다. **한 이름만 틀려도 그 apt 줄 전체가 실패해서 ffmpeg 도 안 깔린다** |
| `clinfo` 가 장치를 하나도 못 찾는다 | ① `groups` 에 `render` 가 있는지 (재로그인했는지) ② 우분투 25.10 이상이면 `intel-opencl-icd-legacy` 로 바꿔 깐다 |
| `ModuleNotFoundError: openvino` | `requirements-edge.txt` 로 안 깔았을 때 난다. `pip install -r requirements-edge.txt` |
| OpenVINO 를 켰더니 빨라졌는데 채점이 70이 아니다 | 크기를 박은 모델로 뽑힌 것이다. §0의 6번을 확인하고 `python -m edge.export --precision fp32 --imgsz 1280` 로 다시 뽑는다 |
| `available_devices` 에 `GPU` 가 없다 | HD 530이 이 OpenVINO 버전에서 빠진 것일 수 있다. §0의 "5번이 왜 가장 중요한가" 참고 |
| 첫 판단이 1분 넘게 걸린다 | 모델 컴파일이다. 두 번째 줄부터 본다 (§0) |

---

## 6. 다음 단계

측정이 끝나 카메라를 고르고 나면:

1. 카메라 구매 (**RTSP 개방 확인 필수**)
2. `plan.md` T8 — 라이브 스트림 읽기. 지금 코드는 **파일 전용**이라 카메라를 꽂아도
   안 돈다. tick마다 영상의 특정 시각으로 되감는 구조인데, 라이브는 되감기가 안 된다
3. `plan.md` T10 — 24시간 무인 운영 (자동 재시작, 로그 정리, **영상 미저장 기본값**)

2번과 3번은 **카메라가 있어야 검증되므로** 지금 만들지 않았다.
