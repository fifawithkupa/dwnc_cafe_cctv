# 엣지 박스 — 위에서 아래로 그대로 치는 순서

> 갱신: 2026-09-03
> 박스: **Dell OptiPlex 7040 Micro / i3-6100T / RAM 4GB / Ubuntu 24.04**
>
> **이 문서는 순서다.** 박스 앞에 앉아서 1번부터 8번까지 차례로 친다.
> 각 단계의 **`→`** 가 나와야 다음으로 간다.
>
> **막히면 "몇 번에서 이렇게 나왔다"고 말해줘.** 그 단계의
> **"막혔을 때"** 칸에 해결 방법을 바로 여기 추가한다.
> 다음 매장에 설치할 때 그대로 다시 쓰는 파일이라, 겪은 것은 전부 남긴다.

> ### 🖥️ 박스에는 노트북에서 SSH로 붙어서 작업한다
>
> 박스에 꽂은 모니터 화면(텍스트 콘솔)은 **한글 글꼴이 없어서 한글이 전부
> 다이아몬드(◆)로 나온다.** 코드 문제가 아니라 그 화면의 한계다.
> 노트북 PowerShell 에서 이렇게 붙으면 한글도 보이고 복사·붙여넣기도 된다:
>
> ```powershell
> ssh seatnow@192.168.0.42
> ```
>
> (`seatnow@192.168.0.42` 는 3단계 ①에서 확인한 값). 비밀번호를 넣으면
> 그때부터 치는 명령은 전부 박스에서 실행된다. 모니터는 IP 확인할 때만 쓴다.

---

## 0단계 — 이미 끝난 것 ✅

박스에서 이걸 이미 했고 통과했다.

```bash
sudo apt update
sudo apt install -y intel-opencl-icd clinfo python3-venv
sudo usermod -aG render,video $USER      # 그리고 재로그인
```

```bash
python3 -m venv /tmp/ovcheck
/tmp/ovcheck/bin/pip install -q openvino
/tmp/ovcheck/bin/python -c "import openvino as ov; print(ov.Core().available_devices)"
```

→ **`['CPU', 'GPU']`** ✅ **받았다.**

**이게 이 프로젝트에서 가장 중요한 한 줄이었다.** 내장 그래픽(HD 530)이 AI
연산에 잡혔다는 뜻이고, 판단 한 번이 1초대로 끝날 가능성이 크다는 뜻이다.
못 잡았으면 10초대라서 박스를 바꿔야 했다.

확인용 임시 환경은 지워도 된다: `rm -rf /tmp/ovcheck`

---

## 1단계 — 나머지 설치 (박스에서)

```bash
sudo apt install -y python3-pip git ffmpeg intel-media-va-driver-non-free vainfo
```

→ 오류 없이 끝나면 된다.

<details>
<summary><b>막혔을 때</b></summary>

- **`Unable to locate package ...`** — 저장소가 꺼진 것이다.
  `sudo add-apt-repository multiverse universe && sudo apt update` 후 다시.
- ⚠️ **`-non-free` 가 붙은 쪽이어야 한다.** 이름만 비슷한
  `intel-media-va-driver` 는 기능이 빠져 있다. 이게 영상을 푸는(디코딩)
  드라이버이고, 없으면 영상 푸는 데만 CPU를 3~6배 쓴다.
- ⚠️ **이름 하나만 틀려도 그 줄 전체가 실패한다.** ffmpeg 도 같이 안 깔린다.

</details>

---

## 2단계 — 저장소 받고 파이썬 준비 (박스에서)

```bash
cd ~
git clone https://github.com/fifawithkupa/dwnc_cafe_cctv.git seatnow
cd seatnow
python3 -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements-edge.txt
```

10분쯤 걸린다. 끝나면 확인:

```bash
./venv/bin/python -c "import torch; print(torch.__version__)"
```

→ **`2.2.2+cpu`** — 끝에 **`+cpu`** 가 붙어야 한다.

> ⚠️ **박스가 코드를 받아가는 곳은 `main` 브랜치다.**
> 노트북에서 한 작업은 `JINOI/main` 에 쌓이므로, **박스에 반영하려면 노트북에서
> 이걸 한 번 밀어야 한다:**
>
> ```bash
> git push origin JINOI/main:main
> ```
>
> 2026-09-03에 이걸 안 해서 박스가 90커밋 뒤처진 코드를 받았고,
> `requirements-edge.txt` 부터 없어서 2단계에서 막혔다.
> **박스로 뭔가 보내기 전에는 항상 이 줄을 먼저 친다.**

<details>
<summary><b>막혔을 때</b></summary>

- **`Could not open requirements file: ... 'requirements-edge.txt'`**
  — 박스가 옛날 코드를 받은 것이다. **노트북에서 `git push origin JINOI/main:main`
  을 먼저 한 다음**, 박스에서 (저장소를 지우고 다시 받을 필요 없다):
  ```bash
  cd ~/seatnow
  git pull
  ls requirements-edge.txt        # 보이면 된 것
  ```
  그다음 `./venv/bin/python -m pip install -r requirements-edge.txt` 로 이어서
  한다. venv 는 이미 만들었으니 다시 안 만들어도 된다.
  같은 원인으로 4·5·7·8단계에서 `No module named ...` 나 `파일이 없다` 가 날 수
  있는데 해결은 똑같다.
- **`+cpu` 가 없다 / 1GB 넘게 받는다** — `requirements.txt` 로 깔고 있는
  것이다. 파일 이름이 **`requirements-edge.txt`** 인지 확인하고 다시.
  (그냥 받으면 리눅스에서는 엔비디아 GPU용이 내려와 2~3GB를 더 먹는다)
- **`python3 --version` 이 3.10 이다** (우분투 22.04) — 이 목록 그대로는
  안 된다. `contourpy`·`matplotlib` 에 3.10용 파일이 없어 소스 빌드로
  넘어가고 컴파일러가 없어 멈춘다. **우분투에 3.11을 따로 넣지는 말 것**
  (22.04 저장소의 것은 rc 다). 24.04(3.12)면 그대로 간다.

</details>

---

## 3단계 — 영상 옮기기 ⚠️ **이건 노트북에서 친다**

**① 박스에서** 받을 폴더를 만들고 주소를 확인한다.

```bash
mkdir -p ~/seatnow/sample_raw
echo "$(whoami)@$(hostname -I | awk '{print $1}')"
```

→ 예: `seatnow@192.168.0.42` — 이걸 그대로 복사해서 ②에 붙인다.

> ⚠️ **`mkdir` 을 건너뛰면 안 된다.** `sample_raw/` 는 용량·개인정보 때문에
> 저장소에서 제외돼 있어서 `git clone` 해도 안 생긴다. 그리고 `scp` 는
> **없는 폴더를 만들어주지 않는다.**

**② 노트북에서** (윈도우면 PowerShell) 영상을 보낸다.

```powershell
scp "C:\Users\jin06\orca\workspaces\dwnc_cafe_cctv\main\sample_raw\cafe_sample_angle1.mov" seatnow@192.168.0.42:~/seatnow/sample_raw/
```

박스 비밀번호를 물어보면 입력한다. 146MB라 공유기 안에서는 1~2분.

→ 박스에서 `ls ~/seatnow/sample_raw/` 했을 때 파일이 보이면 된다.

USB로 옮겨도 된다. **AI 모델은 안 옮겨도 된다** — 5단계 ①에서 받는다.

<details>
<summary><b>막혔을 때</b></summary>

- **`Connection refused`** — 박스에 접속 서버가 안 깔렸다. 박스에서:
  ```bash
  sudo apt install -y openssh-server
  sudo systemctl enable --now ssh
  ```
  그리고 노트북에서 `scp` 를 다시 친다.
- **`hostname -I` 가 주소를 여러 개 뱉는다** — `192.168.` 이나 `10.` 으로
  시작하는 것을 쓴다 (같은 공유기 안 주소).
- **`dest open "seatnow/sample_raw/": Failure`** 또는
  **`No such file or directory`** — 받을 폴더가 없다. ①의 `mkdir` 을
  안 한 것이다. 박스에서 `mkdir -p ~/seatnow/sample_raw` 하고 다시.
- **`mkdir` 을 했는데도 같은 오류** — `~` 를 못 풀고 있는 것이다.
  전체 경로로 친다 (`seatnow` 자리에 박스 사용자 이름):
  `... seatnow@192.168.0.42:/home/seatnow/seatnow/sample_raw/`
- **아무리 해도 안 붙는다** — 노트북과 박스가 **같은 공유기**에 붙어 있는지
  본다. 그래도 안 되면 **USB로 옮긴다.** 결과는 똑같다.

</details>

---

## 4단계 — 검수 (박스에서)

```bash
cd ~/seatnow
./venv/bin/python -m edge.check_edge
```

→ **`[합격] 하드웨어 디코딩`** 이 나와야 한다.

<details>
<summary><b>막혔을 때</b></summary>

- **하드웨어 디코딩 "꺼짐"** — 여기서 멈추고 잡는다. 꺼진 채로 재면 실제보다
  훨씬 나쁜 숫자가 나와서 멀쩡한 박스를 못 쓴다고 잘못 판단하게 된다. 순서대로:
  ① `groups` 에 `render` 가 있는가 (없으면 → 부록 B 재로그인)
  ② `vainfo` 가 `VAProfileH264...` 를 뱉는가
  ③ 1단계의 `-non-free` 드라이버를 깔았는가
  ④ **가상머신 안에서는 원래 안 된다**
- **한글이 전부 다이아몬드(◆)로 나온다** — **코드 문제가 아니다.** 박스에
  꽂은 모니터 화면(텍스트 콘솔)에 한글 글꼴이 없는 것이다. 확인: 같은 화면에서
  `echo "한글"` 을 쳐도 깨지면 화면 문제다. **노트북에서 SSH로 붙는다**
  (맨 위 상자). 우분투 24.04 에서 이 도구들의 출력이 UTF-8 한글인 것은
  `LANG=C` 인 경우까지 확인했다 (2026-09-03) — 코드를 영어로 바꿀 이유가 없다.
- **메모리 3.7GB 불합격** — **그냥 진행한다.** 4GB를 꽂아도 운영체제가
  보고하는 값은 3.6~3.8GB다 (커널·내장 그래픽이 일부를 먼저 가져간다).
  2026-09-03 박스에서 실제로 3.7GB가 찍혔고, 검수 도구가 정확히 4.0으로
  자르던 것을 고쳐서 이제는 합격으로 나온다 (`git pull` 하면 반영).
  다만 **빠듯한 건 사실이다** — 7단계에서 틱 시간이 들쭉날쭉하면(4초↔20초)
  메모리 부족이니 그때 8GB로 늘린다.
- **`ffmpeg를 찾지 못했다`** — `which ffmpeg` 로 확인. 없으면 1단계를 다시.
- **다른 항목이 "합격"이어도 곧이곧대로 믿지 않는다** — 부록 A ③

</details>

---

## 5단계 — 모델 변환

**① AI 모델 파일을 먼저 받는다** (13MB짜리 2개, 1분).

```bash
./venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8n-pose.pt')"
```

→ `yolov8n.pt` 와 `yolov8n-pose.pt` 두 파일이 생긴다.

> 모델 파일은 용량 때문에 저장소에 안 들어 있다. 원래 7단계에서 자동으로
> 받아지는데, **변환은 그보다 먼저라 여기서 직접 받아야 한다.**

**② 변환한다.**

```bash
./venv/bin/python -m edge.export --precision fp32 --imgsz 1280
```

→ 몇 분 걸리고, 폴더 2개가 생긴다:
`yolov8n_openvino_model/` · `yolov8n-pose_openvino_model/`

<details>
<summary><b>막혔을 때</b></summary>

- **`FileNotFoundError: detect weights not found: .../yolov8n.pt`**
  — ①을 안 한 것이다. 위 ①을 치고 ②를 다시. (2026-09-03 박스에서 겪음)
- **`skip (exists)` 만 찍고 끝난다** — 같은 이름의 폴더가 이미 있으면
  건너뛴다. 다시 뽑으려면 `--overwrite` 를 붙인다.
- **`--precision fp32` 를 빼면 안 되나** — 빼면 기본값이 int8 까지 뽑아서
  훨씬 오래 걸린다. 지금 필요한 건 fp32 뿐이다.

</details>

---

## 6단계 — 변환이 제대로 됐는지

```bash
grep -A7 "^args:" yolov8n_openvino_model/metadata.yaml | grep dynamic
grep -A7 "^args:" yolov8n-pose_openvino_model/metadata.yaml | grep dynamic
```

→ 둘 다 **`dynamic: true`**

<details>
<summary><b>막혔을 때</b></summary>

- **`dynamic: false`** — 크기를 박아둔 모델이다. 5단계를 `--overwrite` 를
  붙여 다시 돌린다.
- **왜 이걸 보나** — 판단 한 번에 검출 모델이 **두 가지 크기로** 불린다
  (화면 전체 1280, 테이블 크롭 960). 크기를 박아둔 모델은 **틀린 크기로
  요청받아도 오류를 안 내고 자기 크기로 그냥 돈다.** 그러면 "OpenVINO 켜니
  3배 빨라졌다"가 실은 "1280 대신 640으로 봤다"가 된다.
  **빠른데 틀린 결과**가 나오고 원인은 OpenVINO처럼 보인다.

</details>

---

## 7단계 — 진짜 실행 ⬅️ **오늘의 목표**

```bash
./venv/bin/python -m engine.seatnow sample_raw/cafe_sample_angle1.mov \
  --layout layouts/cafe_angle1.json \
  --det-model yolov8n_openvino_model --pose-model yolov8n-pose_openvino_model \
  --log results/edge_ov/log.jsonl --log-detections --no-video
```

→ 이런 줄이 6개 나오고, 마지막에 `Completed 6 samples in ...` 이 찍힌다.

```
[  0.0s] tables=12 occupied=3 empty=8 ... inference=14748ms
[ 15.0s] tables=12 occupied=3 empty=6 ... inference=643ms
```

### 놀라지 말 것 — 둘 다 정상이다

- **첫 줄이 수십 초** 나온다. OpenVINO가 모델을 이 박스에 맞게 처음 한 번
  굽는 시간이다. **두 번째 줄부터가 진짜다.**
- 로그에 **`inference on (CPU)`** 라고 찍힌다. **CPU로 돈다는 뜻이 아니다.**
  괄호는 "굽는 동안 임시로 CPU를 쓰는 중"이라는 표시고, 다 구워지면 조용히
  내장 그래픽으로 넘어간다. 근거는 부록 A ①.

<details>
<summary><b>막혔을 때</b></summary>

- **`ModuleNotFoundError: openvino`** — 2단계를 `requirements-edge.txt` 로
  안 깐 것이다.
- **틱 시간이 들쭉날쭉하다** (어떤 틱은 4초, 어떤 틱은 20초) — CPU가 아니라
  **메모리가 모자란 것**이다 (스왑). `free -h` 로 확인. RAM 8GB로 올리는 게
  이 로드맵에서 가장 싼 위험 제거다.
- **7500ms를 넘는다** — `docs/edge-first-run.md` 5단계의 속도 카드. 다만
  내장 그래픽이 잡힌 지금은 그럴 가능성이 낮다.

</details>

---

## 8단계 — 채점 (**속도보다 이게 먼저다**)

```bash
./venv/bin/python -m checks.score_answers results/edge_ov \
  --answers results/angle1_layout/angle_answer.md --title "엣지 박스"
```

→ 마지막 줄에 **`accepted=70`** · **`wrong=2`**

```
scored=72 accepted=70 (exact=58 delayed=3 occluded=9) | wrong=2 held=0 | evidence missed=10 imagined=1
```

노트북과 **똑같은 값**이어야 한다. 장비가 달라도 판정은 같아야 한다.

<details>
<summary><b>막혔을 때</b></summary>

- **빨라졌는데 70이 아니다** — 6단계를 다시 본다. 크기를 박은 모델일 때 나는
  증상이다. **속도를 논하기 전에 이것부터 잡는다.**
- **정답지를 못 찾는다** — `--answers` 경로를 확인한다. 그 파일은 저장소에
  들어 있으므로 따로 복사할 필요가 없다.

</details>

---

## ✅ 박스에서 실제로 나온 값 (2026-09-03) — 다음 박스의 기준선

이 순서를 OptiPlex 7040 Micro / i3-6100T / RAM 4GB / Ubuntu 24.04 에서
끝까지 돌린 결과다. **다음에 설치할 때 이 숫자와 비교하면 된다.**

| | 박스 | (참고) 개발 노트북 |
|---|---:|---:|
| 첫 틱 — 버린다 | 42.3초 | 14.7초 |
| 틱당 추론 (2번째부터) | **4.8초** | 0.62초 |
| **진짜 틱 시간** | **5.6초** | 1.07초 |
| 합격선 7.5초 대비 | ✅ 여유 1.9초 (25%) | ✅ |
| 틱 편차 | 2% (4751~4871ms) | — |
| 메모리 | 3.7GB 로 충분했다 (스왑 없음) | — |

```
scored=72 accepted=70 (exact=58 delayed=3 occluded=9) | wrong=2 held=0 | evidence missed=10 imagined=1
```

**노트북과 한 글자도 다르지 않다.** 여섯 틱의 좌석 수도 전부 일치했다
(3/8 · 3/6 · 3/8 · 3/7 · 2/6 · 3/7).

> 박스가 노트북보다 7.8배 느리지만 **HD 530이 노트북 그래픽보다 8배쯤 약한
> 칩이라 정상이다.** 내장 그래픽을 제대로 쓰고 있을 때 나올 값이다.

---

## 끝나면 보내줄 것

1. **7단계가 뱉은 줄 6개 전부** + `Completed 6 samples in Xs` 줄
2. **8단계의 마지막 한 줄**

그거면 합격인지 계산한다. **합격선은 7.5초**(판단 주기 15초의 절반)다.

> 진짜 틱 시간 = **`(Completed 의 X − 첫 틱 inference) ÷ 5`**
> 그냥 6으로 나누면 안 된다 — 이유는 부록 A ②.

---

# 부록

## A. 왜 그렇게 하나 (실측 근거)

### ① 로그의 `(CPU)` 를 실패로 읽지 말 것

내장 그래픽이 멀쩡히 잡힌 개발 노트북에서 위 7단계 명령을 그대로 돌려 비교했다.

| 실행 | 로그에 찍힌 장치 | 2번째 틱부터 |
|---|---|---:|
| 7단계 명령 그대로 (자동) | `(CPU)` | **619 ~ 643ms** |
| `--device intel:GPU` 를 붙임 | `GPU.0` | **644ms** |
| `--device intel:CPU` 를 붙임 (진짜 CPU) | `CPU` | **1964 ~ 2144ms** |

**자동과 그래픽 지정이 같은 속도이고 진짜 CPU는 3배 느리다** → 자동으로 두면
내장 그래픽을 쓴다. `(CPU)` 만 보고 "드라이버가 안 잡혔다"고 판단하면
멀쩡한 박스를 버리게 된다.

눈으로 꼭 확인하고 싶으면 2틱만 이렇게 돌린다:

```bash
./venv/bin/python -m engine.seatnow sample_raw/cafe_sample_angle1.mov \
  --layout layouts/cafe_angle1.json \
  --det-model yolov8n_openvino_model --pose-model yolov8n-pose_openvino_model \
  --device intel:GPU --max-samples 2 --no-video
```

`... inference on GPU.0...` 가 나오면 확실하다 (맨 처음 한 줄은 `--device` 를
줘도 `(CPU)` 다. **뒤의 두 줄**을 본다). 확인이 끝나면 `--device` 는 도로 뺀다.

### ② 틱 시간을 6으로 나누면 안 된다

`Completed 6 samples in Xs` 의 X 안에는 모델 굽는 첫 틱이 통째로 있다.

```
[  0.0s] ... inference=14748ms      ← 굽는 시간. 버린다
[ 15.0s] ... inference=643ms
Completed 6 samples in 20.1s
```

| 계산 | 값 | |
|---|---:|---|
| ❌ `20.1 ÷ 6` | 3.35초 | 첫 틱을 나머지 5번에 뒤집어씌운 값 |
| ✅ `(20.1 − 14.75) ÷ 5` | **1.07초** | 진짜 틱 시간 |

`inference=` 는 **영상 푸는 시간을 안 센다.** 24시간 도는 박스는 디코딩도
같이 하므로 **두 숫자를 다 적어둔다.** 위 예에서 1.07초 대 0.62초,
차이 0.45초가 영상을 푸는 값이다.

### ③ `check_edge` 의 "합격"을 곧이곧대로 믿지 않는다

- **"CPU 코어 수: 4개 합격"은 논리 코어다.** i3-6100T의 실제 연산 코어는 2개다
  (하이퍼스레딩은 코어 2개 몫을 못 한다)
- **RAM 4GB 박스는 "빠듯하다" 경고와 함께 합격이 뜬다** (실제 보고값 3.7GB)
- **openvino 는 아예 안 본다.** 확인하는 패키지가 numpy·cv2·torch·ultralytics
  넷뿐이라 openvino 가 없어도 "전부 합격"이 뜬다
- **이 박스의 진짜 판정은 검수 화면이 아니라 7·8단계의 실측 숫자다**

### ④ 드라이버가 두 종류다

| 드라이버 | 하는 일 | 확인 |
|---|---|---|
| `intel-media-va-driver-non-free` | **영상을 푼다** (디코딩) | `vainfo` |
| `intel-opencl-icd` | **내장 그래픽에 AI 연산을 시킨다** | `clinfo` |

후자는 **OpenVINO 패키지 안에 없다.** pip 로 openvino 를 깔아도 `GPU` 가
안 보일 수 있는 이유다.

`clinfo` 에 HD Graphics 530 이 안 보이면 우분투 버전을 본다. 인텔이
`24.35.30872.22` 부터 Gen8·Gen9·Gen11을 별도 패키지로 분리했다.

| 우분투 | `intel-opencl-icd` | HD 530(Gen9) |
|---|---|---|
| 22.04 LTS | 22.14 | ✅ |
| 24.04 LTS | 23.43 | ✅ (**지금 박스**) |
| 25.10 이상 | 25.31 이상 | ❌ `intel-opencl-icd-legacy` 를 대신 |

> 인텔 `.deb` 파일 이름은 `intel-opencl-icd-legacy1` 이지만 **우분투 패키지
> 이름에는 `1` 이 없다** (`intel-opencl-icd-legacy`). 없는 이름을 넣으면
> `apt install` 이 통째로 실패한다. 헷갈리면 `apt search intel-opencl-icd` 로
> 먼저 확인. 배포판에 없으면
> <https://github.com/intel/compute-runtime/releases> 의 legacy1 릴리스에서 받는다.

## B. 재로그인하는 법

`sudo usermod -aG render,video $USER` 는 **지금 켜져 있는 접속 세션에는
반영되지 않는다.** 접속을 새로 해야 그룹이 붙는다.

| 상황 | 방법 |
|---|---|
| **아무 때나 (가장 확실)** | `sudo reboot` → 1분쯤 뒤 다시 접속 |
| 노트북에서 SSH로 붙어 있다 | `exit` 로 끊고 `ssh <사용자>@<박스IP>` 로 다시 |
| 모니터·키보드를 직접 꽂았다 | 화면 **오른쪽 위** 아이콘 → 전원 → **Log Out** → 다시 로그인 |

확인은 `groups`. `render` 와 `video` 가 보이면 된 것이다.

> **`newgrp render` 는 쓰지 않는다.** 그 창에서만 붙고 다른 창·자동 실행에는
> 안 붙어서, 되는 줄 알았다가 나중에 자동 실행(systemd)으로 돌릴 때 조용히
> 소프트웨어 디코딩으로 떨어진다.

## C. 그다음 — 카메라 고르기 (오늘은 안 함)

7·8단계가 합격하면 **"이 박스로 어떤 카메라를 살 수 있나"** 를 잰다.
카메라는 아직 없어도 된다 — 보유한 카페 영상을 실제 카메라 해상도·화질로
바꿔서 재기 때문이다.

```bash
./venv/bin/python -m edge.bench --frames sample_raw/cafe_sample_angle1.mov --label edge-box
./venv/bin/python -m edge.bench_decode --source sample_raw/cafe_sample_angle1.mov
```

**순서가 중요하다.** `bench` 를 먼저 돌려야 `bench_decode` 가 추론까지 합친
표를 낸다. 그리고 `bench` 는 **5단계의 변환이 끝나 있어야** OpenVINO 줄을
잰다 — 안 하면 `ov-fp32`·`ov-int8` 이 `skip` 으로 빠지고 파이토치 숫자만
남는데, **실제 배포는 OpenVINO라 그 표는 결정에 못 쓴다.**

나오는 표:

```
| 클립     | 디코딩 방식 | 점유 코어 | 전체 대비 | 판정 |
| 4mp_h265 | none        | 0.38      | 5%        | PASS |
| 4mp_h265 | vaapi       | 0.07      | 1%        | PASS |
```

- **점유 코어** = 영상 1초를 푸는 데 드는 CPU 초. 24시간 내내 이만큼 쓴다
- `none` 은 소프트웨어, 나머지는 하드웨어 디코딩.
  **둘의 차이가 곧 드라이버의 값어치다**

| 판정 | 뜻 |
|---|---|
| `PASS` | 전체 코어의 25% 이내 |
| `CONDITIONAL` | 25~50%. 돌긴 하는데 여유가 없다 |
| `FAIL` | 50% 초과. **이 박스로 이 카메라는 못 쓴다** |

### 고르는 법

1. 합산 표에서 `PASS` 인 것 중 **해상도가 가장 높은 것**. 단 **`백엔드` 칸이
   `ov-fp32`(OpenVINO)인 줄**을 본다 — 실제 배포가 그것이라, `pt` 줄이 FAIL
   이어도 그것만으로 카메라를 탈락시키지 않는다
2. 같은 해상도면 **H.265** (같은 화질에 대역폭을 덜 쓴다)
3. **RTSP를 열어주는지 반드시 확인한다.** 하이크비전·다후아 계열은 확정이고,
   국내 저가 브랜드는 검증이 필요하다

> ⚠️ 합산 표는 **근사치다.** 추론이 도는 동안 코어를 전부 쓴다고 가정했다.
> 카메라를 탈락시키는 근거로는 충분하지만 3% 차이를 다투는 데는 못 쓴다.

<details>
<summary><b>막혔을 때</b></summary>

- **`bench` 가 전부 `skip`** — 모델이 없다. 7단계를 한 번 돌리면 받아진다
- **`skip ... (고정 크기 N 로 익스포트된 모델이다)`** — 6단계를 다시
- **클립 만드는 게 너무 느리다** — `--duration 10` 을 붙인다.
  측정값은 길이에 비례하지 않으므로 결과는 같다
- **8MP 클립에서 메모리가 모자란다** —
  `--clips 2mp_h264 2mp_h265 4mp_h264 4mp_h265` 로 8MP를 뺀다.
  4MP까지만 봐도 대부분의 결정은 난다
- **`qsv` 는 실패하고 `vaapi` 는 된다** — 괜찮다. 둘 다 하드웨어 디코딩이고
  자동으로 되는 쪽을 고른다

</details>

## D. 카메라를 산 뒤 (아직 코드가 없다)

1. `plan.md` **T8 — 라이브 스트림 읽기.** 지금 코드는 **녹화 파일 전용**이라
   카메라를 꽂아도 안 돈다. tick마다 영상의 특정 시각으로 되감는 구조인데
   라이브는 되감기가 안 된다. **아키텍처 변경이라 시간이 든다**
2. `plan.md` **T10 — 24시간 무인 운영.** 자동 재시작(systemd), 로그 정리,
   **영상 미저장이 기본값**(개인정보)

둘 다 **카메라가 있어야 검증되므로** 지금 만들지 않았다.

## F. 박스 코드를 최신으로 갱신하는 법 — `git pull`

노트북에서 고친 것을 박스에 반영하는 순서는 **딱 둘**이다.

```bash
# ① 노트북에서 — 박스는 main 에서 받아가므로 main 으로 민다
git push origin JINOI/main:main

# ② 박스에서
cd ~/seatnow
git pull
```

**이게 안전한지 우분투 24.04 에서 GitHub 를 흉내낸 저장소로 통째로 재현해
확인했다** (2026-09-03). 박스에 생기는 파일들 — `venv/`, `sample_raw/` 영상,
`yolov8n.pt`, `*_openvino_model/`, `results/edge_ov/log.jsonl`·`채점표.md`,
`results/edge/export_report.json` — 을 똑같이 만들어놓고 시험했다.

| 시험 | 결과 |
|---|---|
| 처음 받았던 옛 코드(90커밋 전) + 박스 파일들 → `git pull` | ✅ 성공. 박스 파일 9개 전부 그대로, 필요한 파일 전부 도착 |
| 노트북 커밋 → `main` 으로 밀기 → 박스 `git pull` (앞으로 매번 하는 흐름) | ✅ 성공 |
| 박스가 `git checkout JINOI/main` 을 해둔 상태에서 `git pull` | ✅ 성공 (어느 브랜치에 있든 된다) |
| 한글 이름 파일 (`layouts/cafe_sample_짐옮김.json` 등) | ✅ 제대로 받아진다 |
| **박스에 있는 미추적 파일과 같은 경로를 노트북이 커밋** | ⚠️ `git pull` 이 **멈춘다** — 단 박스 파일을 덮어쓰지는 않는다. 아래 |

**깨지는 경우는 하나뿐이고, 그것도 파일을 잃지는 않는다.** 예를 들어 박스가
만든 `results/edge_ov/채점표.md` 와 같은 경로를 노트북에서 커밋해 밀면 박스의
`git pull` 이 이렇게 거부한다:

```
error: The following untracked working tree files would be overwritten by merge:
        results/edge_ov/채점표.md
```

그러면 박스에서 그 파일 이름을 바꾸고 다시 pull 한다:

```bash
mv results/edge_ov/채점표.md results/edge_ov/채점표.박스.md
git pull
```

이걸 피하려면 **박스가 만든 결과 폴더(`results/edge_ov/`)를 노트북에서
커밋하지 않는다.** 노트북 결과는 `results/laptop_*` 처럼 다른 이름을 쓴다.

> `main` 은 **절대 force-push 하지 않는다.** 위 시험은 전부 fast-forward
> (역사가 이어지는 밀기) 전제다. 강제로 밀면 박스의 `git pull` 이 충돌로
> 멈추고, 그때는 박스에서 `git fetch && git reset --hard origin/main` 을
> 해야 하는데 이건 박스의 미커밋 변경을 날린다.

## E. 윈도우에서 돌릴 때

배포는 리눅스다 — 윈도우는 새벽에 강제 업데이트로 재부팅하고 로그인 화면에서
멈춘다. 무인 박스가 그러면 아침에 좌석 정보가 죽어 있고, 매장에서는 "앱이
고장났다"로 보인다. RAM도 1~1.5GB 더 먹는데 4GB에서는 그게 결정적이다.

개발 노트북 등에서 돌릴 일이 있으면 명령만 바꾼다:

- `./venv/bin/python` → `venv\Scripts\python.exe`
- 경로의 `/` → `\`
- ffmpeg 는 <https://www.gyan.dev/ffmpeg/builds/> 의 release full build 를
  `C:\ffmpeg` 에 풀고 `C:\ffmpeg\bin` 을 PATH에 넣는다
  (**넣은 뒤 명령 프롬프트를 새로 연다**)
- 하드웨어 디코딩은 인텔 그래픽 드라이버를 최신으로 올리면 켜진다
