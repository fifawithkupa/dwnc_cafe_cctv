# 엣지 박스 — 위에서 아래로 그대로 치는 순서

> 갱신: 2026-09-05
> 박스: **Dell OptiPlex 7040 Micro / i3-6100T / RAM 4GB / Ubuntu 24.04**
>
> **이 문서는 순서다.** 박스 앞에 앉아서 1번부터 9번까지 차례로 친다.
> 각 단계의 **`→`** 가 나와야 다음으로 간다.
> **1~8 은 끝났다 (2026-09-03). 지금은 9단계다.**
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

## 7단계 — 진짜 실행

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

## 9단계 — 카메라 고르기 ⬅️ **지금 여기**

**무엇을 재나.** 카메라 해상도가 잡아먹는 건 AI 판정이 아니라 **영상을
푸는 일(디코딩)**이다. 판정은 어떤 카메라든 화면을 1280으로 줄여서 보니까
2MP든 8MP든 시간이 거의 같다. 반면 디코딩은 **판정할 때만이 아니라
24시간 내내** 돌고, 화소 수에 비례해 무거워진다. 그래서 "이 박스가 어느
해상도까지 24시간 풀면서 판정도 제때 하나"를 잰다.

**카메라 없이 잰다.** 가진 카페 영상을 카메라 해상도·화질(2MP·4MP·8MP ×
H.264·H.265 = 6가지)로 바꾼 클립을 만들어서 박스에 풀게 한다. 화면은
늘려서 가짜지만, 푸는 비용은 화소 수와 비트레이트를 따르므로 진짜다.

**일이 두 곳으로 나뉜다.** 클립을 **만드는** 건 노트북, **재는** 건 박스.
박스는 코어 2개·4GB라 클립을 만들기엔 너무 느리고(노트북 8코어로 4분 45초,
박스는 그 몇 배), 8MP 클립은 만들다 메모리가 모자랐다. 클립은 재료일 뿐이라
**어디서 만들든 박스의 측정값은 같다.**

| 순서 | 어디서 | 뭘 | 걸리는 시간 |
|---|---|---|---:|
| ① | 박스 | 추론 벤치 | 5분 안팎 |
| ② | 노트북 | 클립 6개 만들기 | 5분 |
| ③ | 노트북 → 박스 | 클립 복사 (150MB) | 1~2분 |
| ④ | 박스 | 디코딩 벤치 + 합산 | 5분 안팎 |

> 이 단계의 코드가 새로 들어갔다. **박스에서 치기 전에 노트북에서
> `git push origin JINOI/main:main`, 박스에서 `cd ~/seatnow && git pull`**
> (부록 F). 안 하면 ②의 `--build-only`, ④의 `--no-build` 를 모른다고 한다.

### ① 박스에서 — 추론 벤치

```bash
cd ~/seatnow
./venv/bin/python -m edge.bench --frames sample_raw/cafe_sample_angle1.mov --label edge-box
```

→ `tick 예산` 표에 이 줄이 있어야 한다:

```
| accuracy_default | ov-fp32 | 30 | 4.8s 안팎 | 32% 안팎 | PASS |
```

`tick 소요`가 **7단계의 "틱당 추론" 4.8초와 비슷**하면 맞게 잰 것이다.
(장치는 7단계와 똑같이 자동으로 고른다. `(CPU)` 라고 찍혀도 내장 그래픽을
쓴다 — 부록 A ①.) 결과는 `results/edge/bench_report.json` 에 남고 ④가 읽는다.

### ② 노트북에서 — 클립 만들기

```powershell
cd C:\Users\jin06\orca\workspaces\dwnc_cafe_cctv\main
venv\Scripts\python.exe -m edge.bench_decode --build-only
```

→ `준비됨:` 줄 6개. 파일은 `results\edge\clips\` 에 생긴다
(`2mp_h264.mp4` … `8mp_h265.mp4`, 15~38MB씩). **노트북 실측 4분 45초.**

이미 만들어 둔 것이 있으면 `재사용:` 이라고 하고 바로 끝난다.

### ③ 노트북에서 — 박스로 복사

```powershell
scp -r C:\Users\jin06\orca\workspaces\dwnc_cafe_cctv\main\results\edge\clips seatnow@192.168.0.42:~/seatnow/results/edge/
```

→ 박스에서 `ls ~/seatnow/results/edge/clips/` 했을 때 mp4 6개.

`~/seatnow/results/edge/` 는 5단계 변환 때 이미 생긴 폴더다. 없다고 하면
박스에서 `mkdir -p ~/seatnow/results/edge` 하고 다시 (3단계와 같은 이유 —
`scp` 는 없는 폴더를 만들어주지 않는다). `mkdir` 을 했는데도 같은 오류면
`~` 를 못 푸는 것이니 전체 경로로: `...:/home/seatnow/seatnow/results/edge/`

### ④ 박스에서 — 디코딩 벤치

```bash
cd ~/seatnow
./venv/bin/python -m edge.bench_decode --no-build --label edge-box
```

`--no-build` 는 "클립을 만들지 말고 있는 것만 재라"다. **박스에서는 항상
붙인다.** 빼면 없는 클립을 박스가 직접 만들려 든다 — 수십 분이고, 8MP는
메모리가 모자란다. (4GB 박스에서 4K H.265 는 만들지 않도록 막아뒀다. 못 만든
클립이 있으면 무엇을 노트북에서 만들어 어떻게 보낼지 화면에 그대로 찍어준다.)

→ 표 세 개가 나온다. **맨 아래 "결정 표"만 보면 된다.** 대략 이렇게 생겼다
(숫자는 노트북 것이라 참고만):

```
## 결정 표 — 배포 설정만 (accuracy_default · ov-fp32)

| 클립     | 디코딩 방식 | 디코딩 | 추론 | 합계 | 판정 |
| 2mp_h264 | none        | 2%     | 4%   | 6%   | PASS |
| 2mp_h264 | vaapi       | 1%     | 4%   | 4%   | PASS |
| ...
| 8mp_h265 | none        | 12%    | 4%   | 16%  | PASS |
| 8mp_h265 | vaapi       | 2%     | 4%   | 6%   | PASS |
```

- **디코딩** = 영상 1초를 푸는 데 박스의 몇 %를 쓰나. 24시간 내내 이만큼.
- **추론** = 판정 한 번이 15초 주기의 몇 %를 쓰나 (①에서 잰 값).
- **합계 ≤ 75% → PASS**, 100%까지 CONDITIONAL, 넘으면 FAIL. 남는 25%는
  운영체제·스트림 재연결·로그 정리 몫이다.
- `none` 은 소프트웨어 디코딩, `vaapi`/`qsv` 는 하드웨어 디코딩이다.
  **둘의 차이가 1단계 드라이버의 값어치다.**

### 고르는 법

1. **결정 표의 하드웨어 줄(`vaapi` 또는 `qsv`)에서 PASS 인 것 중 해상도가
   가장 높은 것.** 같은 해상도면 **H.265** (같은 화질에 대역폭 절반).
2. **같은 클립의 `none` 줄도 본다.** 하드웨어 디코딩은 드라이버 업데이트나
   재로그인 문제로 어느 날 조용히 꺼질 수 있다 (부록 B). 그때는 `none` 값으로
   돈다. `none` 이 FAIL 인 카메라는 "드라이버가 살아있을 때만 되는 카메라"다.
   **`none` 까지 PASS 인 해상도가 안전한 선택이고, `none` 이 CONDITIONAL 이면
   그 위험을 알고 고르는 것이다.**
3. ⚠️ **이 박스에서는 CONDITIONAL 을 FAIL 로 읽는다.** 표는 코어를 4개로
   세는데(부록 A ③) 실제 연산 코어는 2개다. 하이퍼스레딩은 코어 2개 몫을
   못 하므로 표의 %는 실제보다 **최대 1.6배쯤 낙관적**이다. PASS 여유가
   그걸 흡수하고, CONDITIONAL 은 못 한다.
4. **RTSP 를 열어주는 기종인지 반드시 확인한다.** 하이크비전·다후아 계열은
   확정이고, 국내 저가 브랜드는 스펙 표에 "RTSP" 나 "ONVIF" 가 있는지 본다.
   막혀 있으면 위 측정이 전부 무의미하다 — 박스가 영상을 받을 방법이 없다.
   (`plan.md` §2)

> ⚠️ 합산은 **근사치다.** 판정이 도는 동안 코어를 전부 쓴다고 가정했다.
> 카메라를 탈락시키는 근거로는 충분하지만 3% 차이를 다투는 데는 못 쓴다.

<details>
<summary><b>막혔을 때</b></summary>

- **`unrecognized arguments: --build-only` / `--no-build`** — 박스(또는
  노트북)가 옛 코드다. 이 단계 맨 위의 `git push` / `git pull`.
- **① 이 전부 `skip`** — 모델이 없다. 7단계를 한 번 돌리면 받아진다.
- **① 에 `skip ... (고정 크기 N 로 익스포트된 모델이다)`** — 6단계를 다시.
  크기를 박은 모델로 잰 숫자는 거짓말이라 아예 재지 않는다.
- **④ 에 `ov-fp32 줄이 없어 pt 로 대신 보였다`** — ①을 5단계 변환 **뒤에**
  돌리지 않은 것이다. 배포는 OpenVINO 라 그 표로는 결정 못 한다. ① 부터 다시.
- **④ 에 `결정 표: 합산 판정이 없어 못 만든다`** — ①을 안 돌렸거나
  `results/edge/bench_report.json` 이 없다. ① 부터.
- **④ 에 `이 박스에서 못 만든 클립`** — `--no-build` 를 안 붙였거나 ③의
  복사가 안 됐다. 화면에 찍힌 순서대로 하면 된다 (②③④와 같다).
- **④ 에 `하드웨어 디코딩 꺼짐`** — 4단계와 같은 문제다. 4단계 "막혔을 때".
  꺼진 채로 재면 `none` 줄만 나오고, 그건 실제보다 훨씬 나쁜 숫자다.
- **`qsv` 는 실패하고 `vaapi` 는 된다** — 괜찮다. 둘 다 하드웨어 디코딩이고
  자동으로 되는 쪽을 고른다.
- **④ 가 너무 오래 걸린다** — 8MP H.265 소프트웨어 디코딩은 이 박스에서
  실시간보다 느릴 수 있다 (노트북 8코어로도 3.5배속). 30초 클립 하나에
  1분 가까이 걸려도 정상이다. 급하면
  `--clips 2mp_h264 2mp_h265 4mp_h264 4mp_h265` 로 8MP 를 뺀다 —
  4MP 까지만 봐도 결정은 대부분 난다.
- **② 를 박스에서 돌리고 싶다** — 하지 마라. 4K H.265 는 막혀서 안 만들어지고
  나머지도 수십 분이다. 노트북이 없으면 `--clips 2mp_h264 2mp_h265 4mp_h264
  4mp_h265 --duration 10` 으로 작은 것만 박스에서 만든다. 측정값은 클립
  길이에 비례하지 않으므로 10초여도 결과는 같다.

</details>

---

## 끝나면 보내줄 것

**7·8단계 (끝났다):**

1. **7단계가 뱉은 줄 6개 전부** + `Completed 6 samples in Xs` 줄
2. **8단계의 마지막 한 줄**

그거면 합격인지 계산한다. **합격선은 7.5초**(판단 주기 15초의 절반)다.

> 진짜 틱 시간 = **`(Completed 의 X − 첫 틱 inference) ÷ 5`**
> 그냥 6으로 나누면 안 된다 — 이유는 부록 A ②.

**9단계:**

3. ① 의 `tick 예산` 표 전체
4. ④ 의 **결정 표** 전체 (12줄) — 위의 `none` 줄까지 포함해서
5. `results/edge/decode_report.json` 파일 (scp 로 노트북에 가져오거나 `cat`)

그걸로 카메라 해상도·코덱을 확정하고 `plan.md` §2 에 적는다.

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

## C. 카메라 고르기 → **9단계로 옮겼다**

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
