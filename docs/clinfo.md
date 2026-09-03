# 내장 그래픽(HD 530)이 AI 연산에 쓰이는지 확인 — 3분

박스를 막 켰을 때 **이것부터** 본다. 저장소를 받아놓지 않아도 된다.
왜 중요한지는 `docs/edge-setup.md` §0. 한 줄로: **이게 되면 판단 1초대,
안 되면 10초대다.**

## 1. 깐다

```bash
sudo apt update
sudo apt install -y intel-opencl-icd clinfo python3-venv
sudo usermod -aG render,video $USER
```

**로그아웃했다가 다시 로그인한다.** (그래야 그래픽 장치 접근 권한이 붙는다.)

## 2. 두 줄로 확인한다

```bash
clinfo | grep -i "Device Name"
```
→ **`Intel(R) HD Graphics 530`** 같은 줄이 나와야 한다.

```bash
python3 -m venv /tmp/ovcheck
/tmp/ovcheck/bin/pip install -q openvino
/tmp/ovcheck/bin/python -c "import openvino as ov; print(ov.Core().available_devices)"
```
→ **`['CPU', 'GPU']`** 가 나와야 한다.

## 3. 결과

| 나온 것 | 뜻 | 다음 |
|---|---|---|
| `['CPU', 'GPU']` | ✅ **합격.** 판단이 1초대로 끝날 가능성이 크다 | **`docs/edge-setup.md` 1단계**로 간다 |
| `['CPU']` 만 | ⚠️ 내장 그래픽을 못 쓴다. 추론이 5~6배 느려진다 | 아래 |

### `['CPU']` 만 나왔을 때 순서대로

1. **재로그인을 했는가** — `groups` 에 `render` 가 보여야 한다
2. `clinfo` 는 뭐라고 하나 — 장치가 아예 없다고 하면 드라이버 문제다
3. 우분투 버전을 본다 — `lsb_release -a`
   - **24.04 / 22.04** → `intel-opencl-icd` 가 맞다. 다시 깔아본다
   - **25.10 이상** → `sudo apt install -y intel-opencl-icd-legacy` (이 버전부터
     기본 패키지가 HD 530을 뺐다)
4. 그래도 안 되면 CPU만으로 가야 한다. 포기가 아니라 카드가 있다 —
   `docs/edge-first-run.md` 5단계 ①·④

## 끝나면 지운다

```bash
rm -rf /tmp/ovcheck
```
확인용 임시 환경이다. 실제 실행 환경은 `docs/edge-first-run.md` 1단계에서
`requirements-edge.txt` 로 따로 만든다.
