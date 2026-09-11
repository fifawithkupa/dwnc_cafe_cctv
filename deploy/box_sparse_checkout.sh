#!/usr/bin/env bash
# 박스에는 박스가 돌리는 것만 내려오게 한다.
#
# 저장소를 바꾸지 않는다.  git 의 "일부만 받기(sparse-checkout)" 를 박스에서 한 번
# 켜는 것뿐이고, 되돌리기도 한 줄이다.  켠 뒤에도 `git pull` 은 똑같이 쓰면 된다 —
# 안 받기로 한 것은 파일로 풀리지 않을 뿐 이력은 그대로 따라온다.
#
#   박스에서:  bash deploy/box_sparse_checkout.sh
#   확인:      git sparse-checkout list
#   되돌리기:  git sparse-checkout disable
#
# 방식은 **허용 목록**이다 — "필요한 것만 적고 나머지는 다 뺀다".
# 그래서 저장소에 무엇이 새로 생기든 박스는 깨끗하게 유지된다.
# 대신 **박스가 돌리는 코드 폴더를 새로 만들면 아래 목록에 추가해야 한다.**
# 빠뜨리면 아래 "새 폴더 확인" 이 알려준다.

set -euo pipefail

cd "$(dirname "$0")/.."

before=$(find . -path ./.git -prune -o -type f -print | wc -l | tr -d ' ')

# ── 박스가 실제로 돌리는 것 ────────────────────────────────────────────────
#   engine/   판정 본체            python -m engine.seatnow
#   edge/     카메라·전송·측정     python -m edge.{check_edge,export,bench,live_report,telemetry_upload}
#   checks/   채점 (8단계)         python -m checks.score_answers
#   install/  설치 당일 좌석 확인  install/calibrate.py
#   layouts/  이 매장의 좌석 배치  판정이 매 틱 읽는다
#   deploy/   서비스 파일·설정·SQL
#   results/angle1_layout/angle_answer.md   채점용 정답지 (이 한 장만 쓴다)
#
#   README.md·CLAUDE.md 은 돌리는 데 필요하진 않지만, 박스에 들어온 사람이
#   "이게 뭐 하는 기계인지" 읽을 곳이 있어야 해서 남긴다.  둘 합쳐 15KB 다.
# ──────────────────────────────────────────────────────────────────────────
#
# ⚠️ `--no-cone` 을 `set` 에도 반드시 준다.  `init --no-cone` 만으로는 부족해서,
# 빼먹으면 cone 모드로 해석돼 **패턴이 폴더 이름 취급을 받고 engine/ edge/ 가
# 통째로 사라진다.**  (2026-09-11 실제로 겪음 — 코드가 하나도 안 남았다.)
git sparse-checkout init --no-cone
git sparse-checkout set --no-cone --stdin <<'PATTERNS'
/engine/
/edge/
/checks/
/install/
/layouts/
/deploy/
/results/angle1_layout/angle_answer.md
/requirements.txt
/requirements-edge.txt
/README.md
/CLAUDE.md
PATTERNS

after=$(find . -path ./.git -prune -o -type f -print | wc -l | tr -d ' ')

# ── 안전 확인: 박스가 돌리는 것이 실제로 남아 있나 ─────────────────────────
# 하나라도 없으면 즉시 되돌린다.  반쯤 지워진 박스로 밤을 보내지 않기 위해서다.
missing=0
for must in \
  engine/seatnow.py engine/seatnow_core.py \
  edge/publish.py edge/telemetry.py edge/telemetry_upload.py \
  checks/score_answers.py install/calibrate.py \
  deploy/seatnow.service deploy/supabase/telemetry.sql \
  results/angle1_layout/angle_answer.md \
  requirements-edge.txt
do
  if [ ! -e "$must" ]; then echo "⚠ 빠졌다: $must"; missing=1; fi
done
if [ "$missing" = "1" ]; then
  echo
  echo "필요한 파일이 빠졌다.  되돌린다 — 박스는 손대기 전 그대로다."
  git sparse-checkout disable
  exit 1
fi

# ── 새 폴더 확인: 허용 목록에 없는 코드 폴더가 저장소에 생겼나 ────────────
# 허용 목록 방식이라 새 폴더는 저절로 안 내려온다.  그게 기본값으로 안전하지만,
# 박스가 돌려야 할 코드가 새로 생긴 거라면 알아야 한다.
allowed="engine edge checks install layouts deploy results"
unknown=""
for d in $(git -c core.quotepath=false ls-files | awk -F/ 'NF>1 {print $1}' | sort -u); do
  found=0
  for a in $allowed; do
    if [ "$d" = "$a" ]; then found=1; fi
  done
  if [ "$found" = "0" ]; then unknown="$unknown $d"; fi
done
if [ -n "$unknown" ]; then
  echo "참고 — 박스에 안 내려오는 폴더 (의도한 것이면 그냥 두면 된다):"
  echo "$unknown" | sed 's/^/    /'
  echo
fi

echo "박스에 풀린 파일: ${before}개 → ${after}개"
echo
echo "되돌리기:  git sparse-checkout disable"
