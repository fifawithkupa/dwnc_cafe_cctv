#!/usr/bin/env bash
# Install (or update) the SeatNow user service on the box and start it.
# Run ON THE BOX:   ~/seatnow/deploy/install_service.sh
# Needs no sudo. Re-run after every `git pull` that touched deploy/.
set -eu
cd ~/seatnow
UNIT_DIR=~/.config/systemd/user
mkdir -p "$UNIT_DIR"

if [ ! -f deploy/seatnow.env ]; then
  cp deploy/seatnow.env.example deploy/seatnow.env
  echo "deploy/seatnow.env 를 예시로 만들었다 — 카메라 주소와 레이아웃을 여기에 적는다."
fi
chmod 600 deploy/seatnow.env

cp deploy/seatnow.service "$UNIT_DIR/seatnow.service"
cp deploy/seatnow-fake-camera.service "$UNIT_DIR/seatnow-fake-camera.service"
systemctl --user daemon-reload

# Keep user services alive without a login session (survives reboot).
if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]; then
  loginctl enable-linger "$USER" || echo "⚠️ enable-linger 실패 — 관리자에게 'sudo loginctl enable-linger $USER' 를 부탁한다"
fi

systemctl --user enable --now seatnow.service
sleep 2
systemctl --user --no-pager status seatnow.service | head -12
echo
echo "로그 보기:   journalctl --user -u seatnow -f"
echo "멈추기:      systemctl --user stop seatnow"
echo "다시 켜기:   systemctl --user restart seatnow"
echo "판정 기록:   ls ~/seatnow/results/live/"
