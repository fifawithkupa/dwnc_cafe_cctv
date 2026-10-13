"""박스 → 카페 연결줄(boxes 표) 바꾸기. 설치 당일 6-3 에서 본인이 `!` 로 실행한다.

저장소 루트에서:
  python deploy/supabase/set_box_cafe.py            -> 지금 상태만 읽는다 (아무것도 안 바꿈)
  python deploy/supabase/set_box_cafe.py moonq      -> boxes 줄을 moonq 로 바꾸고 다시 읽는다
                                                      (문큐의 카페 id 가 moonq 다)

env.local 의 SUPABASE_ACCESS_TOKEN(관리 API 토큰)을 쓴다. 비밀값은 화면에 안 찍는다.
바꾸기 전에 cafes 표에 그 id 가 있는지 먼저 본다 — 없으면 서버가 거부하므로 여기서 멈춘다.
"""
import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # 윈도우 콘솔에서 한글 깨짐 방지

REF = "mlosncixlpogyljifssb"
BOX_EMAIL = "edge-box1@sitnow.net"


def token():
    for line in open("env.local", encoding="utf-8"):
        if line.startswith("SUPABASE_ACCESS_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("env.local 에 SUPABASE_ACCESS_TOKEN 이 없다")


def sql(query):
    req = urllib.request.Request(
        f"https://api.supabase.com/v1/projects/{REF}/database/query",
        data=json.dumps({"query": query}).encode(),
        headers={
            "Authorization": f"Bearer {token()}",
            "Content-Type": "application/json",
            "User-Agent": "seatnow-setup/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")


def show():
    rows = sql(
        "select b.cafe_id, c.name as cafe_name, b.label, u.email "
        "from public.boxes b join auth.users u on u.id = b.auth_user_id "
        "left join public.cafes c on c.id = b.cafe_id"
    )
    for r in rows:
        print(f"  박스 {r['email']} → 카페 {r['cafe_id']} ({r.get('cafe_name') or '이름 없음'})  라벨: {r.get('label')}")
    return rows


def main():
    print("지금 boxes 표:")
    show()
    if len(sys.argv) < 2:
        return
    cafe_id = sys.argv[1].strip()
    cafes = sql(f"select id, name, address from public.cafes where id = '{cafe_id}'")
    if not cafes:
        raise SystemExit(f"cafes 표에 '{cafe_id}' 줄이 없다. 앱 팀이 먼저 넣어야 한다. 아무것도 안 바꿨다.")
    print(f"cafes 표 확인: {cafes[0]['id']} = {cafes[0]['name']} ({cafes[0].get('address')})")
    sql(
        f"update public.boxes set cafe_id = '{cafe_id}', "
        f"label = 'uhho ({cafes[0]['name']})' "
        f"where auth_user_id = (select id from auth.users where email = '{BOX_EMAIL}')"
    )
    print("바꾼 뒤 boxes 표:")
    rows = show()
    ok = any(r["cafe_id"] == cafe_id for r in rows)
    print("결과:", "성공 — 이제 박스 설정 SEATNOW_CAFE_ID 도 같은 값으로 바꾸고 재시작" if ok else "실패 — 바뀌지 않았다")


if __name__ == "__main__":
    main()
