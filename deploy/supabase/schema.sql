-- SeatNow: 박스가 쓰고 앱이 읽는 표.
-- Supabase 대시보드 → SQL Editor 에 통째로 붙여넣고 Run.
-- 여러 번 실행해도 안전하다 (if not exists / drop ... if exists).

create table if not exists public.cafes (
  id          text primary key,             -- 짧은 이름. 박스의 SEATNOW_CAFE_ID 와 같다
  name        text not null,
  address     text,
  created_at  timestamptz not null default now()
);

create table if not exists public.boxes (
  auth_user_id uuid primary key references auth.users (id) on delete cascade,
  cafe_id      text not null references public.cafes (id),
  label        text
);

create table if not exists public.cafe_live (
  cafe_id          text primary key references public.cafes (id),
  status           text not null check (status in ('live', 'gap')),
  total_tables     integer not null,
  occupied_tables  integer not null,
  free_tables      integer not null,
  unknown_tables   integer not null,
  seats            jsonb not null,
  tick_at          timestamptz,
  updated_at       timestamptz not null default now(),  -- 서버가 찍는다. 앱은 이걸 본다
  box_version      text,
  schema_version   integer not null default 1
);

-- 이름표 사진: 카페당 한 줄. 설치 때 박스가 카메라 화면에 자리 이름표만 써 넣은 사진을
-- Storage `seat-sheets/<cafe_id>.jpg` 에 올리고 여기 줄을 하나 남긴다. 앱 팀은 이 사진을
-- 보고 피그마 네모마다 이름표를 붙여 앱 코드에 넣는다 (앱팀할일.md). 다시 찍으면 덮어쓴다.
create table if not exists public.cafe_seat_sheets (
  cafe_id      text primary key references public.cafes (id),
  seat_ids     jsonb not null,               -- [{"seat_id":"T1","kind":"table","zone":null}, ...]
  image_path   text not null,                -- 'seat-sheets/<cafe_id>.jpg'
  taken_at     timestamptz,                  -- 박스 시계
  box_version  text,
  updated_at   timestamptz not null default now()
);

-- 예전 자동 지도 표는 쓰지 않는다 (2026-09-10, 피그마로 대체). 남아 있으면 지운다.
drop table if exists public.cafe_maps;
drop function if exists public.seatnow_bump_map_version();

-- updated_at 은 박스 시계가 아니라 서버 시계다.
-- 박스 시계가 틀어져도 앱의 45초 규칙이 흔들리지 않게 하려는 것이다.
create or replace function public.seatnow_touch_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end $$;

drop trigger if exists cafe_live_touch on public.cafe_live;
create trigger cafe_live_touch
  before update on public.cafe_live
  for each row execute function public.seatnow_touch_updated_at();

-- 지도는 바뀔 때마다 번호가 올라간다. 앱은 번호가 달라졌을 때만 다시 받는다.
drop trigger if exists cafe_seat_sheets_touch on public.cafe_seat_sheets;
create trigger cafe_seat_sheets_touch
  before update on public.cafe_seat_sheets
  for each row execute function public.seatnow_touch_updated_at();

-- 권한: 앱(anon)은 읽기만. 박스(authenticated)는 boxes 에 적힌 자기 카페 줄만 쓴다.
-- 카페에 놓인 박스를 누가 가져가도 다른 카페 데이터는 건드릴 수 없다.
alter table public.cafes     enable row level security;
alter table public.boxes     enable row level security;
alter table public.cafe_live enable row level security;
alter table public.cafe_seat_sheets enable row level security;

drop policy if exists "cafes: anyone reads" on public.cafes;
create policy "cafes: anyone reads" on public.cafes
  for select to anon, authenticated using (true);

drop policy if exists "boxes: box reads itself" on public.boxes;
create policy "boxes: box reads itself" on public.boxes
  for select to authenticated using (auth_user_id = auth.uid());

drop policy if exists "cafe_live: anyone reads" on public.cafe_live;
create policy "cafe_live: anyone reads" on public.cafe_live
  for select to anon, authenticated using (true);

drop policy if exists "cafe_live: box inserts own cafe" on public.cafe_live;
create policy "cafe_live: box inserts own cafe" on public.cafe_live
  for insert to authenticated
  with check (cafe_id in (select cafe_id from public.boxes where auth_user_id = auth.uid()));

drop policy if exists "cafe_live: box updates own cafe" on public.cafe_live;
create policy "cafe_live: box updates own cafe" on public.cafe_live
  for update to authenticated
  using (cafe_id in (select cafe_id from public.boxes where auth_user_id = auth.uid()))
  with check (cafe_id in (select cafe_id from public.boxes where auth_user_id = auth.uid()));

drop policy if exists "cafe_seat_sheets: anyone reads" on public.cafe_seat_sheets;
create policy "cafe_seat_sheets: anyone reads" on public.cafe_seat_sheets
  for select to anon, authenticated using (true);

drop policy if exists "cafe_seat_sheets: box inserts own cafe" on public.cafe_seat_sheets;
create policy "cafe_seat_sheets: box inserts own cafe" on public.cafe_seat_sheets
  for insert to authenticated
  with check (cafe_id in (select cafe_id from public.boxes where auth_user_id = auth.uid()));

drop policy if exists "cafe_seat_sheets: box updates own cafe" on public.cafe_seat_sheets;
create policy "cafe_seat_sheets: box updates own cafe" on public.cafe_seat_sheets
  for update to authenticated
  using (cafe_id in (select cafe_id from public.boxes where auth_user_id = auth.uid()))
  with check (cafe_id in (select cafe_id from public.boxes where auth_user_id = auth.uid()));

-- 이름표 사진 저장소. 비공개 — 손님 앱은 안 읽고 앱 팀·우리만 대시보드에서 본다.
-- 박스는 자기 카페 파일(<cafe_id>.jpg) 하나만 올리고 덮어쓸 수 있다.
insert into storage.buckets (id, name, public)
  values ('seat-sheets', 'seat-sheets', false)
  on conflict (id) do nothing;

drop policy if exists "seat-sheets: box writes own file" on storage.objects;
create policy "seat-sheets: box writes own file" on storage.objects
  for insert to authenticated
  with check (
    bucket_id = 'seat-sheets'
    and name in (select cafe_id || '.jpg' from public.boxes where auth_user_id = auth.uid())
  );

drop policy if exists "seat-sheets: box overwrites own file" on storage.objects;
create policy "seat-sheets: box overwrites own file" on storage.objects
  for update to authenticated
  using (
    bucket_id = 'seat-sheets'
    and name in (select cafe_id || '.jpg' from public.boxes where auth_user_id = auth.uid())
  );

drop policy if exists "seat-sheets: signed-in users read" on storage.objects;
create policy "seat-sheets: signed-in users read" on storage.objects
  for select to authenticated using (bucket_id = 'seat-sheets');

-- 앱이 구독할 수 있게 실시간 발행에 넣는다 (이미 들어 있으면 넘어간다).
do $$
begin
  begin
    alter publication supabase_realtime add table public.cafe_live;
  exception when duplicate_object then null;
  end;
end $$;

-- 첫 카페와 박스 (값을 바꿔서 실행한다):
-- insert into public.cafes (id, name) values ('dwnc', '카페 이름');
-- insert into public.boxes (auth_user_id, cafe_id, label)
--   values ('<Authentication 에서 만든 박스 사용자의 UUID>', 'dwnc', 'uhho');

-- ── 앱이 먼저 만든 cafes 표는 박스가 건드리지 않는다 (2026-09-10 확인·결정) ────
-- 실제 프로젝트를 읽어보니:
--  * 앱에는 자리별 표 `seats` 가 따로 있고, 트리거 `sync_cafe_seat_count` 가
--    `cafes.seats_total`·`seats_available` 을 그 표에서 다시 계산한다.
--  * `cafes.congestion` 은 check 제약(`cafes_congestion_ck`)으로
--    'available'/'full' 두 값만 받는다.
-- 그래서 cafes 에 쓰기 정책을 열지 않는다.  박스는 cafe_live·cafe_seat_sheets 에만 쓴다.
