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

create table if not exists public.cafe_maps (
  cafe_id     text primary key references public.cafes (id),
  floorplan   jsonb not null,
  version     integer not null default 1,
  updated_at  timestamptz not null default now()
);

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
create or replace function public.seatnow_bump_map_version()
returns trigger language plpgsql as $$
begin
  new.version = old.version + 1;
  new.updated_at = now();
  return new;
end $$;

drop trigger if exists cafe_maps_bump on public.cafe_maps;
create trigger cafe_maps_bump
  before update on public.cafe_maps
  for each row execute function public.seatnow_bump_map_version();

-- 권한: 앱(anon)은 읽기만. 박스(authenticated)는 boxes 에 적힌 자기 카페 줄만 쓴다.
-- 카페에 놓인 박스를 누가 가져가도 다른 카페 데이터는 건드릴 수 없다.
alter table public.cafes     enable row level security;
alter table public.boxes     enable row level security;
alter table public.cafe_live enable row level security;
alter table public.cafe_maps enable row level security;

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

drop policy if exists "cafe_maps: anyone reads" on public.cafe_maps;
create policy "cafe_maps: anyone reads" on public.cafe_maps
  for select to anon, authenticated using (true);

drop policy if exists "cafe_maps: box inserts own cafe" on public.cafe_maps;
create policy "cafe_maps: box inserts own cafe" on public.cafe_maps
  for insert to authenticated
  with check (cafe_id in (select cafe_id from public.boxes where auth_user_id = auth.uid()));

drop policy if exists "cafe_maps: box updates own cafe" on public.cafe_maps;
create policy "cafe_maps: box updates own cafe" on public.cafe_maps
  for update to authenticated
  using (cafe_id in (select cafe_id from public.boxes where auth_user_id = auth.uid()))
  with check (cafe_id in (select cafe_id from public.boxes where auth_user_id = auth.uid()));

-- 앱이 구독할 수 있게 실시간 발행에 넣는다 (이미 들어 있으면 넘어간다).
do $$
begin
  begin
    alter publication supabase_realtime add table public.cafe_live;
  exception when duplicate_object then null;
  end;
  begin
    alter publication supabase_realtime add table public.cafe_maps;
  exception when duplicate_object then null;
  end;
end $$;

-- 첫 카페와 박스 (값을 바꿔서 실행한다):
-- insert into public.cafes (id, name) values ('dwnc', '카페 이름');
-- insert into public.boxes (auth_user_id, cafe_id, label)
--   values ('<Authentication 에서 만든 박스 사용자의 UUID>', 'dwnc', 'uhho');
