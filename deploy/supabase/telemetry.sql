-- SeatNow 텔레메트리 표 — 판정을 계속 좋게 만들기 위한 기록
-- 설계 근거: 문서/판정개선_데이터설계.md §10
--
-- 손님 앱 표(cafes, cafe_live, seats ...)와 완전히 분리된 seatnow_ 접두사 표다.
-- 앱 동작에 영향을 주지 않는다.
--
-- Supabase 대시보드 → SQL Editor 에 이 파일을 통째로 붙여넣으면 된다.

-- 실행 정보(seatnow_runs)·장식 지도(seatnow_scenery) 표는 쓰지 않는다 (2026-09-11).
-- 카메라가 굳이 서버로 보낼 자료가 아니다 — 지난주와 비교할 열쇠(박스 버전·설정 지문)는
-- 순간 기록 줄마다 두 칸으로 싣고, 장식 지도는 필요해지면 박스 안에서만 만든다.
-- 먼저 만들어 둔 표가 있으면 지운다.  순간 기록 표는 **옛 모양(run_id 칸이 있는 것)일 때만**
-- 지운다 — 이 파일은 여러 번 실행해도 안전해야 하고, 모아 둔 기록을 날리면 안 된다.
do $$
begin
  if exists (
    select 1 from information_schema.columns
    where table_schema = 'public' and table_name = 'seatnow_seat_ticks' and column_name = 'run_id'
  ) then
    drop table seatnow_seat_ticks;
  end if;
end $$;
drop table if exists seatnow_runs;
drop table if exists seatnow_scenery;

-- ---------------------------------------------------------------------------
-- ① 학습 자료 본체 — 한 줄 = 한 자리의 한 순간
-- 전부 넣지 않는다. §10-3 규칙으로 고른 것만 (전체의 10% 안팎).
-- 90일 뒤 ② 로 접고 지운다.
-- ---------------------------------------------------------------------------
create table if not exists seatnow_seat_ticks (
  id             bigserial   primary key,
  cafe_id        text        not null,
  seat_id        text        not null,
  tick_at        timestamptz not null,
  schema_version smallint    not null,
  box_version    text,                   -- 어떤 코드로 돌 때였나 (git 커밋)
  settings_hash  text,                   -- 판정 설정 전체의 지문. 바뀌면 지난주와 비교 불가

  seat_kind      text        not null,   -- table | bar_seat
  capacity       smallint,
  zone           text,

  raw_state      text        not null,   -- 그 순간 본 그대로
  settled_state  text        not null,   -- 흔들림 걸러낸 값
  shown_state    text,                   -- 앱에 실제로 나간 값
  reason_code    text,                   -- 모름이면 왜

  votes_seen     smallint,               -- 5장 중 몇 장이 최종 답에 동의했나
  votes_total    smallint,               -- seen < total 이면 갈린 것 (§4-3)
  vote_counts    jsonb       not null default '{}'::jsonb,  -- {"occupied":3,"empty":2}
  is_open        boolean,                -- 그때 영업중이었나
  border_margin  real,                   -- 0 = 화면 끝에 붙음
  novelty        real,                   -- 빈 상태와 얼마나 달라졌나 (아직 없음)
  chairs_linked  smallint,
  seated_people  smallint,
  confidence     real,

  persons        jsonb       not null default '[]'::jsonb,
  objects        jsonb       not null default '[]'::jsonb,

  sample_kind    text        not null    -- hard | transition | control
);
create index if not exists seatnow_ticks_cafe_time_idx
  on seatnow_seat_ticks (cafe_id, tick_at desc);
create index if not exists seatnow_ticks_reason_idx
  on seatnow_seat_ticks (reason_code) where reason_code is not null;

-- ---------------------------------------------------------------------------
-- ② 요약 — 한 줄 = 한 자리의 하루 (아주 작다, 영구 보관)
-- ① 을 지워도 이건 남는다.  "어느 매장 어느 자리가 문제인가" 는 이것만 봐도 답이 나온다.
-- 주의: 고른 것만이 아니라 그날 전체 틱을 세야 비율이 맞다.
-- ---------------------------------------------------------------------------
create table if not exists seatnow_seat_daily (
  cafe_id       text  not null,
  seat_id       text  not null,
  day           date  not null,
  ticks         int   not null,
  occupied      int   not null default 0,
  empty         int   not null default 0,
  unknown       int   not null default 0,
  ignored       int   not null default 0,
  reason_counts jsonb not null default '{}'::jsonb,  -- 다음 개선 백로그
  unknown_rate  real,                                -- 엔지니어링 지표
  ignore_rate   real,                                -- 설치 품질 지표 (CLAUDE.md)
  primary key (cafe_id, seat_id, day)
);

-- ---------------------------------------------------------------------------
-- 접근 제한 — 이걸 안 하면 손님 앱에 박힌 공개 키로 사람 위치 기록이 읽힌다.
-- 앱과 같은 프로젝트를 쓰므로 표를 나눈 것만으로는 부족하다.
--   · anon (앱)            : 아무 권한 없음
--   · authenticated (박스) : insert 만 (요약은 덮어쓰기용 update 도).  읽기는 없다
-- 분석은 service_role 키로 한다 — service_role 은 RLS 를 우회한다.
-- ---------------------------------------------------------------------------
alter table seatnow_seat_ticks enable row level security;
alter table seatnow_seat_daily enable row level security;

do $$
declare t text;
begin
  foreach t in array array['seatnow_seat_ticks', 'seatnow_seat_daily'] loop
    execute format('drop policy if exists box_insert on %I', t);
    execute format('drop policy if exists box_upsert on %I', t);
    execute format(
      'create policy box_insert on %I for insert to authenticated with check (true)', t);
    -- 요약은 덮어써야 하므로 update 도 연다.
    if t <> 'seatnow_seat_ticks' then
      execute format(
        'create policy box_upsert on %I for update to authenticated using (true) with check (true)', t);
    end if;
  end loop;
end $$;

revoke all on seatnow_seat_ticks, seatnow_seat_daily from anon;
