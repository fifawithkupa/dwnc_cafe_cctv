---
name: two-computers-dev-and-field
description: "2026-10-17 부터 컴퓨터 두 대 — 새 고성능 PC 는 개발, 기존 노트북(jin06)은 카페 현장용(git pull · 박스 접속 · 기록 받아오기). 기억 메모는 저장소 .claude/memory/ 로 git 동기화"
metadata:
  type: project
---

2026-10-17 사용자가 정했다: **개발은 새 고성능 PC**, 기존 노트북은 **카페마다 들고 가는 현장용**으로만 쓴다.

- 현장 노트북이 하는 일: 카페 가기 전 `git pull`, 박스 ssh 접속(열쇠 `~/.ssh/id_ed25519` 는 노트북에만 있다),
  좌석 다시 그리기 → 그 자리에서 커밋 → 인터넷 되는 곳에서 push, 박스 판정 기록 받아오기.
- **git 으로 안 넘어가는 것** — 두 컴퓨터 사이에 USB 로 직접 옮긴다:
  `env.local`(Supabase 열쇠 — 공개 저장소라 절대 커밋 금지), `results/box_pull/`(자리별 사람 위치가 있어
  .gitignore 로 막아 둠). 시험 영상 `sample_raw/` 은 팀 구글 드라이브에서 받는다.
- 두 컴퓨터가 같은 브랜치 `JINOI/main` 을 쓴다. **작업 시작 전 pull, 끝나면 push** — 안 하면 엉킨다.
- 기억 메모는 저장소 `.claude/memory/` 에 있고 각 컴퓨터의 `.claude/settings.local.json`
  (`autoMemoryDirectory`, 절대경로)이 그 폴더를 가리킨다. 메모를 고치면 커밋·push 해야 다른 컴퓨터로 간다.
  설정 방법은 CLAUDE.md "컴퓨터 두 대" 절. 옛 메모 폴더(`~/.claude/projects/C--Users-jin06-orca-dwnc-cafe-cctv/memory/`)는
  노트북에 백업으로 남겨 뒀다 — 더는 안 쓴다.

**Why:** 노트북은 학습·채점 돌리기엔 느리고, 현장에는 박스 열쇠·유선 연결이 있는 노트북이 필요하다.
**How to apply:** 새 PC 에서 채점 결과가 노트북 기준(`accepted=68 wrong=4`)과 다르면 개발 전에 먼저 원인을 본다.
노트북에서 "카페 간다"류 요청을 받으면 먼저 `git pull`. 관련: [[moonq-offline-weekly-pull]], [[cafe-install-day-protocol]].
