---
name: occluded-streak-rule-decision
description: "2026-09-14 사용자 결정 — 사람 가림 모름 3판정(45초) 연속이면 사용중, 비어 보이면 1판정(15초)에 복귀; 주말 카페 CCTV 영상으로 재채점 예정, 이 규칙 발동 판정은 파인튜닝 재료"
metadata: 
  node_type: memory
  type: project
  originSessionId: b720b488-91fe-445c-abbd-97cb4262cdcb
  modified: 2026-09-14T13:25:25.860Z
---

2026-09-14 사용자가 정한 판정 규칙: 자리 위에 사람이 있어 모름이 **3판정 연속(45초)**이면
사용중으로 확정하고, 그렇게 올라간 사용중은 **비어 보이는 판정 한 번(15초)**이면 빈자리로
되돌린다. 코드는 `engine/seatnow_core.py` 의 `OCCLUDED_CONFIRMATIONS`, 로그 이름표는
`occluded_streak_occupied:<횟수>:<원래사유>`. 박스(uhho)에도 21f307c 로 올라가 있다.

angle1 72칸은 이 규칙으로 70/2 → 68/4 가 됐고(45·60초 T3, 서성이던 사람), 사용자는
"시험 영상이라 감수, 실제 CCTV로 본다"고 했다. **주말(2026-09-19~20)에 카페 CCTV 영상으로
다시 채점할 계획**이며, 이 규칙이 발동한 판정을 모아 사람이 정답을 매기면 파인튜닝 재료다.

**Why:** 앉은 손님은 하반신이 가려 "앉음"이 끝내 안 잡히고, 앱엔 마지막 확정값(빈자리)이
무한정 나갔다. 손님을 남의 자리로 보내는 방향이라 시간 규칙으로 막기로 했다.

**How to apply:** 이 규칙을 "점수 올리려고" 되돌리지 말 것. 카페 실사 채점 때
`occluded_streak_occupied` 줄만 골라 맞았는지 세는 것이 다음 단계. 관련: [[answer-key-is-human-eye-not-rules]],
[[machine-fixes-everything-but-finetuning]].
