---
name: answer-key-is-human-eye-not-rules
description: "SeatNow 정답지는 \"사람 눈에 뭐가 있나\"이지 규칙이 뭘 해야 하는지가 아니다 — 가림 모름과 확정 대기는 오답이 아니라 맞는 판단"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b3187b4f-f183-4759-8031-6ad382832bac
  modified: 2026-09-02T12:42:15.765Z
---

사장님이 준 좌석 정답지(`results/angle1_layout/angle_answer.md`)는 **사람의
눈으로 "지금 거기 사람/물건이 있나 없나"를 적은 것**이다. 우리 규칙이 무엇을
내야 하는지를 적은 게 아니다.

그래서 다음 둘은 정답과 달라도 **맞는 판단**이고, 채점에서 오답으로 세면 안 된다
(2026-09-02 사장님 공지):

- **가림 모름** — 사람이 자리를 덮어서 모름을 냈다. 비키면 풀린다
- **확정 대기** — 사용중↔빈자리로 바뀌는 중이라 확정을 기다린다

**Why:** 이걸 오답으로 세면 점수를 올리려고 확정 지연을 없애거나 가림 규칙을
푸는 방향으로 "고치게" 되고, 그건 손님을 남의 자리로 보내는 규칙 완화다.
사장님이 이 둘을 "잘 짜여진 로직"이라고 명시적으로 못박았다.

**How to apply:** 채점은 로그의 `state`(앱에 나간 답)와 `raw_state`(이번에 실제로
본 것)를 **두 겹으로** 잰다. 근거가 맞는데 앱만 다르면 '지연', 사유가
`occluded_by_person`·`spans_multiple_seats`·`nearby_person_pose_unknown` 이면
'가림모름' — 둘 다 고치지 않는다. 구현은 `checks/score_answers.py`,
절차는 `/goal` 스킬. 관련: [[machine-fixes-everything-but-finetuning]]
