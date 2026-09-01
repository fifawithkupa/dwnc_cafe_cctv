"""SeatNow calibration: register table/chair zones and their links.

State machine is GUI-free (unit tested); the OpenCV window is a thin shell.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

from seatnow_layout import (
    FLOOR_REFERENCE_POINTS,
    SCHEMA_VERSION,
    FloorReference,
    LayoutChair,
    LayoutSeat,
    LayoutTable,
    SeatLayout,
    # Imported despite the underscore on purpose: a copy of this rule here
    # would drift from load_layout's, and the day the tolerance changes on
    # one side only, saving succeeds and loading refuses the same file --
    # exactly the failure invalid_seat_zones() exists to prevent.
    _box_contains,
)

Box = Tuple[float, float, float, float]


def _contains(box: Box, x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


class CalibrationState:
    def __init__(self) -> None:
        self.tables: List[Dict] = []  # {"box": Box, "chairs": List[Box]}
        self.unassigned_chairs: List[Box] = []
        self.floor_points: List[Tuple[float, float]] = []
        self.selected: Optional[Tuple[str, int, int]] = None
        self.pending_reassign: Optional[Tuple[str, int, int]] = None
        self._history: List[
            Tuple[
                List[Dict],
                List[Box],
                List[Tuple[float, float]],
                Optional[Tuple[str, int, int]],
            ]
        ] = []

    def _snapshot(self) -> None:
        self._history.append(
            (
                copy.deepcopy(self.tables),
                list(self.unassigned_chairs),
                list(self.floor_points),
                self.selected,
            )
        )

    def add_table(self, box: Box) -> None:
        self._snapshot()
        self.tables.append(
            {"box": tuple(box), "chairs": [], "kind": "table", "seats": []}
        )
        self.selected = ("table", len(self.tables) - 1, -1)

    def add_zone(self, box: Box) -> None:
        """Add a counted_zone: a bar counter or wall desk the model cannot see.

        Its capacity is however many seat slots get drawn inside it, so the
        zone is useless until ``add_seat`` runs at least once.
        """
        self._snapshot()
        self.tables.append(
            {"box": tuple(box), "chairs": [], "kind": "counted_zone", "seats": []}
        )
        self.selected = ("table", len(self.tables) - 1, -1)

    def add_seat(self, box: Box) -> bool:
        """Draw one seat slot inside the selected counted_zone."""
        table_index = self._selected_table_index()
        if table_index is None:
            return False
        if self.tables[table_index].get("kind") != "counted_zone":
            return False
        self._snapshot()
        self.tables[table_index]["seats"].append(tuple(box))
        return True

    def add_chair(self, box: Box) -> bool:
        """Attach to the selected table, or park it as unassigned.

        Install step 3-c draws chair boxes before anyone has decided which
        table they serve; ownership is set later on the floor plan, where
        perspective is gone and the answer is obvious.  Refusing the draw
        here just left the installer with no way forward.
        """
        table_index = self._selected_table_index()
        self._snapshot()
        if table_index is None:
            self.unassigned_chairs.append(tuple(box))
            return True
        self.tables[table_index]["chairs"].append(tuple(box))
        return True

    def _selected_table_index(self) -> Optional[int]:
        if self.selected is None or self.selected[0] == "orphan":
            return None
        return self.selected[1]

    def select_at(self, x: float, y: float) -> None:
        candidates: List[Tuple[float, Tuple[str, int, int]]] = []
        for ti, table in enumerate(self.tables):
            if _contains(table["box"], x, y):
                candidates.append((_area(table["box"]), ("table", ti, -1)))
            for ci, chair in enumerate(table["chairs"]):
                if _contains(chair, x, y):
                    candidates.append((_area(chair), ("chair", ti, ci)))
            for si, seat in enumerate(table.get("seats", [])):
                if _contains(seat, x, y):
                    candidates.append((_area(seat), ("seat", ti, si)))
        for orphan_index, chair in enumerate(self.unassigned_chairs):
            if _contains(chair, x, y):
                candidates.append((_area(chair), ("orphan", -1, orphan_index)))
        self.selected = min(candidates)[1] if candidates else None

    def delete_selected(self) -> None:
        if self.selected is None:
            return
        self._snapshot()
        kind, ti, ci = self.selected
        if kind == "orphan":
            del self.unassigned_chairs[ci]
        elif kind == "table":
            del self.tables[ti]
        elif kind == "seat":
            del self.tables[ti]["seats"][ci]
        else:
            del self.tables[ti]["chairs"][ci]
        self.selected = None

    def undo(self) -> None:
        if not self._history:
            return
        (
            self.tables,
            self.unassigned_chairs,
            self.floor_points,
            self.selected,
        ) = self._history.pop()

    def begin_reassign(self) -> bool:
        """Arm a chair for reassignment.  True if a chair was selected.

        Delete-and-redraw loses the box position, which is the part the
        installer already got right; only the ownership is wrong.
        """
        if self.selected is None or self.selected[0] not in ("chair", "orphan"):
            return False
        self.pending_reassign = self.selected
        return True

    def _table_at(self, x: float, y: float) -> Optional[int]:
        """Smallest table/zone containing the point, ignoring chairs and seats.

        ``select_at`` prefers the smallest box of any kind, so on a table
        crowded with chairs it always lands on a chair.  Picking a
        reassignment target has to look past them.
        """
        hits = [
            (_area(table["box"]), index)
            for index, table in enumerate(self.tables)
            if _contains(table["box"], x, y)
        ]
        return min(hits)[1] if hits else None

    def reassign_to(self, x: float, y: float) -> Optional[str]:
        """Move the armed chair to whatever table is under the point.

        Returns "table", "zone", or "unassigned"; None when nothing is armed.
        A counted_zone is judged by its seat slots and ``unit_chair_assignments``
        hands those an empty list, so a chair parked on a bar does nothing at
        all -- it is left unassigned rather than pretending it attached.
        """
        if self.pending_reassign is None:
            return None
        kind, table_index, chair_index = self.pending_reassign
        self._snapshot()

        if kind == "orphan":
            box = self.unassigned_chairs.pop(chair_index)
        else:
            box = self.tables[table_index]["chairs"].pop(chair_index)

        target = self._table_at(x, y)
        self.pending_reassign = None
        self.selected = None
        if target is None:
            self.unassigned_chairs.append(box)
            return "unassigned"
        if self.tables[target].get("kind") == "counted_zone":
            self.unassigned_chairs.append(box)
            return "zone"
        self.tables[target]["chairs"].append(box)
        return "table"

    def add_floor_point(self, x: float, y: float) -> int:
        """Collect one of the four floor corners; the fifth click starts over.

        Re-clicking is the only correction anyone needs here, and it costs no
        extra key to learn during an install that already has seven of them.
        """
        self._snapshot()
        if len(self.floor_points) >= FLOOR_REFERENCE_POINTS:
            self.floor_points = []
        self.floor_points.append((float(x), float(y)))
        return len(self.floor_points)

    def invalid_seat_zones(self) -> List[Tuple[int, int]]:
        """Seat slots drawn outside their zone, as (table index, seat index).

        ``load_layout`` refuses such a file (seatnow_layout.py:243-249) but
        ``save_layout`` does not, so without this check the failure surfaces
        hours later on a machine the installer has already left.
        """
        offenders: List[Tuple[int, int]] = []
        for table_index, table in enumerate(self.tables):
            if table.get("kind") != "counted_zone":
                continue
            for seat_index, seat in enumerate(table.get("seats", [])):
                if not _box_contains(table["box"], seat):
                    offenders.append((table_index, seat_index))
        return offenders

    def to_layout(self, source: Dict) -> SeatLayout:
        tables = tuple(
            LayoutTable(
                id=index,
                name=(
                    f"BAR{index}"
                    if table.get("kind") == "counted_zone"
                    else f"T{index}"
                ),
                box=table["box"],
                chairs=tuple(
                    LayoutChair(id=chair_index, box=chair)
                    for chair_index, chair in enumerate(table["chairs"], start=1)
                ),
                kind=table.get("kind", "table"),
                seats=tuple(
                    LayoutSeat(id=seat_index, box=seat)
                    for seat_index, seat in enumerate(
                        table.get("seats", []), start=1
                    )
                ),
            )
            for index, table in enumerate(self.tables, start=1)
        )
        return SeatLayout(
            schema_version=SCHEMA_VERSION,
            source=dict(source),
            tables=tables,
            unassigned_chairs=tuple(
                LayoutChair(id=index, box=box)
                for index, box in enumerate(self.unassigned_chairs, start=1)
            ),
            floor_reference=(
                FloorReference(image_points=tuple(self.floor_points))
                if len(self.floor_points) == FLOOR_REFERENCE_POINTS
                else None
            ),
        )

    @classmethod
    def from_layout(cls, layout: SeatLayout) -> "CalibrationState":
        state = cls()
        for table in layout.tables:
            state.tables.append(
                {
                    "box": tuple(table.box),
                    "chairs": [tuple(chair.box) for chair in table.chairs],
                    "kind": table.kind,
                    "seats": [tuple(seat.box) for seat in table.seats],
                }
            )
        state.unassigned_chairs = [
            tuple(chair.box) for chair in layout.unassigned_chairs
        ]
        if layout.floor_reference is not None:
            state.floor_points = [
                (float(px), float(py))
                for px, py in layout.floor_reference.image_points
            ]
        state.selected = None
        return state


def _grab_frame(video, at_seconds):
    import subprocess
    import tempfile
    from pathlib import Path

    import cv2

    out = Path(tempfile.mkdtemp()) / "calib_frame.png"
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-ss", str(at_seconds),
            "-i", str(video), "-frames:v", "1", "-y", str(out),
        ],
        check=True,
    )
    frame = cv2.imread(str(out))
    if frame is None:
        raise RuntimeError(f"Could not extract frame at {at_seconds}s from {video}")
    return frame


def _preseed(frame, det_model_path):
    """Pre-populate state with detected tables and auto-linked chairs."""
    from ultralytics import YOLO

    from seatnow_core import (
        Detection,
        associate_chairs_to_tables,
        deduplicate_tables,
    )

    model = YOLO(str(det_model_path))
    result = model.predict(source=frame, conf=0.12, imgsz=1280, device="cpu", verbose=False)[0]
    tables, chairs = [], []
    if result.boxes is not None:
        for box_result in result.boxes:
            name = result.names[int(box_result.cls.item())]
            confidence = float(box_result.conf.item())
            box = tuple(float(v) for v in box_result.xyxy[0].tolist())
            if name == "dining table":
                tables.append(Detection(name=name, box=box, confidence=confidence))
            elif name in {"chair", "couch", "bench"} and confidence >= 0.35:
                chairs.append(Detection(name=name, box=box, confidence=confidence))
    tables = deduplicate_tables(tables)
    height, width = frame.shape[:2]
    links = associate_chairs_to_tables(tables, chairs, (height, width))

    state = CalibrationState()
    for table_index, table in enumerate(tables):
        state.add_table(table.box)
        for chair_index in links[table_index]:
            state.add_chair(chairs[chair_index].box)
    state.selected = None
    state._history.clear()
    return state


HELP_TEXT = (
    "[t]able [c]hair [z]one(bar) seat[x] [f]loor [m]ove-chair [d]elete [u]ndo [s]ave [q]uit"
)

TABLE_COLOR = (80, 200, 80)
CHAIR_COLOR = (60, 200, 230)
ZONE_COLOR = (200, 140, 60)
SEAT_COLOR = (230, 190, 90)
ORPHAN_CHAIR_COLOR = (200, 60, 200)
FLOOR_COLOR = (255, 255, 255)
SELECT_COLOR = (60, 60, 235)


def _draw(frame, state, mode, drag):
    import cv2

    canvas = frame.copy()
    for ti, table in enumerate(state.tables):
        tx1, ty1, tx2, ty2 = [int(v) for v in table["box"]]
        is_zone = table.get("kind") == "counted_zone"
        base_color = ZONE_COLOR if is_zone else TABLE_COLOR
        selected = state.selected == ("table", ti, -1)
        cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2),
                      SELECT_COLOR if selected else base_color, 3 if selected else 2)
        seats = table.get("seats", [])
        # cv2.putText draws with a Hershey font, which has no Korean glyphs:
        # every Hangul character renders as "?".  On-canvas labels stay ASCII;
        # Korean belongs in the terminal, where it renders fine.
        label = f"BAR{ti + 1} ({len(seats)} seats)" if is_zone else f"T{ti + 1}"
        cv2.putText(canvas, label, (tx1, max(16, ty1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, base_color, 2, cv2.LINE_AA)
        for si, seat in enumerate(seats):
            sx1, sy1, sx2, sy2 = [int(v) for v in seat]
            seat_selected = state.selected == ("seat", ti, si)
            cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2),
                          SELECT_COLOR if seat_selected else SEAT_COLOR, 2)
            cv2.putText(canvas, str(si + 1), (sx1 + 6, sy2 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, SEAT_COLOR, 2, cv2.LINE_AA)
        if is_zone and not seats:
            cv2.putText(canvas, "! select this zone, press [x], drag one box per seat",
                        (tx1, min(canvas.shape[0] - 8, ty2 + 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 235), 2, cv2.LINE_AA)
        tcx, tcy = (tx1 + tx2) // 2, (ty1 + ty2) // 2
        for ci, chair in enumerate(table["chairs"]):
            cx1, cy1, cx2, cy2 = [int(v) for v in chair]
            chair_selected = state.selected == ("chair", ti, ci)
            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2),
                          SELECT_COLOR if chair_selected else CHAIR_COLOR, 2)
            cv2.line(canvas, (tcx, tcy), ((cx1 + cx2) // 2, (cy1 + cy2) // 2),
                     CHAIR_COLOR, 1, cv2.LINE_AA)
    for orphan_index, chair in enumerate(state.unassigned_chairs):
        ox1, oy1, ox2, oy2 = [int(v) for v in chair]
        selected = state.selected == ("orphan", -1, orphan_index)
        cv2.rectangle(canvas, (ox1, oy1), (ox2, oy2),
                      SELECT_COLOR if selected else ORPHAN_CHAIR_COLOR, 2)
        cv2.putText(canvas, "?", (ox1 + 4, oy2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ORPHAN_CHAIR_COLOR, 2, cv2.LINE_AA)
    if state.pending_reassign is not None:
        cv2.putText(canvas, "MOVE: click the target table", (12, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, SELECT_COLOR, 2, cv2.LINE_AA)
    for index, (px, py) in enumerate(state.floor_points, start=1):
        cv2.circle(canvas, (int(px), int(py)), 7, FLOOR_COLOR, -1)
        cv2.putText(canvas, str(index), (int(px) + 10, int(py) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, FLOOR_COLOR, 2, cv2.LINE_AA)
    if len(state.floor_points) == FLOOR_REFERENCE_POINTS:
        pts = [(int(px), int(py)) for px, py in state.floor_points]
        for start, end in zip(pts, pts[1:] + pts[:1]):
            cv2.line(canvas, start, end, FLOOR_COLOR, 2, cv2.LINE_AA)
    if drag is not None:
        (x1, y1), (x2, y2) = drag
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), SELECT_COLOR, 1)
    cv2.putText(canvas, f"mode={mode}  {HELP_TEXT}", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def run_gui(frame, state, output_path, source):
    import cv2

    from seatnow_layout import save_layout

    mode = "table"
    drag_start = None
    drag_current = None
    window = "SeatNow calibrate"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, _param):
        nonlocal drag_start, drag_current
        if event == cv2.EVENT_LBUTTONDOWN:
            drag_start = (x, y)
            drag_current = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drag_start is not None:
            drag_current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
            x1, y1 = drag_start
            box = (min(x1, x), min(y1, y), max(x1, x), max(y1, y))
            drag_start = None
            drag_current = None
            if state.pending_reassign is not None:
                result = state.reassign_to(x, y)
                if result == "table":
                    print("의자를 그 테이블로 옮겼습니다")
                elif result == "zone":
                    print(
                        "바 구역은 자리 칸이 판정 단위라 의자를 붙여도 쓰이지 "
                        "않습니다 — 소속 미정으로 두었습니다"
                    )
                else:
                    print("의자를 소속 미정으로 두었습니다")
                return
            if mode == "floor":
                count = state.add_floor_point(x, y)
                if count == FLOOR_REFERENCE_POINTS:
                    print("바닥 네 점을 다 찍었습니다. 다시 찍으려면 한 번 더 클릭하세요")
                else:
                    print(f"바닥 점 {count}/{FLOOR_REFERENCE_POINTS}")
                return
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                state.select_at(x, y)  # 클릭 = 선택
            elif mode == "table":
                state.add_table(box)
            elif mode == "zone":
                state.add_zone(box)
            elif mode == "seat":
                if not state.add_seat(box):
                    print(
                        "자리 칸을 넣을 바 구역을 먼저 선택하세요. "
                        "클릭은 겹친 것 중 '가장 작은' 상자를 고르므로, 의자가 없는 "
                        "빈 곳을 클릭해야 구역이 잡힙니다. "
                        "또는 [z]로 다시 그리면 그린 즉시 선택됩니다"
                    )
            else:
                state.add_chair(box)

    cv2.setMouseCallback(window, on_mouse)
    while True:
        drag = (drag_start, drag_current) if drag_start is not None else None
        cv2.imshow(window, _draw(frame, state, mode, drag))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("t"):
            mode = "table"
        elif key == ord("c"):
            mode = "chair"
        elif key == ord("z"):
            mode = "zone"
        elif key == ord("x"):
            mode = "seat"
        elif key == ord("f"):
            mode = "floor"
        elif key == ord("m"):
            if state.begin_reassign():
                print("옮길 테이블을 클릭하세요 (빈 곳을 클릭하면 소속 미정)")
            else:
                print("먼저 옮길 의자를 클릭해 선택하세요")
        elif key == ord("d"):
            state.delete_selected()
        elif key == ord("u"):
            state.undo()
        elif key == ord("s"):
            layout = state.to_layout(source)
            if not layout.tables:
                print("테이블이 없어 저장하지 않았습니다")
                continue
            # 자리 칸이 없는 바 구역은 저장해도 다시 못 연다 (capacity=len(seats)).
            empty_zones = [
                table.name
                for table in layout.tables
                if table.kind == "counted_zone" and not table.seats
            ]
            offenders = state.invalid_seat_zones()
            if offenders:
                spots = ", ".join(
                    f"BAR{table_index + 1}의 {seat_index + 1}번 칸"
                    for table_index, seat_index in offenders
                )
                print(
                    f"자리 칸이 바 구역 밖으로 나가 저장하지 않았습니다: {spots} — "
                    f"[d]로 지우고 구역 안에 다시 그으세요. "
                    f"(구역 밖 자리 칸이 있으면 나중에 판정을 돌릴 때 파일이 거부됩니다)"
                )
                continue
            save_layout(layout, output_path)
            zones = [table for table in layout.tables if table.kind == "counted_zone"]
            summary = (
                f"테이블 {len(layout.tables) - len(zones)}개, "
                f"의자 {len(layout.chair_boxes())}개"
            )
            if zones:
                summary += (
                    f", 바 구역 {len(zones)}개"
                    f"({sum(len(zone.seats) for zone in zones)}석)"
                )
            print(f"저장됨: {output_path} ({summary})")
            if empty_zones:
                # Saving this is fine -- drawing the bar and slicing it into
                # seats are two steps and an install gets interrupted.  What
                # is not fine is judging with it, and seatnow.py refuses that.
                print(
                    f"⚠ 자리 칸이 없는 바 구역: {', '.join(empty_zones)} — "
                    f"그 구역을 선택하고 seat[x]로 자리마다 칸을 그으세요. "
                    f"칸이 없으면 그 구역의 좌석이 아무 집계에도 안 잡히고, "
                    f"이 상태로는 seatnow.py 가 실행을 거부합니다"
                )
        elif key == ord("q") or key == 27:
            break
    cv2.destroyAllWindows()


def main(argv=None) -> int:
    import argparse
    from pathlib import Path

    from seatnow_layout import load_layout

    parser = argparse.ArgumentParser(description="Register table/chair zones for SeatNow")
    parser.add_argument("video", type=Path, help="Reference video (first frame is calibrated)")
    parser.add_argument("--at", type=float, default=0.0, help="Timestamp (s) of the reference frame")
    parser.add_argument("--output", type=Path, help="Layout JSON path (default: layouts/<video stem>.json)")
    parser.add_argument("--edit", type=Path, help="Load an existing layout instead of auto pre-seeding")
    parser.add_argument("--det-model", default="yolov8x.pt", help="Detector for pre-seeding (one-time, accuracy first)")
    parser.add_argument("--no-preseed", action="store_true", help="Start from an empty canvas")
    args = parser.parse_args(argv)

    if not args.video.exists():
        print(f"video not found: {args.video}")
        return 1
    frame = _grab_frame(args.video, args.at)
    height, width = frame.shape[:2]

    if args.edit is not None:
        state = CalibrationState.from_layout(
            load_layout(args.edit).scaled_to(width, height)
        )
    elif args.no_preseed:
        state = CalibrationState()
    else:
        print("자동 탐지로 초기 배치를 채우는 중... (수십 초)")
        state = _preseed(frame, args.det_model)

    output = args.output or Path("layouts") / f"{args.video.stem}.json"
    source = {
        "video": str(args.video),
        "frame_at_seconds": args.at,
        "width": width,
        "height": height,
    }
    run_gui(frame, state, output, source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
