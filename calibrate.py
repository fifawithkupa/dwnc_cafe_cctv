"""SeatNow calibration: register table/chair zones and their links.

State machine is GUI-free (unit tested); the OpenCV window is a thin shell.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

from seatnow_layout import (
    SCHEMA_VERSION,
    LayoutChair,
    LayoutTable,
    SeatLayout,
)

Box = Tuple[float, float, float, float]


def _contains(box: Box, x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


class CalibrationState:
    def __init__(self) -> None:
        self.tables: List[Dict] = []  # {"box": Box, "chairs": List[Box]}
        self.selected: Optional[Tuple[str, int, int]] = None
        self._history: List[Tuple[List[Dict], Optional[Tuple[str, int, int]]]] = []

    def _snapshot(self) -> None:
        self._history.append((copy.deepcopy(self.tables), self.selected))

    def add_table(self, box: Box) -> None:
        self._snapshot()
        self.tables.append({"box": tuple(box), "chairs": []})
        self.selected = ("table", len(self.tables) - 1, -1)

    def add_chair(self, box: Box) -> bool:
        table_index = self._selected_table_index()
        if table_index is None:
            return False
        self._snapshot()
        self.tables[table_index]["chairs"].append(tuple(box))
        return True

    def _selected_table_index(self) -> Optional[int]:
        if self.selected is None:
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
        self.selected = min(candidates)[1] if candidates else None

    def delete_selected(self) -> None:
        if self.selected is None:
            return
        self._snapshot()
        kind, ti, ci = self.selected
        if kind == "table":
            del self.tables[ti]
        else:
            del self.tables[ti]["chairs"][ci]
        self.selected = None

    def undo(self) -> None:
        if not self._history:
            return
        self.tables, self.selected = self._history.pop()

    def to_layout(self, source: Dict) -> SeatLayout:
        tables = tuple(
            LayoutTable(
                id=index,
                name=f"T{index}",
                box=table["box"],
                chairs=tuple(
                    LayoutChair(id=chair_index, box=chair)
                    for chair_index, chair in enumerate(table["chairs"], start=1)
                ),
            )
            for index, table in enumerate(self.tables, start=1)
        )
        return SeatLayout(
            schema_version=SCHEMA_VERSION, source=dict(source), tables=tables
        )

    @classmethod
    def from_layout(cls, layout: SeatLayout) -> "CalibrationState":
        state = cls()
        for table in layout.tables:
            state.tables.append(
                {
                    "box": tuple(table.box),
                    "chairs": [tuple(chair.box) for chair in table.chairs],
                }
            )
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


HELP_TEXT = "[t]able  [c]hair->selected  [d]elete  [u]ndo  [s]ave  [q]uit"

TABLE_COLOR = (80, 200, 80)
CHAIR_COLOR = (60, 200, 230)
SELECT_COLOR = (60, 60, 235)


def _draw(frame, state, mode, drag):
    import cv2

    canvas = frame.copy()
    for ti, table in enumerate(state.tables):
        tx1, ty1, tx2, ty2 = [int(v) for v in table["box"]]
        selected = state.selected == ("table", ti, -1)
        cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2),
                      SELECT_COLOR if selected else TABLE_COLOR, 3 if selected else 2)
        cv2.putText(canvas, f"T{ti + 1}", (tx1, max(16, ty1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TABLE_COLOR, 2, cv2.LINE_AA)
        tcx, tcy = (tx1 + tx2) // 2, (ty1 + ty2) // 2
        for ci, chair in enumerate(table["chairs"]):
            cx1, cy1, cx2, cy2 = [int(v) for v in chair]
            chair_selected = state.selected == ("chair", ti, ci)
            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2),
                          SELECT_COLOR if chair_selected else CHAIR_COLOR, 2)
            cv2.line(canvas, (tcx, tcy), ((cx1 + cx2) // 2, (cy1 + cy2) // 2),
                     CHAIR_COLOR, 1, cv2.LINE_AA)
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
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                state.select_at(x, y)  # 클릭 = 선택
            elif mode == "table":
                state.add_table(box)
            else:
                if not state.add_chair(box):
                    print("의자를 붙일 테이블을 먼저 클릭으로 선택하세요")

    cv2.setMouseCallback(window, on_mouse)
    while True:
        drag = (drag_start, drag_current) if drag_start is not None else None
        cv2.imshow(window, _draw(frame, state, mode, drag))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("t"):
            mode = "table"
        elif key == ord("c"):
            mode = "chair"
        elif key == ord("d"):
            state.delete_selected()
        elif key == ord("u"):
            state.undo()
        elif key == ord("s"):
            layout = state.to_layout(source)
            if not layout.tables:
                print("테이블이 없어 저장하지 않았습니다")
                continue
            save_layout(layout, output_path)
            print(f"저장됨: {output_path} (테이블 {len(layout.tables)}개, "
                  f"의자 {len(layout.chair_boxes())}개)")
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
