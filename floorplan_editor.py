"""Serve the floor plan editor and take its edits back into both files.

The install has to finish in one visit at the cafe, so the editor is a page
the edge box serves over the cafe wifi: a laptop or a phone can open it and
nobody carries a monitor and mouse.  The same page doubles as the customer
view, which makes the final check free.

Editing the map is not only cosmetic.  Chair ownership is judged as ground
truth (seatnow_core.py:1531-1534) and is nearly impossible to get right on a
camera image, where perspective lifts a far chair onto the neighbouring
table.  With perspective gone it is obvious, so the editor writes ownership
back into the layout -- which is why a save touches two files and why it must
not be able to write only one of them.

    python floorplan_editor.py --layout layouts/cafe_angle1.json \
        --log sample_results/angle1_layout.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import webbrowser
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from floor_projection import floor_anchor
from floorplan import (
    FloorChair,
    FloorPlan,
    Landmark,
    build_draft,
    load_floorplan,
    save_floorplan,
)
from seatnow_layout import (
    COUNTED_ZONE_KIND,
    LayoutChair,
    SeatLayout,
    load_layout,
    save_layout,
)


PROJECT_DIR = Path(__file__).resolve().parent
PAGE_PATH = PROJECT_DIR / "static" / "floorplan_editor.html"


def latest_states(log_path: Path) -> Dict[str, str]:
    """seat_id -> state from the newest record of a run log.

    Only plain tables carry a per-seat state.  A counted_zone reports counts
    over its slots, so individual stools cannot be coloured from it; those
    seats are left out rather than guessed at.
    """
    path = Path(log_path)
    if not path.exists():
        return {}
    last = None
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                last = line
    if last is None:
        return {}
    report = json.loads(last).get("seat_report") or {}
    states: Dict[str, str] = {}
    for seat in report.get("seats") or []:
        if seat.get("kind") == COUNTED_ZONE_KIND:
            continue
        if "state" in seat:
            states[str(seat["seat_id"])] = str(seat["state"])
    return states


def editor_state(
    layout: SeatLayout, plan: FloorPlan, states: Dict[str, str]
) -> Dict[str, object]:
    """Everything the page needs, in one object."""
    drawn = {seat.seat_id for seat in plan.seats}
    return {
        "extent": {"width": plan.extent[0], "height": plan.extent[1]},
        "seats": [
            {
                "seat_id": seat.seat_id,
                "kind": seat.kind,
                "x": seat.x,
                "y": seat.y,
                "w": seat.w,
                "h": seat.h,
                "needs_review": seat.needs_review,
                "state": states.get(seat.seat_id, "unknown"),
            }
            for seat in plan.seats
        ],
        "chairs": [
            {
                "image_anchor": list(chair.image_anchor),
                "seat_id": chair.seat_id,
                "x": chair.x,
                "y": chair.y,
                "w": chair.w,
                "h": chair.h,
                "needs_review": chair.needs_review,
            }
            for chair in plan.chairs
        ],
        "landmarks": [
            {
                "kind": landmark.kind,
                "label": landmark.label,
                "x": landmark.x,
                "y": landmark.y,
                "w": landmark.w,
                "h": landmark.h,
            }
            for landmark in plan.landmarks
        ],
        "unmapped_seats": sorted(set(states) - drawn),
    }


def _seat_owner_names(layout: SeatLayout) -> Dict[str, str]:
    """Every drawable owner name mapped onto the layout table it belongs to."""
    owners: Dict[str, str] = {}
    for table in layout.tables:
        owners[table.name] = table.name
        if table.kind == COUNTED_ZONE_KIND:
            for seat in table.seats:
                owners[f"{table.name}-{seat.id}"] = table.name
    return owners


def apply_edits(
    layout: SeatLayout, plan: FloorPlan, payload: Dict[str, object]
) -> Tuple[SeatLayout, FloorPlan]:
    """Fold the edits from the page into a new layout and a new floor plan.

    ``image_anchor`` is read from the existing plan, never from the payload:
    it identifies a chair across a reassignment and is half of the
    correspondence stage 4 fits its transform to.  A browser has no business
    rewriting it.
    """
    owners = _seat_owner_names(layout)

    seat_edits = {
        str(seat["seat_id"]): seat for seat in payload.get("seats", [])
    }
    seats = tuple(
        replace(
            seat,
            x=float(seat_edits[seat.seat_id]["x"]),
            y=float(seat_edits[seat.seat_id]["y"]),
            w=float(seat_edits[seat.seat_id]["w"]),
            h=float(seat_edits[seat.seat_id]["h"]),
            needs_review=False,
        )
        if seat.seat_id in seat_edits
        else seat
        for seat in plan.seats
    )

    chair_edits = {
        (float(chair["image_anchor"][0]), float(chair["image_anchor"][1])): chair
        for chair in payload.get("chairs", [])
    }
    chairs: List[FloorChair] = []
    ownership: Dict[Tuple[float, float], Optional[str]] = {}
    for chair in plan.chairs:
        edit = chair_edits.get(chair.image_anchor)
        if edit is None:
            chairs.append(chair)
            ownership[chair.image_anchor] = chair.seat_id
            continue
        seat_id = edit.get("seat_id")
        if seat_id is not None and str(seat_id) not in owners:
            raise ValueError(f"의자를 붙일 좌석이 레이아웃에 없습니다: {seat_id!r}")
        chairs.append(
            replace(
                chair,
                seat_id=None if seat_id is None else str(seat_id),
                x=float(edit["x"]),
                y=float(edit["y"]),
                w=float(edit["w"]),
                h=float(edit["h"]),
                needs_review=False,
            )
        )
        ownership[chair.image_anchor] = None if seat_id is None else str(seat_id)

    landmarks = tuple(
        Landmark(
            kind=str(landmark["kind"]),
            label=str(landmark["label"]),
            x=float(landmark["x"]),
            y=float(landmark["y"]),
            w=float(landmark["w"]),
            h=float(landmark["h"]),
        )
        for landmark in payload.get("landmarks", [])
    )

    updated_plan = FloorPlan(
        schema_version=plan.schema_version,
        extent=plan.extent,
        seats=seats,
        chairs=tuple(chairs),
        landmarks=landmarks,
    )
    return _rebuild_layout(layout, ownership), updated_plan


def _rebuild_layout(
    layout: SeatLayout, ownership: Dict[Tuple[float, float], Optional[str]]
) -> SeatLayout:
    """Re-hang every chair under the table the map says owns it.

    Only which list a chair sits in changes.  Its image box is untouched --
    that is the part the installer already drew correctly, and judgement
    reads the lists, so moving it here is enough to change the verdict.
    """
    owners = _seat_owner_names(layout)
    all_chairs: List[Tuple[Optional[str], LayoutChair]] = []
    for table in layout.tables:
        for chair in table.chairs:
            anchor = floor_anchor(chair.box)
            all_chairs.append((ownership.get(anchor, table.name), chair))
    for chair in layout.unassigned_chairs:
        anchor = floor_anchor(chair.box)
        all_chairs.append((ownership.get(anchor, None), chair))

    by_table: Dict[str, List[LayoutChair]] = {table.name: [] for table in layout.tables}
    orphans: List[LayoutChair] = []
    for owner, chair in all_chairs:
        table_name = owners.get(owner or "")
        if table_name is None:
            orphans.append(chair)
        else:
            by_table[table_name].append(chair)

    tables = tuple(
        replace(
            table,
            chairs=tuple(
                replace(chair, id=index)
                for index, chair in enumerate(by_table[table.name], start=1)
            ),
        )
        for table in layout.tables
    )
    return replace(
        layout,
        tables=tables,
        unassigned_chairs=tuple(
            replace(chair, id=index) for index, chair in enumerate(orphans, start=1)
        ),
    )


def save_both(
    layout: SeatLayout, plan: FloorPlan, layout_path: Path, floorplan_path: Path
) -> None:
    """Write both files, or neither.

    Half a save leaves the map and the judgement disagreeing with nothing
    saying so, which is worse than losing the edit.
    """
    layout_path = Path(layout_path)
    floorplan_path = Path(floorplan_path)
    staged: List[Tuple[Path, Path]] = []
    try:
        for path, writer, value in (
            (layout_path, save_layout, layout),
            (floorplan_path, save_floorplan, plan),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            os.close(handle)
            writer(value, Path(temporary))
            staged.append((Path(temporary), path))
    except Exception:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        raise
    for temporary, path in staged:
        os.replace(temporary, path)


class _Handler(BaseHTTPRequestHandler):
    layout_path: Path
    floorplan_path: Path
    log_path: Optional[Path]

    def log_message(self, *args) -> None:  # keep the console for our messages
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _current(self) -> Tuple[SeatLayout, FloorPlan]:
        layout = load_layout(self.layout_path)
        plan = (
            load_floorplan(self.floorplan_path)
            if self.floorplan_path.exists()
            else build_draft(layout)
        )
        return layout, plan

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if self.path == "/state":
            try:
                layout, plan = self._current()
                states = latest_states(self.log_path) if self.log_path else {}
                body = json.dumps(
                    editor_state(layout, plan, states), ensure_ascii=False
                ).encode("utf-8")
            except Exception as exc:  # noqa: BLE001 - the page shows the message
                body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
                self._send(400, body, "application/json; charset=utf-8")
                return
            self._send(200, body, "application/json; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        if self.path != "/save":
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        try:
            layout, plan = self._current()
            updated_layout, updated_plan = apply_edits(layout, plan, payload)
            save_both(
                updated_layout, updated_plan, self.layout_path, self.floorplan_path
            )
        except Exception as exc:  # noqa: BLE001 - the page shows the message
            body = json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
            self._send(400, body.encode("utf-8"), "application/json; charset=utf-8")
            return
        body = json.dumps({"ok": True}, ensure_ascii=False).encode("utf-8")
        self._send(200, body, "application/json; charset=utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout", type=Path, required=True, help="calibrate.py 가 만든 레이아웃"
    )
    parser.add_argument(
        "--floorplan", type=Path, help="평면도 경로 (기본: <layout>.floorplan.json)"
    )
    parser.add_argument("--log", type=Path, help="미리보기에 쓸 JSONL (없으면 회색)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="0.0.0.0 이면 같은 와이파이의 폰에서도 열린다"
    )
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument(
        "--no-open", action="store_true", help="브라우저를 자동으로 열지 않는다"
    )
    args = parser.parse_args(argv)

    _Handler.layout_path = args.layout
    _Handler.floorplan_path = args.floorplan or args.layout.with_suffix(
        ".floorplan.json"
    )
    _Handler.log_path = args.log

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"평면도 편집기: http://localhost:{args.port}/")
    print(
        f"같은 와이파이의 폰에서 열려면 이 PC의 IP로 접속하세요 "
        f"(예: http://192.168.0.10:{args.port}/)"
    )
    print("Ctrl+C 로 종료")
    if not args.no_open:
        webbrowser.open(f"http://localhost:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
