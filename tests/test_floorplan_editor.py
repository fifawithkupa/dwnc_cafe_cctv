"""Unit tests for the floor plan editor's server side.

No socket is opened and no browser runs: what matters is that an edit lands
in both files and that a half-written save cannot happen.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from floorplan import build_draft, load_floorplan
from floorplan_editor import apply_edits, editor_state, latest_states, save_both
from seatnow_layout import (
    FloorReference,
    LayoutChair,
    LayoutTable,
    SeatLayout,
    load_layout,
)


REFERENCE = FloorReference(
    image_points=((700.0, 900.0), (900.0, 600.0), (1300.0, 600.0), (1600.0, 900.0))
)


def layout():
    return SeatLayout(
        schema_version=3,
        source={"width": 1920, "height": 1080},
        tables=(
            LayoutTable(
                id=1,
                name="T1",
                box=(800.0, 700.0, 1000.0, 850.0),
                chairs=(LayoutChair(id=1, box=(780.0, 800.0, 830.0, 880.0)),),
            ),
            LayoutTable(id=2, name="T2", box=(1200.0, 700.0, 1400.0, 850.0)),
        ),
        floor_reference=REFERENCE,
    )


class EditorStateTests(unittest.TestCase):
    def test_state_carries_seats_chairs_and_extent(self):
        plan = build_draft(layout())
        state = editor_state(layout(), plan, {})
        self.assertEqual(len(state["seats"]), 2)
        self.assertEqual(len(state["chairs"]), 1)
        self.assertIn("extent", state)

    def test_live_states_are_attached_when_known(self):
        plan = build_draft(layout())
        state = editor_state(layout(), plan, {"T1": "occupied"})
        by_id = {seat["seat_id"]: seat for seat in state["seats"]}
        self.assertEqual(by_id["T1"]["state"], "occupied")
        self.assertEqual(by_id["T2"]["state"], "unknown")

    def test_a_seat_only_in_the_report_is_counted_not_drawn(self):
        # Layout and map disagreeing is a real failure worth reporting, but
        # inventing a position for the stray seat would hide it.
        plan = build_draft(layout())
        state = editor_state(layout(), plan, {"T1": "occupied", "GHOST": "empty"})
        self.assertEqual([seat["seat_id"] for seat in state["seats"]], ["T1", "T2"])
        self.assertEqual(state["unmapped_seats"], ["GHOST"])


class ApplyEditsTests(unittest.TestCase):
    def _payload(self, **overrides):
        plan = build_draft(layout())
        payload = {
            "seats": [
                # Spread out: two seats dropped on the same spot would be
                # pushed apart by separate_overlaps, which is correct but
                # would hide whether the move itself was kept.
                {
                    "seat_id": seat.seat_id,
                    "x": 11.0 + 400.0 * index,
                    "y": 22.0,
                    "w": seat.w,
                    "h": seat.h,
                }
                for index, seat in enumerate(plan.seats)
            ],
            "chairs": [
                {
                    "image_anchor": list(chair.image_anchor),
                    "seat_id": chair.seat_id,
                    "x": 33.0,
                    "y": 44.0,
                    "w": chair.w,
                    "h": chair.h,
                }
                for chair in plan.chairs
            ],
            "landmarks": [],
        }
        payload.update(overrides)
        return plan, payload

    def test_moved_seat_position_is_kept(self):
        plan, payload = self._payload()
        _, updated = apply_edits(layout(), plan, payload)
        self.assertEqual((updated.seats[0].x, updated.seats[0].y), (11.0, 22.0))

    def test_image_anchor_is_never_overwritten(self):
        # It is the correspondence stage 4 needs and the key that identifies
        # a chair; letting the browser rewrite it would destroy both.
        plan, payload = self._payload()
        payload["seats"][0]["image_anchor"] = [1.0, 2.0]
        _, updated = apply_edits(layout(), plan, payload)
        self.assertEqual(updated.seats[0].image_anchor, plan.seats[0].image_anchor)

    def test_chair_reassignment_moves_it_in_the_layout(self):
        plan, payload = self._payload()
        payload["chairs"][0]["seat_id"] = "T2"
        updated_layout, _ = apply_edits(layout(), plan, payload)
        by_name = {table.name: table for table in updated_layout.tables}
        self.assertEqual(len(by_name["T1"].chairs), 0)
        self.assertEqual(len(by_name["T2"].chairs), 1)

    def test_chair_unassignment_moves_it_to_unassigned(self):
        plan, payload = self._payload()
        payload["chairs"][0]["seat_id"] = None
        updated_layout, _ = apply_edits(layout(), plan, payload)
        self.assertEqual(len(updated_layout.unassigned_chairs), 1)
        self.assertEqual(len(updated_layout.tables[0].chairs), 0)

    def test_chair_image_box_is_unchanged_by_reassignment(self):
        plan, payload = self._payload()
        payload["chairs"][0]["seat_id"] = "T2"
        updated_layout, _ = apply_edits(layout(), plan, payload)
        moved = updated_layout.tables[1].chairs[0]
        self.assertEqual(moved.box, (780.0, 800.0, 830.0, 880.0))

    def test_landmarks_are_stored(self):
        plan, payload = self._payload()
        payload["landmarks"] = [
            {"kind": "entrance", "label": "입구", "x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0}
        ]
        _, updated = apply_edits(layout(), plan, payload)
        self.assertEqual(updated.landmarks[0].label, "입구")

    def test_unknown_seat_id_on_a_chair_is_refused(self):
        plan, payload = self._payload()
        payload["chairs"][0]["seat_id"] = "NOPE"
        with self.assertRaises(ValueError):
            apply_edits(layout(), plan, payload)


class SaveBothTests(unittest.TestCase):
    def test_both_files_are_written(self):
        plan = build_draft(layout())
        with tempfile.TemporaryDirectory() as raw:
            layout_path = Path(raw) / "layout.json"
            plan_path = Path(raw) / "floorplan.json"
            save_both(layout(), plan, layout_path, plan_path)
            self.assertEqual(len(load_layout(layout_path).tables), 2)
            self.assertEqual(len(load_floorplan(plan_path).seats), 2)

    def test_a_failing_write_leaves_both_files_untouched(self):
        # Half a save is worse than none: the map and the judgement would
        # disagree with nothing saying so.
        plan = build_draft(layout())
        with tempfile.TemporaryDirectory() as raw:
            layout_path = Path(raw) / "layout.json"
            plan_path = Path(raw) / "floorplan.json"
            save_both(layout(), plan, layout_path, plan_path)
            before = layout_path.read_text(encoding="utf-8")

            blocked = Path(raw) / "blocked"
            blocked.write_text("not a directory", encoding="utf-8")
            with self.assertRaises(Exception):
                save_both(layout(), plan, layout_path, blocked / "floorplan.json")
            self.assertEqual(layout_path.read_text(encoding="utf-8"), before)


class LatestStatesTests(unittest.TestCase):
    def test_last_record_wins(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "run.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "timestamp": stamp,
                            "seat_report": {
                                "seats": [
                                    {"seat_id": "T1", "kind": "table", "state": state}
                                ]
                            },
                        }
                    )
                    for stamp, state in ((0.0, "empty"), (15.0, "occupied"))
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(latest_states(path), {"T1": "occupied"})

    def test_counted_zone_seats_are_left_uncoloured(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "run.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "timestamp": 0.0,
                        "seat_report": {
                            "seats": [
                                {
                                    "seat_id": "BAR7",
                                    "kind": "counted_zone",
                                    "capacity": 2,
                                    "occupied": 1,
                                    "free": 1,
                                    "unknown": 0,
                                }
                            ]
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            # A zone reports counts, not per-slot states, so the map cannot
            # colour individual stools from it and says so by leaving them out.
            self.assertEqual(latest_states(path), {})

    def test_missing_file_is_empty(self):
        self.assertEqual(latest_states(Path("does-not-exist.jsonl")), {})


class WallsTests(unittest.TestCase):
    """The room outline the installer clicks once.

    A camera sees seats and never walls, so nothing can derive them.  Without
    them a customer sees boxes floating in the dark instead of a cafe, which
    is the whole complaint the outline exists to answer.
    """

    def test_walls_start_empty(self):
        self.assertEqual(build_draft(layout()).walls, ())

    def test_walls_survive_an_edit(self):
        plan = build_draft(layout())
        payload = {"seats": [], "chairs": [], "landmarks": [],
                   "walls": [[0.0, 0.0], [10.0, 0.0], [10.0, 20.0]]}
        _, updated = apply_edits(layout(), plan, payload)
        self.assertEqual(updated.walls, ((0.0, 0.0), (10.0, 0.0), (10.0, 20.0)))

    def test_walls_reach_the_page(self):
        plan = build_draft(layout())
        payload = {"seats": [], "chairs": [], "landmarks": [],
                   "walls": [[1.0, 2.0], [3.0, 4.0]]}
        _, updated = apply_edits(layout(), plan, payload)
        state = editor_state(layout(), updated, {})
        self.assertEqual(state["walls"], [[1.0, 2.0], [3.0, 4.0]])


class SeparationThroughEditsTests(unittest.TestCase):
    def test_seats_dropped_on_each_other_are_pushed_apart(self):
        # Two tables on the same spot cannot both be read, and the installer
        # dragging one onto another is an accident, not an instruction.
        plan = build_draft(layout())
        payload = {
            "seats": [
                {"seat_id": seat.seat_id, "x": 500.0, "y": 500.0,
                 "w": seat.w, "h": seat.h}
                for seat in plan.seats
            ],
            "chairs": [],
            "landmarks": [],
        }
        _, updated = apply_edits(layout(), plan, payload)
        first, second = updated.seats
        self.assertTrue(
            abs(first.x - second.x) >= first.w or abs(first.y - second.y) >= first.h
        )


if __name__ == "__main__":
    unittest.main()
