"""
Stage 4: Virtual Tripwire Line-Crossing & Counting Engine.
Calculates directional vector intersections for surveillance boundaries and gates.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from cctv_pipeline.config.settings import TripwireConfig


def ccw(A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int]) -> bool:
    """Tests whether three 2D points are listed in a counter-clockwise orientation."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def line_segments_intersect(
    A: Tuple[int, int], B: Tuple[int, int],
    C: Tuple[int, int], D: Tuple[int, int]
) -> bool:
    """Returns True if line segment AB and line segment CD intersect."""
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))


def resolve_tripwire_coords(
    tripwires: List[TripwireConfig],
    width: int,
    height: int
) -> List[dict]:
    """
    Converts tripwire coordinates from normalized ratios (0.0-1.0) or absolute pixels
    into concrete pixel coordinates for the current frame dimensions.
    """
    resolved = []
    for wire in tripwires:
        x1 = int(wire.pt1[0] * width) if wire.pt1[0] <= 1.0 else int(wire.pt1[0])
        y1 = int(wire.pt1[1] * height) if wire.pt1[1] <= 1.0 else int(wire.pt1[1])
        x2 = int(wire.pt2[0] * width) if wire.pt2[0] <= 1.0 else int(wire.pt2[0])
        y2 = int(wire.pt2[1] * height) if wire.pt2[1] <= 1.0 else int(wire.pt2[1])
        resolved.append({
            "name": wire.name,
            "pt1": (x1, y1),
            "pt2": (x2, y2),
            "color": tuple(wire.color),
        })
    return resolved


class TripwireCounter:
    """
    Maintains directional crossing counts across one or more virtual tripwires.
    Prevents duplicate double-counting using a debounce history set.
    """

    def __init__(self, tripwires: List[TripwireConfig]):
        self.tripwires = tripwires
        self.counts: Dict[str, int] = {wire.name: 0 for wire in tripwires}
        self.counted_events: Set[Tuple[str, int]] = set()

    def reset(self) -> None:
        """Resets all crossing counts and debounce sets."""
        self.counts = {wire.name: 0 for wire in self.tripwires}
        self.counted_events.clear()

    def update(
        self,
        active_tracks: List[dict],
        trajectories: Dict[int, List[Tuple[int, int]]],
        frame_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, int]:
        """
        Evaluates the latest movement step of each active track against every virtual tripwire.
        """
        width, height = frame_size if frame_size else (1, 1)

        for trk in active_tracks:
            tid = trk["track_id"]
            pts = trajectories.get(tid, [])
            if len(pts) < 2:
                continue

            # Latest displacement vector: from previous centroid to current centroid
            p_prev = pts[-2]
            p_curr = pts[-1]

            for wire in self.tripwires:
                event_key = (wire.name, tid)
                if event_key in self.counted_events:
                    continue

                x1 = int(wire.pt1[0] * width) if (frame_size and wire.pt1[0] <= 1.0) else int(wire.pt1[0])
                y1 = int(wire.pt1[1] * height) if (frame_size and wire.pt1[1] <= 1.0) else int(wire.pt1[1])
                x2 = int(wire.pt2[0] * width) if (frame_size and wire.pt2[0] <= 1.0) else int(wire.pt2[0])
                y2 = int(wire.pt2[1] * height) if (frame_size and wire.pt2[1] <= 1.0) else int(wire.pt2[1])

                w1 = (x1, y1)
                w2 = (x2, y2)

                if line_segments_intersect(p_prev, p_curr, w1, w2):
                    self.counts[wire.name] += 1
                    self.counted_events.add(event_key)

        return self.counts.copy()
