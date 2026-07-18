from __future__ import annotations

import math
from dataclasses import dataclass, field

from myraytracer.geometry import Hit, Plane, Quad, Sphere
from myraytracer.ray import Ray
from myraytracer.vec import Vec3

Primitive = Sphere | Plane | Quad
Bounded = Sphere | Quad

# Stop subdividing once a node holds this few primitives; below this size a
# linear scan of the leaf is cheaper than further tree traversal.
_LEAF_SIZE = 4


@dataclass(frozen=True)
class AABB:
    min: Vec3
    max: Vec3

    def union(self, other: AABB) -> AABB:
        return AABB(
            Vec3(
                min(self.min.x, other.min.x),
                min(self.min.y, other.min.y),
                min(self.min.z, other.min.z),
            ),
            Vec3(
                max(self.max.x, other.max.x),
                max(self.max.y, other.max.y),
                max(self.max.z, other.max.z),
            ),
        )

    def surface_area(self) -> float:
        extent = self.max - self.min
        return 2.0 * (
            extent.x * extent.y + extent.y * extent.z + extent.z * extent.x
        )

    def centroid(self) -> Vec3:
        return (self.min + self.max) * 0.5

    def hit(self, ray: Ray, t_min: float, t_max: float) -> bool:
        # Standard slab test (Kay & Kajiya 1986): intersect the ray with each
        # axis-aligned pair of planes and shrink [t_min, t_max] to the overlap.
        for origin, direction, lo, hi in (
            (ray.origin.x, ray.direction.x, self.min.x, self.max.x),
            (ray.origin.y, ray.direction.y, self.min.y, self.max.y),
            (ray.origin.z, ray.direction.z, self.min.z, self.max.z),
        ):
            if direction == 0.0:
                if origin < lo or origin > hi:
                    return False
                continue
            inv_direction = 1.0 / direction
            t0 = (lo - origin) * inv_direction
            t1 = (hi - origin) * inv_direction
            if t0 > t1:
                t0, t1 = t1, t0
            t_min = max(t_min, t0)
            t_max = min(t_max, t1)
            if t_max <= t_min:
                return False
        return True


def bounding_box(obj: Primitive) -> AABB | None:
    """Return `obj`'s world-space AABB, or None if it is unbounded (Plane)."""
    if isinstance(obj, Sphere):
        radius = Vec3(obj.radius, obj.radius, obj.radius)
        return AABB(obj.center - radius, obj.center + radius)
    if isinstance(obj, Quad):
        corners = [
            obj.corner,
            obj.corner + obj.edge1,
            obj.corner + obj.edge2,
            obj.corner + obj.edge1 + obj.edge2,
        ]
        lo = corners[0]
        hi = corners[0]
        for corner in corners[1:]:
            lo = Vec3(min(lo.x, corner.x), min(lo.y, corner.y), min(lo.z, corner.z))
            hi = Vec3(max(hi.x, corner.x), max(hi.y, corner.y), max(hi.z, corner.z))
        # Pad a hair so a perfectly axis-aligned (zero-thickness) quad still
        # yields a non-degenerate slab on that axis.
        pad = Vec3(1e-4, 1e-4, 1e-4)
        return AABB(lo - pad, hi + pad)
    return None


def _axis_value(v: Vec3, axis: int) -> float:
    return (v.x, v.y, v.z)[axis]


@dataclass
class _BVHNode:
    box: AABB
    left: _BVHNode | None = None
    right: _BVHNode | None = None
    axis: int | None = None
    indices: list[int] | None = None

    def is_leaf(self) -> bool:
        return self.indices is not None


_Entry = tuple[int, AABB, Vec3]  # (primitive index, box, centroid)


def _build_node(entries: list[_Entry]) -> _BVHNode:
    """Top-down SAH build: at each node, pick the axis/split minimizing the
    surface-area heuristic cost sum(child_area * child_count), approximating
    expected traversal + intersection cost (Goldsmith & Salmon 1987)."""
    box = entries[0][1]
    for _, entry_box, _ in entries[1:]:
        box = box.union(entry_box)

    if len(entries) <= _LEAF_SIZE:
        return _BVHNode(box=box, indices=[index for index, _, _ in entries])

    best_cost = math.inf
    best_axis = 0
    best_split = 0
    best_order: list[_Entry] = entries

    for axis in range(3):
        order = sorted(entries, key=lambda e: _axis_value(e[2], axis))
        count = len(order)

        prefix_boxes = [order[0][1]]
        for entry in order[1:]:
            prefix_boxes.append(prefix_boxes[-1].union(entry[1]))

        suffix_boxes = [None] * count
        suffix_boxes[-1] = order[-1][1]
        for i in range(count - 2, -1, -1):
            suffix_boxes[i] = suffix_boxes[i + 1].union(order[i][1])

        for split in range(1, count):
            left_area = prefix_boxes[split - 1].surface_area()
            right_area = suffix_boxes[split].surface_area()
            cost = left_area * split + right_area * (count - split)
            if cost < best_cost:
                best_cost = cost
                best_axis = axis
                best_split = split
                best_order = order

    left = _build_node(best_order[:best_split])
    right = _build_node(best_order[best_split:])
    return _BVHNode(box=box, left=left, right=right, axis=best_axis)


def _traverse(
    node: _BVHNode, ray: Ray, t_min: float, t_max: float, primitives: list[Bounded]
) -> Hit | None:
    if not node.box.hit(ray, t_min, t_max):
        return None

    if node.is_leaf():
        closest: Hit | None = None
        closest_t = t_max
        for index in node.indices:  # type: ignore[union-attr]
            hit = primitives[index].hit(ray, t_min, closest_t)
            if hit is not None:
                closest = hit
                closest_t = hit.t
        return closest

    assert node.left is not None and node.right is not None and node.axis is not None
    # Traverse the near child first: the SAH split put lower-centroid
    # primitives in `left`, so it's the near side when the ray travels in
    # the negative direction along the split axis.
    if _axis_value(ray.direction, node.axis) < 0.0:
        near, far = node.right, node.left
    else:
        near, far = node.left, node.right

    closest = _traverse(near, ray, t_min, t_max, primitives)
    if closest is not None:
        t_max = closest.t
    far_hit = _traverse(far, ray, t_min, t_max, primitives)
    return far_hit if far_hit is not None else closest


@dataclass
class BVH:
    """SAH-built acceleration structure over bounded primitives (Sphere,
    Quad); unbounded primitives (Plane) are kept aside and tested linearly."""

    bounded: list[Bounded]
    unbounded: list[Plane]
    _root: _BVHNode | None = field(default=None)

    @staticmethod
    def build(objects: list[Primitive]) -> BVH:
        bounded: list[Bounded] = []
        unbounded: list[Plane] = []
        for obj in objects:
            if isinstance(obj, Plane):
                unbounded.append(obj)
            else:
                bounded.append(obj)

        root: _BVHNode | None = None
        if bounded:
            entries = [
                (index, box, box.centroid())
                for index, obj in enumerate(bounded)
                for box in (bounding_box(obj),)
            ]
            root = _build_node(entries)

        return BVH(bounded=bounded, unbounded=unbounded, _root=root)

    def nearest_hit(self, ray: Ray, t_min: float, t_max: float) -> Hit | None:
        closest: Hit | None = None
        closest_t = t_max

        if self._root is not None:
            hit = _traverse(self._root, ray, t_min, closest_t, self.bounded)
            if hit is not None:
                closest = hit
                closest_t = hit.t

        for obj in self.unbounded:
            hit = obj.hit(ray, t_min, closest_t)
            if hit is not None:
                closest = hit
                closest_t = hit.t

        return closest
