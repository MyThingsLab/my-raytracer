from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Vec3:
        return self.__mul__(scalar)

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length_squared(self) -> float:
        # Ray/sphere intersection and nearest-hit comparisons only need the
        # squared magnitude; exposing it lets hot paths skip the sqrt.
        return self.dot(self)

    def length(self) -> float:
        return math.sqrt(self.length_squared())

    def normalized(self) -> Vec3:
        # Raise on zero-length: a silent zero-vector fallback would hide
        # degenerate geometry (e.g. coincident ray points) instead of surfacing it.
        length = self.length()
        if length == 0.0:
            raise ValueError("cannot normalize a zero-length vector")
        return Vec3(self.x / length, self.y / length, self.z / length)
