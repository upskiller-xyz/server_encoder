from dataclasses import dataclass
from enum import Enum


class CoordinateAxis(str, Enum):
    """Coordinate axis enumeration (Enumerator Pattern)"""
    X = "x"
    Y = "y"
    Z = "z"


@dataclass
class ReferencePointResult:
    """
    Result model for window reference point calculation

    Encapsulates a 3D reference point with proper OOP design.
    Uses enum members for coordinate keys instead of magic strings.
    """
    x: float
    y: float
    z: float

    @property
    def to_dict(self) -> dict:
        """
        Convert to dictionary with enum-based keys

        Returns:
            Dictionary mapping CoordinateAxis enum values to coordinates
        """
        return {
            CoordinateAxis.X.value: round(self.x, 4),
            CoordinateAxis.Y.value: round(self.y, 4),
            CoordinateAxis.Z.value: round(self.z, 4)
        }

    @classmethod
    def from_point(cls, point) -> "ReferencePointResult":
        """
        Factory method to create from a point object

        Args:
            point: Object with x, y, z attributes

        Returns:
            ReferencePointResult instance
        """
        return cls(x=point.x, y=point.y, z=point.z)
