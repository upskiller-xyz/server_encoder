"""Test direction_angle calculation fix - windows should face outward."""

import pytest
import math
from src.components.geometry import WindowGeometry, RoomPolygon


class TestDirectionAngleFix:
    """Test that direction_angle points outward from room, not inward."""

    def test_window_on_east_wall_faces_east(self):
        """
        Window on east wall (x=0) should face east (0° / 0 rad).

        Room: [[0, 0], [0, 7], [-3, 7], [-3, 0]]
        East wall is the edge from [0, 0] to [0, 7] at x=0
        Window should face outward (east, positive X direction) = 0°
        """
        room_polygon = RoomPolygon.from_dict([[0, 0], [0, 7], [-3, 7], [-3, 0]])

        # Window on east wall (x=0)
        window = WindowGeometry(
            x1=0, y1=1,
            x2=0, y2=2,
            z1=1.0, z2=2.0
        )

        direction_angle = window.calculate_direction_from_polygon(room_polygon)
        direction_degrees = direction_angle * 180 / math.pi

        # Should be 0° (facing east)
        assert abs(direction_degrees - 0.0) < 1.0, (
            f"Window on east wall should face east (0°), got {direction_degrees:.2f}°"
        )

    def test_window_on_west_wall_faces_west(self):
        """
        Window on west wall (x=-3) should face west (180° / π rad).
        """
        room_polygon = RoomPolygon.from_dict([[0, 0], [0, 7], [-3, 7], [-3, 0]])

        # Window on west wall (x=-3)
        window = WindowGeometry(
            x1=-3, y1=1,
            x2=-3, y2=2,
            z1=1.0, z2=2.0
        )

        direction_angle = window.calculate_direction_from_polygon(room_polygon)
        direction_degrees = direction_angle * 180 / math.pi

        # Should be 180° (facing west)
        assert abs(direction_degrees - 180.0) < 1.0, (
            f"Window on west wall should face west (180°), got {direction_degrees:.2f}°"
        )

    def test_window_on_north_wall_faces_north(self):
        """
        Window on north wall (y=7) should face north (90° / π/2 rad).
        """
        room_polygon = RoomPolygon.from_dict([[0, 0], [0, 7], [-3, 7], [-3, 0]])

        # Window on north wall (y=7)
        window = WindowGeometry(
            x1=-1, y1=7,
            x2=-2, y2=7,
            z1=1.0, z2=2.0
        )

        direction_angle = window.calculate_direction_from_polygon(room_polygon)
        direction_degrees = direction_angle * 180 / math.pi

        # Should be 90° (facing north)
        assert abs(direction_degrees - 90.0) < 1.0, (
            f"Window on north wall should face north (90°), got {direction_degrees:.2f}°"
        )

    def test_window_on_south_wall_faces_south(self):
        """
        Window on south wall (y=0) should face south (270° / 3π/2 rad).
        """
        room_polygon = RoomPolygon.from_dict([[0, 0], [0, 7], [-3, 7], [-3, 0]])

        # Window on south wall (y=0)
        window = WindowGeometry(
            x1=-1, y1=0,
            x2=-2, y2=0,
            z1=1.0, z2=2.0
        )

        direction_angle = window.calculate_direction_from_polygon(room_polygon)
        direction_degrees = direction_angle * 180 / math.pi

        # Should be 270° (facing south)
        assert abs(direction_degrees - 270.0) < 1.0, (
            f"Window on south wall should face south (270°), got {direction_degrees:.2f}°"
        )

    def test_example_from_user(self):
        """Test the exact example from the user's question."""
        room_polygon = RoomPolygon.from_dict([[0, 0], [0, 7], [-3, 7], [-3, 0]])

        # User's window (fixed to be on edge)
        window = WindowGeometry(
            x1=0, y1=1,
            x2=0, y2=1.8,
            z1=10.9, z2=11.9
        )

        direction_angle = window.calculate_direction_from_polygon(room_polygon)
        direction_degrees = direction_angle * 180 / math.pi

        print(f"\nUser's example:")
        print(f"  Direction angle: {direction_angle:.4f} rad")
        print(f"  Direction angle: {direction_degrees:.2f}°")

        # Should be 0° (facing east), NOT 180° (π)
        assert abs(direction_degrees - 0.0) < 1.0, (
            f"Expected 0° (east), got {direction_degrees:.2f}°"
        )
