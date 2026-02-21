import math
from shapely.geometry import Point as ShapelyPoint
from src.components.geometry.point_2d import Point2D
from src.components.geometry.point_3d import Point3D


class GeometryOps:

    @classmethod
    def project(cls, vv: Point2D | Point3D, sin_a: float, cos_a: float):
        return vv.x * cos_a + vv.y * sin_a

    @classmethod
    def offset_coords(cls, vv: Point2D | Point3D, vv1: Point2D | Point3D):
        dx = vv.x - vv1.x  # Along façade
        dy = vv.y - vv1.y  # Perpendicular to façade
        return [dx, dy]

    @classmethod
    def projection_dist(cls, vv: Point2D | Point3D, vv1: Point2D | Point3D, angle):
        # Unit vector perpendicular to direction_angle
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Project both points onto the perpendicular direction
        # Point 1 projection
        proj1 = cls.project(vv, sin_a, cos_a)
        proj2 = cls.project(vv1, sin_a, cos_a)

        # Window width is the distance between projections
        return abs(proj2 - proj1)

    @classmethod
    def rotate_vertex(cls, vv: Point2D | Point3D, sin_a: float, cos_a: float):
        rot_x = cls.rotate_coord(vv, sin_a, cos_a)
        rot_y = cls.rotate_coord(vv, sin_a, cos_a, False)
        return (rot_x, rot_y)

    @classmethod
    def rotate_coord(cls, vv: Point2D | Point3D, sin_a: float, cos_a: float, x_axis=True):
        if x_axis:
            return vv.x * cos_a - vv.y * sin_a
        return vv.x * sin_a + vv.y * cos_a

    @classmethod
    def perpendicular_dir_inside_polygon(cls, room_poly, edge_coords, perp) -> bool:
        test_offset = 0.1
        edge_center_x = (edge_coords[0][0] + edge_coords[1][0]) * 0.5
        edge_center_y = (edge_coords[0][1] + edge_coords[1][1]) * 0.5
        test_x1 = edge_center_x + test_offset * math.cos(perp)
        test_y1 = edge_center_y + test_offset * math.sin(perp)
        test_point1 = ShapelyPoint(test_x1, test_y1)
        return room_poly.contains(test_point1)

    @classmethod
    def normalize_angle(cls, angle):
        # Normalize to [0, 2π)
        while angle < 0:
            angle += 2 * math.pi
        while angle >= 2 * math.pi:
            angle -= 2 * math.pi
        return angle
