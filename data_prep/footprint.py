"""Ground footprint of a nadir UAV image and of its north-up square crop.

GSD_uav (m/px) = rel_alt * sensor_width_mm / focal_mm / image_width_px
Footprint W x H (m) = GSD_uav * (w, h)
North-up crop: take the centred square of side min(w, h) px, rotate it by yaw (expand=True), then keep
the largest axis-aligned square inscribed in the rotated square: side * 1 / (|cos t| + |sin t|).
"""
import math


def gsd_uav(rel_alt, sensor_w_mm, focal_mm, width_px):
    return rel_alt * sensor_w_mm / focal_mm / width_px


def footprint_m(rel_alt, sensor_w_mm, focal_mm, w, h):
    g = gsd_uav(rel_alt, sensor_w_mm, focal_mm, w)
    return g * w, g * h


def inscribed_square_factor(angle_deg):
    t = math.radians(angle_deg)
    return 1.0 / (abs(math.cos(t)) + abs(math.sin(t)))


def north_up_side(rel_alt, sensor_w_mm, focal_mm, w, h, yaw_deg):
    """Returns (side_px in the original image, side_m) of the north-up inscribed square."""
    g = gsd_uav(rel_alt, sensor_w_mm, focal_mm, w)
    side_px = min(w, h) * inscribed_square_factor(yaw_deg)
    return side_px, side_px * g
