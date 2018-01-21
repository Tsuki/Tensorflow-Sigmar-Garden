import numpy as np

from Parameters import FIELD_SIZE, SCAN_RADIUS, FIELD_X, FIELD_DX, FIELD_DY, FIELD_Y


def field_positions():
    d = FIELD_SIZE - 1
    result = []
    for y in range(-d, d + 1):
        for x in range(-d, d + 1):
            if not abs(y - x) > d:
                result.append((x + d, y + d))
    return result


def pixels_to_scan():
    pxs = []
    for dy in range(-SCAN_RADIUS + 1, SCAN_RADIUS):
        for dx in range(-SCAN_RADIUS + 1, SCAN_RADIUS):
            pxs.append((dx, dy))
    return pxs


FIELD_POSITIONS = field_positions()
PIXELS_TO_SCAN = pixels_to_scan()


def img_pos(x, y):
    return FIELD_X + FIELD_DX * (x * 2 - y) / 2, FIELD_Y + FIELD_DY * y


def lightness_at(img, x, y):
    _min, _max = img.getpixel((x, y))
    return _min / 255


def edges_at(img, x, y):
    result = []
    for (xx, yy) in PIXELS_TO_SCAN:
        result.append(lightness_at(img, x + xx, y + yy))
    return result
