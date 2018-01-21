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
            if (abs(dx) + abs(dy)) * 2 > SCAN_RADIUS * 3:
                continue
            if dy * 2 < -SCAN_RADIUS and dx * 5 < SCAN_RADIUS:
                continue
            else:
                pxs.append((dx, dy))
    return pxs


FIELD_POSITIONS = field_positions()
PIXELS_TO_SCAN = pixels_to_scan()


def img_pos(x, y):
    return FIELD_X + FIELD_DX * (x * 2 - y) / 2, FIELD_Y + FIELD_DY * y


def lightness_at(img, x, y):
    _min, _max = img.getpixel((x, y))
    return _min / _max


def edges_at(img, x, y):
    def sorting(d):
        dx, dy = d

        def neigh(dd):
            ddx, ddy = dd
            return lightness_at(img, x + dx + ddx, y + dy + ddy)

        _neigh = list(map(neigh, [(-1, 0), (0, -1), (1, 0), (0, 1)]))
        _max, _min = max(_neigh), min(_neigh)
        return -(_max - _min)

    result = sorted(PIXELS_TO_SCAN, key=sorting)
    return result[:int(len(result) / 4)]
