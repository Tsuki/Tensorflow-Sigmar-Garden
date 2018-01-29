from __future__ import print_function

import os

import numpy as np
from PIL import Image

from Marble import Marble
from State import State, neighbors, free, step, frees
from testing2 import predict_img
from utils import img_pos, edges_at, FIELD_POSITIONS


def init_image(img):
    status = State()
    images = np.zeros((len(FIELD_POSITIONS), 33, 33, 1))
    idx = 0
    for pos in FIELD_POSITIONS:
        images[idx, :, :, 0] = np.array(edges_at(img, *img_pos(*pos))).reshape(33, 33)
        idx = idx + 1
    for i, value in enumerate(list(predict_img(images))):
        if value == 0:
            continue
        status.state[FIELD_POSITIONS[i]] = Marble.symbol(Marble(value))
    status.update_key_dict()
    return status


def main():
    status = init_image(Image.open(os.path.join("sample", "1.png")).convert('LA'))
    # status.solve()
    print(status)
    print("neighbors")
    print(neighbors(status, 0, 0))
    print("free")
    print(free(status, 1, 0))
    print("frees")
    print(frees(status))
    print("step")
    print(list(step(status)))
    pass


if __name__ == '__main__':
    main()
