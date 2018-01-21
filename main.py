import random
from enum import Enum
import os

import itertools

import sys
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from Marble import Marble

# Parameters
from State import State
from utils import field_positions, pixels_to_scan, img_pos, edges_at

learning_rate = 0.1
num_steps = 1000
batch_size = 128
display_step = 100

FIELD_X = 1052
FIELD_DX = 66
FIELD_Y = 221
FIELD_DY = 57
FIELD_SIZE = 6

SCAN_RADIUS = 17
MARBLE_BY_SYMBOL = dict(zip([Marble.symbol(e) for e in Marble], [e.name for e in Marble]))

FIELD_POSITIONS = field_positions()
PIXELS_TO_SCAN = pixels_to_scan()

TRAIN_CASES = dict.fromkeys([e.name for e in Marble], [])


def sample():
    for i in range(1, 7):
        img = Image.open(os.path.join("sample", str(i) + ".png")).convert('LA')
        samples = list(itertools.chain.from_iterable(
            [lines.split() for lines in open(os.path.join("sample", str(i) + ".txt"), "r").readlines()]))
        for j, (pos, symbol) in enumerate(zip(FIELD_POSITIONS, samples)):
            marble = MARBLE_BY_SYMBOL[symbol]
            edge_pixels = set(edges_at(img, *img_pos(*pos)))
            TRAIN_CASES[marble] = TRAIN_CASES[marble] + [edge_pixels]


def train():
    marble = random.choice([e.name for e in Marble])
    edge_pixels = random.choice(TRAIN_CASES[marble])
    a = list(map(lambda x: 1.0 if x in edge_pixels else 0.0, PIXELS_TO_SCAN))
    b = list(map(lambda x: 1.0 if marble is x else 0.0, [e.name for e in Marble]))


def initMap(img):
    status = State()
    for pos in FIELD_POSITIONS:
        try_edges = edges_at(img, *img_pos(*pos))
        # result = ANN.run(list(map(lambda x: 1.0 if x in try_edges else 0.0, PIXELS_TO_SCAN)))
        # marble = sorted(list(zip(result, [e.name for e in Marble])), reverse=True)[0]
        # status.state[pos] = Marble.symbol(Marble[marble[1]])
    print(status)


def init():
    input_y = [len(PIXELS_TO_SCAN)]
    hidden_y = [int(len(PIXELS_TO_SCAN) / 2), int(len(PIXELS_TO_SCAN) / 4)]
    output_y = [len(Marble)]
    layer = input_y + hidden_y + output_y
    if os.path.exists("network.fann"):
        print("Load Network from network.fann")
        # ANN.create_from_file("network.fann")
    else:
        print("Train Network")
        sample()
        # ANN.create_standard_array(layer)
        train()
        # ANN.save("network.fann")


def main():
    # print(Marble.symbol(Marble.Fire))
    init()
    initMap(Image.open(os.path.join("sample", "1.png")).convert('LA'))
    # print(pixels_to_scan())
    # print(field_positions())
    # print(img_pos(1, 1))
    pass


if __name__ == '__main__':
    main()
