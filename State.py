import sys
from itertools import *

import numpy as np

from Marble import Marble
from Parameters import FIELD_SIZE
from utils import FIELD_POSITIONS

MARBLE_BY_SYMBOL = dict(zip([Marble.symbol(e) for e in Marble], [e.name for e in Marble]))


class State:
    # state = dict.fromkeys([e.name for e in Marble], ())
    state = dict()

    def __str__(self):
        py = -1
        for (x, y) in FIELD_POSITIONS:
            if y != py:
                if y > 0:
                    sys.stdout.write('\n')
                sys.stdout.write(" " * abs(y - FIELD_SIZE + 1))
            sys.stdout.write(self.state.get((x, y), "-"))
            sys.stdout.write(' ')
            py = y
        return ''

    def frees(self):
        result = []
        for (x, y) in self.state:
            if self.free(x, y):
                result.append((x, y))
        return result

    def free(self, x, y):
        neg = repeat(self.neighbors(x, y), 2)
        xs = np.hstack(neg)
        return ('-', '-', '-') in list(zip(*(xs[i:] for i in range(3))))

    def neighbors(self, x, y):
        result = []
        for (dx, dy) in [(0, -1), (1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1)]:
            n = (x + dx, y + dy)
            if n in self.state:
                result.append(self.state[n])
            else:
                result.append('-')
        return result

    def step(self):
        free = {}
        frees = [(MARBLE_BY_SYMBOL[self.state[x]], x) for x in self.frees()]
        for (k, v) in frees:
            free.setdefault(k, []).append(v)
