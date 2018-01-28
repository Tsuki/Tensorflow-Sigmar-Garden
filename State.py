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
    keyState = dict()

    def update_key_dict(self):
        for (k, v) in [(Marble[MARBLE_BY_SYMBOL[self.state[x]]], x) for x in self.state]:
            self.keyState.setdefault(k, []).append(v)
        # self.keyState = [(k, status.keyState[k]) for k in sorted(status.keyState.keys()) if k.value > 9]

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

    def score(self):
        return len(self.state)

    def solve(self):
        todo = self.state
        solutions = {str(todo): []}
        # while len(todo) == 0:
        #     continue

    def step(self):
        # for Quintessence use
        buckets = {}
        frees = sorted([(Marble[MARBLE_BY_SYMBOL[self.state[x]]], x) for x in self.frees()])
        for (k, v) in frees:
            buckets.setdefault(k, []).append(v)
        for a in frees:
            (marbleA, posA) = a
            for b in frees:
                (marbleB, posB) = b
                if a == b:
                    continue
                elif marbleA.value in range(Marble.Salt.value, Marble.Earth.value + 1):
                    if marbleB == marbleA or marbleB == Marble.Salt:
                        yield {posA, posB}
                elif marbleA.value in range(Marble.Vitae.value, Marble.Mors.value + 1):
                    if marbleB.value in range(Marble.Vitae.value, Marble.Mors.value + 1) and marbleA != marbleB:
                        yield {posA, posB}
                elif marbleA.value in range(Marble.Tin.value, Marble.Silver.value + 1):
                    if marbleB == Marble.Quicksilver and marbleA.previous() not in self.keyState:
                        yield {posA, posB}
                elif marbleA == Marble.Lead and marbleB == Marble.Quicksilver:
                    yield {posA, posB}
                elif marbleA == Marble.Gold:
                    yield {posA}
                elif marbleA == Marble.Quintessence:
                    continue
                elif marbleA == Marble.Quicksilver:
                    continue
                else:
                    print(marbleA, marbleB)
