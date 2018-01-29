import sys
from copy import deepcopy, copy
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

    def __init__(self, state=None):
        if state is not None:
            self.state = state

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
    for (x, y) in self:
        if free(self, x, y):
            result.append((x, y))
    return result


def free(self, x, y):
    neg = repeat(neighbors(self, x, y), 2)
    xs = np.hstack(neg)
    return ('-', '-', '-') in list(zip(*(xs[i:] for i in range(3))))


def neighbors(self, x, y):
    result = []
    for (dx, dy) in [(0, -1), (1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1)]:
        n = (x + dx, y + dy)
        if n in self:
            result.append(self[n])
        else:
            result.append('-')
    return result


def score(self):
    return len(self)


def solve(status):
    todo = [status]
    solutions = {str(status.state): []}
    while len(solutions) > 0:
        # eval
        cur_state = sorted(todo, key=lambda x: len(x.state))[0]
        todo.remove(cur_state)
        for _step in step(cur_state.state, status.keyState):
            state = deepcopy(cur_state.state)
            for pos in _step:
                if pos in state:
                    state.pop(pos)
                if str(state) in solutions:
                    continue
                todo += [State(state)]
                solution = copy(solutions[str(cur_state.state)])
                solution += _step
                solutions[str(state)] = solution
                if len(state) == 0:
                    print(solutions)
                    return solution
    return None


def step(self, keyState):
    # for Quintessence use
    buckets = {}
    _frees = sorted([(Marble[MARBLE_BY_SYMBOL[self[x]]], x) for x in frees(self)])
    for (k, v) in _frees:
        buckets.setdefault(k, []).append(v)
    for a in _frees:
        (marbleA, posA) = a
        for b in _frees:
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
                if marbleB == Marble.Quicksilver and marbleA.previous() not in keyState:
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
