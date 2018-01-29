import sys
from copy import deepcopy
from itertools import *
from random import shuffle

import numpy as np

from Marble import Marble
from Parameters import FIELD_SIZE
from utils import FIELD_POSITIONS

MARBLE_BY_SYMBOL = dict(zip([Marble.symbol(e) for e in Marble], [e.name for e in Marble]))


class State:
    # state = dict.fromkeys([e.name for e in Marble], ())
    state = dict()

    def __init__(self, state):
        self.state = state

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
    # https://stackoverflow.com/questions/9001509/how-can-i-sort-a-dictionary-by-key
    # collections.OrderedDict(sorted(d.items()))
    todo = [status]
    solutions = {status: []}
    while len(todo) > 0:
        # increase to fast
        cur_state = sorted(todo, key=lambda x: len(x.state))[0]
        solutions.update({cur_state: []})
        todo.remove(cur_state)
        # print((len(todo), len(cur_state.state)))
        for _step in step(cur_state.state):
            state = State(cur_state.state)
            for pos in _step:
                if pos in state.state:
                    state.state.pop(pos)
                if state in solutions:
                    continue
                todo = todo + [deepcopy(state)]
                solution = solutions[cur_state]
                solution += _step
                solutions[cur_state] = solution
                if len(state.state) == 1:
                    print(state)
                if len(state.state) == 0:
                    print(solution)
                    return solution
    return None


def step(self):
    # for Quintessence use
    buckets = {}
    _frees = sorted([(Marble[MARBLE_BY_SYMBOL[self[x]]], x) for x in frees(self)])
    shuffle(_frees)
    for (k, v) in _frees:
        buckets.setdefault(k, []).append(v)
    for a in _frees:
        (marbleA, posA) = a
        for b in _frees:
            (marbleB, posB) = b
            if a == b:
                continue
            elif marbleA.value in range(Marble.Vitae.value, Marble.Mors.value + 1):
                if marbleB.value in range(Marble.Vitae.value, Marble.Mors.value + 1) and marbleA != marbleB:
                    yield {posA, posB}

            elif marbleA.value in range(Marble.Lead.value, Marble.Silver.value + 1):
                if marbleA == Marble.Gold:
                    yield {posA}
                #  break if state.each_value.any? { |m| (Marble::Lead...ma) === m }
                elif marbleB == Marble.Quicksilver and Marble.symbol(marbleA.previous()) not in self.values():
                    yield {posA, posB}

            elif marbleA.value in range(Marble.Salt.value, Marble.Earth.value + 1):
                if marbleB == marbleA or marbleB == Marble.Salt:
                    yield {posA, posB}

            elif marbleA == Marble.Quintessence:
                continue

            # elif marbleA == Marble.Quicksilver:
            #     continue
            # else:
            #     print(marbleA, marbleB)
            #     continue
