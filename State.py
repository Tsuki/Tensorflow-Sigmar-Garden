import sys

from main import FIELD_POSITIONS, FIELD_SIZE


class State:
    # state = dict.fromkeys([e.name for e in Marble], ())
    state = dict()

    def __str__(self):
        py = -1
        for (x, y) in FIELD_POSITIONS:
            if y != py:
                if y > 0: sys.stdout.write('\n')
                sys.stdout.write(" " * abs(y - FIELD_SIZE + 1))
            sys.stdout.write(self.state.get((x, y), "-"))
            sys.stdout.write(' ')
            py = y
        return ''
