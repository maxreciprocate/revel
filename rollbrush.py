from importblob import *
from roll import *

MOVE = 0x3
NEGMOVE = 0x4
UPMOVE = 0x5
DOWNMOVE = 0x6
SAVESTATE = 0x7
RESUMESTATE = 0x8

def move(d, bs):
    return [(MOVE, d)] + bs

def neg(d, bs):
    return [(NEGMOVE, d)] + bs

def up(d, bs):
    return [(UPMOVE, d)] + bs

def down(d, bs):
    return [(DOWNMOVE, d)] + bs

def loop(n, b, bs):
    return b * n + bs

def savex(b, bs):
    return [(SAVESTATE, SAVESTATE)] + b + [(RESUMESTATE, RESUMESTATE)] + bs

@njit
def renderbrushes(brushes: np.ndarray, canvas: np.ndarray):
    x, y = 0, 0
    saved_x, saved_y = 0, 0
    bound_y, bound_x = canvas.shape

    for op, d in brushes:
        if op == MOVE:
            if 0 > x or x >= bound_x or 0 > y or y >= bound_y:
                return x, y

            canvas[y, x] = ONSET
            x += 1

            for _ in range(d-1):
                if 0 > x or x >= bound_x or 0 > y or y >= bound_y:
                    return x, y

                canvas[y, x] = TAIL
                x += 1

        elif op == NEGMOVE:
            x += d
        elif op == UPMOVE:
            y -= d
        elif op == DOWNMOVE:
            y += d
        elif op == SAVESTATE:
            saved_x = x
            saved_y = y
        elif op == RESUMESTATE:
            x = saved_x
            y = saved_y

    return x, y

def stage(start: int, brushes: list):
    xs = array(down(start, brushes), dtype=int)

    if len(brushes) == 0 or len(brushes[0]) == 0:
        return lambda x: x

    return partial(renderbrushes, xs)

def stagebrushes(start: int):
    return partial(stage, start)

if __name__ == '__main__':
    g = savex(move(4, neg(4, move(8, []))), loop(4, down(4, move(4, [])), []))

    canvas = zeros((20, 20), int)
    stagebrushes(0)(g)(canvas)
    iimshow(canvas)
