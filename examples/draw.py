from blob import *
from numba import njit

TAIL = 0x1
ONSET = 0x2

MOVE = 0x0
NEGMOVE = 0x1
UPMOVE = 0x2
DOWNMOVE = 0x3
SAVESTATE = 0x4
RESUMESTATE = 0x5
PENUP = 0x6
PENDOWN = 0x7
ROTATE = 0x8

def move(distance, next):
    return ((MOVE, distance),) + next

def neg(distance, next):
    return ((NEGMOVE, distance),) + next

def up(distance, next):
    return ((UPMOVE, distance),) + next

def down(distance, next):
    return ((DOWNMOVE, distance),) + next

def loop(n, brushes, next):
    return brushes * n + next

def savex(brushes, next):
    return ((SAVESTATE, SAVESTATE),) + brushes + ((RESUMESTATE, RESUMESTATE),) + next

def rotate(angle, next):
    return ((ROTATE, angle),) + next

def penup(brushes, next):
    return ((PENUP, PENUP),) + brushes + ((PENDOWN, PENDOWN),) + next

@njit
def bresenham(canvas, x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    for x in range(dx):
        x_ = (x0 + x*xx + y*yx) % canvas.shape[0]
        y_ = (y0 + x*xy + y*yy) % canvas.shape[1]
        canvas[x_, y_] = TAIL

        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy

defaultstate = (1, 0, 0, 0)

@njit
def render(brushes: np.ndarray, canvas: np.ndarray, state=defaultstate):
    isdrawing, x, y, angle = state
    saved_x, saved_y, saved_angle = 0, 0, 0
    bound_y, bound_x = canvas.shape

    for op, arg in brushes:
        if op == MOVE:
            end_x = x + np.cos(angle) * arg
            end_y = y + np.sin(angle) * arg

            if isdrawing:
                bresenham(canvas, round(x), round(y), round(end_x), round(end_y))

            x = round(end_x)
            y = round(end_y)

        if op == PENDOWN:
            isdrawing = 1
        elif op == PENUP:
            isdrawing = 0
        elif op == ROTATE:
            # temporary restriction on keeping everything int
            # instead of casting every instruction to float
            if arg == 0:
                angle -= np.pi / 2
            elif arg == 1:
                angle += np.pi / 2
        elif op == SAVESTATE:
            saved_x = x
            saved_y = y
            saved_angle = angle
        elif op == RESUMESTATE:
            x = saved_x
            y = saved_y
            angle = saved_angle

    return isdrawing, x, y, angle

def stage(brushes):
    if len(brushes) == 0 or len(brushes[0]) == 0:
        return lambda *args: args[-1]

    return partial(render, np.array(brushes))

if __name__ == '__main__':
    # line = penup(move(8, []), move(8, []))
    # brushes = savex(rotate(pi/6, line), rotate(pi/3, line))
    # brushes = savex(rotate(1, line), line)
    # brushes = move(2, rotate(1, move(2, (rotate(0, move(2, []))))))

    brushes = np.array(move(4, []))
    canvas = zeros((10, 10), int)
    state = render(brushes, canvas, (1, 16, 4, 0))
    # starshow(canvas)
    iimshow(canvas)

# ■ ~

    brushes = loop(2, move(3, []), rotate(1, move(4, rotate(1, move(4, [])))))

    brushes = np.array(brushes)

    canvas = zeros((16, 16), int)
    stime = time()
    stage(brushes)(canvas)
    print(f'first {(time() - stime)*1000:.6f}ms')

    stime = time()
    stage(brushes)(canvas)
    print(f'jit-ed {(time() - stime)*1000:.6f}ms')

    starshow(canvas)
