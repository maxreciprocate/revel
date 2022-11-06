from lang import Language, Term, Type, maketree, evalg, length, grepr
from render.draw import move, neg, down, up, loop, savex, stagebrushes
from render.roll import rolled
from dataclasses import dataclass
import numpy as np
from data import T, Term, Type
from blob import iimshow
from time import time
from tqdm import trange

@dataclass
class Beam:
    x: np.ndarray
    error: float = np.inf
    tree: T = None

def explore(L, xs, beams, number_range):
    canvas = np.zeros((1, *xs[0].shape), np.int8)
    cerrors = np.zeros(len(xs), np.int64)
    errors = np.full(cerrors.shape, (10000,))

    trees = [None for _ in range(len(xs))]
    lengths = [0 for _ in range(len(xs))]

    c = 0
    tbar = trange(*number_range)
    print_every = ((number_range[1] - number_range[0]) // 10)
    stime = time()

    for n in tbar:
        c += 1
        tree = maketree(L, L.type, n)

        canvas.fill(0)
        evalg(L, (), tree)(canvas[0])
        clength = length(tree)

        np.sum(np.abs(xs - canvas), axis=(1, 2), out=cerrors)

        for ix in np.where(cerrors < errors)[0]:
            trees[ix] = tree
            lengths[ix] = clength
            errors[ix] = cerrors[ix]

        for ix in np.where(cerrors == errors)[0]:
            if clength < lengths[ix]:
                trees[ix] = tree
                lengths[ix] = clength
                errors[ix] = cerrors[ix]

        if n % print_every == 0:
            print(f'{grepr(L, tree)} {c / (time() - stime):.0f}/s')

    for ix, solution in enumerate(beams):
        solution.tree = trees[ix]
        solution.error = errors[ix]

    return beams

