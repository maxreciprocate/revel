import os
from lang import Language, maketree, evaltree, length, grepr, parse, tolang, normalize
from render.draw import *
from render.roll import rolled
from dataclasses import dataclass
import numpy as np
from data import T, Term, Type
from blob import iimshow
from time import time
from tqdm import trange
from copy import deepcopy
import multiprocessing as mp
from render.font import renderalphabet, starshow

@dataclass
class Solution:
    x: np.ndarray
    xhat: np.ndarray = None
    error: float = np.inf
    tree: T = None

def explore(L, X, number_range, solutions=None):
    if solutions is None:
        solutions = [Solution(x) for x in X]

    canvas = np.zeros((1, *X[0].shape), np.int8)
    cerrors = np.zeros(len(X), np.int64)
    errors = np.full(cerrors.shape, (10000,))

    trees = [None for _ in range(len(X))]
    lengths = [0 for _ in range(len(X))]

    c = 0
    tbar = trange(*number_range, disable=mp.current_process().name[-1] != '1')
    print_every = ((number_range[1] - number_range[0]) // 20)
    stime = time()

    for n in tbar:
        c += 1
        tree = maketree(L, L.type, n)

        canvas.fill(0)
        evaltree(L, tree)(canvas[0])
        clength = length(tree)

        np.sum(np.abs(X - canvas), axis=(1, 2), out=cerrors)

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
            tbar.set_postfix({'tree': L<tree.tails[0]})

    for ix, solution in enumerate(solutions):
        solution.tree = trees[ix]
        canvas.fill(0)
        evaltree(L, solution.tree)(canvas[0])
        solution.xhat = np.copy(canvas[0])
        solution.error = np.sum(np.abs(solution.x - solution.xhat))

    return solutions

def plotsolutions(L, solutions, epoch=0):
    canvas = zeros(solutions[0].x.shape, np.int8)

    N = len(solutions)
    fig, axs = pyplot.subplots(N, 1, figsize=(32, N * 4))
    totalerror = sum(s.error for s in solutions)

    for s, ax in zip(solutions, axs):
        diff = s.x - s.xhat
        nnotcovered = len(diff[diff > 0])
        nredundant = len(diff[diff < 0])
        npixels = len(s.x[s.x > 0])

        xi, yi = tuple(map(np.max, s.x.nonzero()))
        padx = (s.x.shape[0] - xi) // 2
        pady = (s.x.shape[1] - yi) // 2

        x = np.roll(s.x, (padx, pady), axis=(0, 1))
        xhat = np.roll(s.xhat, (padx, pady), axis=(0, 1))
        im = np.hstack((xhat, np.ones((s.x.shape[0], 4)), x))

        ax.imshow(im, interpolation=None, cmap='hot', vmin=-1, vmax=1)
        ax.grid(None)
        ax.axis('off')
        ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)

        title = f"covered {npixels - nnotcovered}/{npixels}, extra {nredundant}, total {s.error}"
        ax.set_title(f"{title}\n{grepr(L, s.tree)}", size=20, y=1.15)

    suptitle = f'Epoch #{epoch} Total error = {totalerror}'
    fig.suptitle(suptitle, size=20, y=0.9)
    print(suptitle)

    pyplot.subplots_adjust(left=0.0, right=1, hspace=0.7)
    savefig(f"stash/{epoch}.png")

def convert(L: Language, ast: Union[List, str], tailtypes=None, level=0):
    "Replace debruijns with integer references (e.g. $1 -> -2) and infer their types for the most outer lambdas"
    if not isinstance(ast, list):
        return ast, None, None

    if ast[0] == 'λ':
        return convert(L, ast[1], tailtypes, level + 1)

    if tailtypes is None: # done with outer lambdas
        tailtypes = [None] * level
        level = 0

    if ast[0] == '@':
        return convert(L, ast[1:], tailtypes, level) # ignores hof

    for ix, tree in enumerate(ast[1:]):
        if isinstance(tree, str) and tree.startswith('$'):
            index = int(tree[1:])
            if index >= level: # otherwise it's referencing other lambda
                index = -(index+1)
                tailtypes[index] = Type(L[ast[0]].tailtypes[ix].type)
                ast[ix+1] = str(index)
        else:
            ast[ix+1] = convert(L, tree, tailtypes, level)[0]

    # ignores hof returns
    return ast, L[ast[0]].type, tailtypes

def compress(L, trees):
    as_string = " ".join(L<tree for tree in trees)
    open('src_expr', 'w').write(as_string)
    print(os.popen('babble-this/target/release/babble-this').read())
    s = open('target/rec_expr').read()
    print(s)
    ast = parse(s)[0]

    L.reset()
    queue = [ast]
    newtrees = []
    libraries = []
    while queue:
        tree = queue.pop(0)
        match tree:
            case ['lib', name, binding, rest]:
                libraries.append([name, binding])
                queue.append(rest)
            case ['list', *trees]:
                newtrees.extend(trees)

    blocking = set()
    while libraries: # why aren't these in at least in some order?
        name, binding = libraries.pop(0)
        try:
            binding, type, tailtypes = convert(L, deepcopy(binding))
        except Exception:
            print(f'{binding} references not yet learned function')
            if name in blocking:
                break

            blocking.add(name)
            libraries.append((name, binding))
            continue

        print(name, type, tailtypes, binding)
        binding = tolang(L, binding)
        if not tailtypes:
            binding = evaltree(L, binding)

        term = Term(binding, type, tailtypes, repr=name)
        L.add(term)

    L.update_types()
    return newtrees

# ■ ~

Int = Type('Int')
Float = Type('Float')
Brush = Type('Brush')
Render = Type('Render')

maxnum = 8
L = Language([
    Term([], Brush, repr='ø'),
    *[Term(n, Int, repr=repr(n)) for n in range(maxnum+1)],
    *[Term(i, Float, repr=repr) for i, repr in enumerate(['-pi/2', 'pi/2'])],
    Term(move,
         type=Brush,
         tailtypes=[Int, Brush],
         forbidden=[['0'], ['move']],
         repr='move'),
    Term(rotate,
         type=Brush,
         tailtypes=[Float, Brush],
         forbidden=[[], ['rotate']],
         repr='rotate'),
    Term(penup,
         type=Brush,
         tailtypes=[Brush, Brush],
         forbidden=[['penup', 'ø'], ['penup']],
         repr='penup'),
    Term(stage,
         type=Render,
         tailtypes=[Brush],
         repr='render'),
], type=Render)

X = renderalphabet('geo/Geo-Regular.ttf', (13, 13))
size = 10**6

if __name__ == '__main__':
    for epoch in range(5):
        ranges = [(ix*size, ((ix+1)*size)) for ix in range(ncores)]
        solutions, *manysolutions = multicore(explore, zip(repeat(L), repeat(X), ranges))

        for bsolutions in manysolutions:
            for ix, (s1, s2) in enumerate(zip(solutions, bsolutions)):
                if s2.error < s1.error or (s2.error == s1.error and length(s2.tree) < length(s1.tree)):
                    solutions[ix] = s2

        trees = [s.tree.tails[0] for s in solutions]
        normalized = [normalize(L, tree) for tree in trees]

        for t, nt in zip(trees, normalized):
            assert evaltree(L, t) == evaltree(L, nt), f"{t} != {nt}"

        trees = compress(L, normalized)

        for s, tree in zip(solutions, trees):
            try: # fix hof
                tree = tolang(L, convert(L, tree)[0])
            except Exception as e:
                print(e)

            print(tree)
            s.tree = tree

        plotsolutions(L, solutions, epoch=epoch)

