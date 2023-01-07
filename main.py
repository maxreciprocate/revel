import os
import re
import pickle
from render.draw import *
from render.font import *
from enumerate import Type, Library, Term, growtree
from λ import *

@dataclass
class Solution:
    source: np.ndarray
    canvas: np.ndarray = None
    error: float = np.inf
    state: tuple = None
    length: int = np.inf
    tree: any = None

def plotsolutions(L, solutions, epoch=0):
    canvas = zeros(solutions[0].canvas.shape, np.int8)

    N = len(solutions)
    fig, axs = pyplot.subplots(N, 1, figsize=(32, N * 4), constrained_layout=True)
    totalerror = sum(s.error for s in solutions)

    for s, ax in zip(solutions, axs):
        diff = s.source - s.canvas
        nnotcovered = len(diff[diff > 0])
        nredundant = len(diff[diff < 0])
        npixels = len(s.source[s.source > 0])

        xi, yi = tuple(map(np.max, s.source.nonzero()))
        padx = (s.source.shape[0] - xi) // 2
        pady = (s.source.shape[1] - yi) // 2

        source = np.roll(s.source, (padx, pady), axis=(0, 1))
        canvas = np.roll(s.canvas, (padx, pady), axis=(0, 1))
        im = np.hstack((source, np.ones((s.source.shape[0], 4)), canvas))

        ax.imshow(im, interpolation=None, cmap='hot', vmin=-1, vmax=1)
        ax.grid(None)
        ax.axis('off')
        ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)

        title = f"covered {npixels - nnotcovered}/{npixels}, extra {nredundant}, total {s.error}"
        ax.set_title(f"{title}\n{decode(s.tree, L)}", size=20, y=1.12)

    suptitle = f'Epoch #{epoch} Total error = {totalerror}'
    fig.suptitle(suptitle, size=20)
    print(suptitle)

    pyplot.subplots_adjust(left=0.0, right=1, hspace=0.8)
    savefig(f"stash/{epoch}.png")
    pickle.dump((L, solutions), open(f"stash/{epoch}.pkl", "wb"))

def explore(L, S, numrange=(0, 10**6)):
    N = len(S)

    sources = np.array([s.source for s in S])
    saved_errors = np.array([s.error for s in S])
    saved_states = np.array([s.state for s in S])
    saved_canvas = np.array([s.canvas for s in S])

    newtrees = [None] * N
    canvas = np.empty_like(saved_canvas)
    states = np.empty_like(saved_states)
    errors = np.empty_like(saved_errors)

    tbar = trange(*numrange, disable=mp.current_process()._identity and mp.current_process()._identity[0] % ncores != 0)

    for nthtree in tbar:
        tree = growtree(L, L.type, nthtree)
        program = reduce(tree, L)

        canvas[:] = saved_canvas
        for ix in range(N):
            states[ix] = program(canvas[ix], saved_states[ix])

        np.sum(sources - canvas, axis=(1, 2), out=errors)
        errors += length(tree) * 0.25

        for ix in np.where(errors < saved_errors)[0]:
            saved_errors[ix] = errors[ix]
            newtrees[ix] = tree

    S = deepcopy(S)
    for ix, s in enumerate(S):
        if newtrees[ix] is not None:
            if s.tree is not None:
                tree = fixpoint(normalize, s.tree, L)
                mutcat(tree, newtrees[ix].xs[-1], encode("ø", L))
                newtrees[ix] = App(encode("render", L), tree)

            s.tree = newtrees[ix]
            s.length = length(s.tree)
            s.canvas = np.zeros_like(s.source)
            s.state = reduce(s.tree, L)(s.canvas)
            s.error = np.sum(s.source - s.canvas)
            print(f'[{s.error}] [{s.state}] {s.tree=}')
    return S

def mutcat(a1, a2, end):
    while a1.xs[-1] != end and a1:
        a1 = a1.xs[-1]
    a1.xs = (*a1.xs[:-1], a2)

def normalize(λ, L):
    match λ:
        case App(Lam(f), (x,)): return shift(subst(f, 0, shift(x, 1, 0)), -1, 0)
        case Lam(body): return Lam(normalize(body, L))
        case App(fn, xs): return App(normalize(fn, L), *[normalize(x, L) for x in xs])
        case Ref(ix) if ix >= len(L.primitives): return L[ix]
    return λ

def fixpoint(fn, *args, limit=1000):
    while (out := fn(*args)) != args and limit:
        args = (out, *args[1:])
        limit -= 1
    return out

def whichtype(λ, types, args=tuple(), level=0):
    match λ:
        case Lam(body): return Type(
           f'{whichtype(args[level], types)} -> {whichtype(body, types, args, level+1)}')
        case Ref(ix): return types[ix]
        case Var(ix):
            if len(args) <= ix:
                return None
            return whichtype(args[level+ix], types, args)
        case App(Lam(body), x): return whichtype(redux(λ), types, args)
        case App(fn, xs): return whichtype(fn, types, args + xs)
        case _: return None

def fixtypes(L, types, libraries, returns):
    newtypes = deepcopy(types)
    while True:
        for name, ast in libraries:
            for λ in returns.keys():
                if isinstance(λ, App) and λ.fn == encode(name, L):
                    λt = λ
                    while True:
                        try:
                            if (λx := returns[λt]) == λt:
                                break
                            if (tp := whichtype(λx, newtypes)):
                                newtypes[L.index(name)] = tp
                        except (KeyError, IndexError):
                            break
                        finally:
                            λt = λx
        if newtypes != types:
            types = newtypes
        else:
            return newtypes

def compress(L, trees):
    as_string = " ".join(decode(tree, L) for tree in trees)
    open('src_expr', 'w').write(as_string)
    print(os.popen('babble-this/target/release/babble-this').read())
    s = open('target/rec_expr').read()
    s = re.sub(r"l(\d+)", r"f\1", s)
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

    # first add new terms with empty definitions
    # just for encoding to work out of the box
    for name, ast in libraries:
        L.add(Term(None, type=None, tailtypes=[], repr=name))

    # update definitions, but not yet types
    for ix, (name, ast) in enumerate(libraries):
        term = L.invented[ix]
        L.invented[ix] = term._replace(fn=encode(ast, L))
    L.prepare()

    returns = {}
    inputs = defaultdict(list)

    def traceredux(λ, L):
        match λ:
            case App(Lam(f), (x,)): out = shift(subst(f, 0, shift(x, 1, 0)), -1, 0)
            case Lam(body): out = Lam(traceredux(body, L))
            case App(f, xs) if callable(f): out = f(*[tracereduce(x, L) for x in xs])
            case App(f, xs):
                inputs[f].append(xs)
                out = App(traceredux(f, L), *[traceredux(x, L) for x in xs])
            case Ref(ix) if ix >= len(L.primitives): out = L[ix]
            case _: out = λ
        returns[λ] = out
        return out

    for tree in newtrees:
        fixpoint(traceredux, encode(tree, L), L)
    for name, ast in libraries:
        fixpoint(traceredux, encode(ast, L), L)

    types = [term.type for term in L.terms]
    types = fixtypes(L, types, libraries, returns)
    for ix, (name, _) in enumerate(libraries):
        L.invented[ix] = L.invented[ix]._replace(type=types[L.index(name)])

    for ix, (name, _) in enumerate(libraries):
        all_tails = inputs[encode(name, L)]
        tailarity = len(all_tails[0])
        tailtypes = [None] * tailarity
        for tails in inputs[encode(name, L)]:
            for tix, tail in enumerate(tails):
                if tailtype := whichtype(tail, types):
                    tailtypes[tix] = tailtype
        L.invented[ix] = L.invented[ix]._replace(tailtypes=tailtypes)

    L.prepare()
    return [encode(tree, L) for tree in newtrees]


if __name__ == '__main__':
    Nat = Type('Nat')
    Float = Type('Float')
    Action = Type('Action')
    Render = Type('Render')

    maxnum = 16
    L = Library([
        Term(tuple(), Action, repr='ø'),
        *[Term(n, Nat, repr=repr(n)) for n in range(maxnum+1)],
        *[Term(i, Float, repr=repr) for i, repr in enumerate(['-pi/2', 'pi/2'])],
        Term(move,
             type=Action,
             tailtypes=[Nat, Action],
             forbidden=[['0'], []],
             repr='move'),
        Term(rotate,
             type=Action,
             tailtypes=[Float, Action],
             forbidden=[[], ['rotate', 'ø']],
             repr='rotate'),
        Term(penup,
             type=Action,
             tailtypes=[Action, Action],
             forbidden=[['penup', 'ø'], ['penup', 'ø']],
             repr='penup'),
        Term(stage,
             type=Render,
             tailtypes=[Action],
             repr='render'),
    ], type=Render)

    sources = renderalphabet('geo/Geo-Regular.ttf', (13, 13))
    size = 10**7
    ncores = os.cpu_count() // 2

    os.system('rm -rf stash && mkdir stash')
    S = [Solution(x, canvas=np.zeros_like(x), state=defaultstate) for x in sources]
    for epoch in range(5):
        ranges = [(ix*size, ((ix+1)*size)) for ix in range(ncores)]
        solutions, *manysolutions = multicore(explore, zip(repeat(L), repeat(S), ranges))

        for bsolutions in manysolutions:
            for ix, (s1, s2) in enumerate(zip(solutions, bsolutions)):
                if s2.error < s1.error or (s2.error == s1.error and length(s2.tree) < length(s1.tree)):
                    solutions[ix] = s2

        trees = [solution.tree for solution in solutions]
        trees = [tree.xs[-1] if tree.fn.ix == L.index('render') else tree for tree in trees]
        normalized = [fixpoint(normalize, tree, L) for tree in trees]
        trees = compress(L, normalized)

        for s, tree in zip(solutions, trees):
            s.tree = tree

        plotsolutions(L, solutions, epoch=epoch)
        S = solutions + [Solution(x, canvas=np.zeros_like(x), state=defaultstate) for x in sources]
