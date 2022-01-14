from importblob import *
from brush import *

class Gamma(NamedTuple):
    head: int
    type: type
    tailtypes: list = []
    forbidden: list = []
    repr: str = '?'

    def __repr__(self):
        return self.repr

class Gammas:
    def __init__(self, core, type):
        self.core = core
        self.invented = []
        self.type = type

        self.infer()

    def infer(self):
        atoms = [g for g in self.core + self.invented if not g.tailtypes]
        ops = [g for g in self.core + self.invented if g.tailtypes]
        self.gammas = atoms + ops

        self.bytype = defaultdict(list)

        for gind, g in enumerate(self):
            self.bytype[g.type].append(gind)

        for gind, g in enumerate(self.gammas):
            if not g.forbidden:
                continue

            newtailtypes = []
            for tailind, (tailtype, forbidden) in enumerate(zip(g.tailtypes, g.forbidden)):
                if len(forbidden) == 0:
                    newtailtypes.append(tailtype)
                else:
                    newbytype = list(filter(lambda ind: not self[ind].repr in forbidden, self.bytype[tailtype]))
                    newtype = f"{tailtype[:-1]}:{g.repr}:{tailind}>"

                    self.bytype[newtype] = newbytype
                    newtailtypes.append(newtype)

            self.gammas[gind] = Gamma(g.head, g.type, newtailtypes, forbidden, g.repr)

        self.rbytype = {}
        for tau, ginds in self.bytype.items():
            self.rbytype[tau] = [self.gammas[gind] for gind in ginds]

        self.views = {}
        for tau in self.rbytype.keys():
            self.views[tau] = self.view(tau)

    def reset(self):
        self.invented = []
        self.infer()

    def add(self, g):
        self.invented.append(g)
        self.infer()

    def __getitem__(self, ind):
        if isinstance(ind, int):
            return self.gammas[ind]

        return self.gammas[self.index(ind)]

    def __len__(self):
        return len(self.gammas)

    def __repr__(self):
        return repr(self.gammas)

    def index(self, repr: str):
        for ind, g in enumerate(self.gammas):
            if g.repr == repr:
                return ind

        return None

    def view(self, tau: type):
        "returns view on Gt, mapping of ops, splits for each op, natoms, nops"
        inds = self.bytype[tau]

        # recover ind from view to real ind in G
        mapping = {cind: ind for cind, ind in zip(count(), inds)}

        Gt = [self.gammas[ind] for ind in inds]
        natoms, nops = getns(Gt)
        # the number of splits for each non-atom
        nsplitmap = array([len(self[i].tailtypes) for i in inds[natoms:]])
        return Gt, mapping, nsplitmap, natoms, nops

    def __hash__(self):
        return len(self.invented)

def getns(G: List[Gamma]) -> (int, int):
    "gives nops, natoms for G"
    nops, natoms = 0, 0

    for g in G:
        if not g.tailtypes:
            natoms += 1
        else:
            nops += 1

    return natoms, nops

class Gm(NamedTuple):
    head: int
    tails: tuple = tuple()

    def __repr__(self):
        # abstraction
        if isinstance(self.head, Gm):
            if self.tails:
                return f"(λ[{self.head}]. {' '.join(map(str, self.tails))})"

            return f"λ[{self.head}]"

        if isinstance(self.head, int) and self.head < 0:
            return str(self.head)

        if not self.tails:
            return G[self.head].repr

        return f"({G[self.head].repr} {' '.join(map(str, self.tails))})"

def length(g: Gm) -> int:
    return 1 + sum(map(length, g.tails))

@lru_cache(maxsize=1 << 15)
def evalg(G, args, g):
    # debruijn index
    if isinstance(g.head, int) and g.head < 0:
        return args[g.head]

    # atom
    if not g.tails:
        return G[g.head].head

    # application
    gamma = G[g.head]
    tails = tuple([evalg(G, args, tail) for tail in g.tails])

    # abstraction
    if isinstance(gamma.head, Gm):
        return evalg(G, tails, gamma.head)

    return gamma.head(*tails)

G = Gammas([
    Gamma(1, '<N>', repr='1'),
    Gamma(2, '<N>', repr='2'),
    Gamma(3, '<N>', repr='3'),
    Gamma(4, '<N>', repr='4'),
    Gamma(5, '<N>', repr='5'),
    Gamma(20, '<N>', repr='inf'),
    Gamma(add, '<N>', ['<N>', '<N>'], [['inf'], ['inf', '1', '2', '3', '4']], repr='+'),
    Gamma(π/100, '<A>', repr='ε'),
    Gamma(divpi, '<A>', ['<N>', '<N>'], [['inf'], ['1', 'inf']], repr='π'),
    Gamma(neg, '<A>', ['<A>'], [['-']], repr='-'),
    Gamma(omv, '<B>', ['<N>', '<A>', '<B>'], [['inf'], [], []], repr='mv'),
    Gamma(opu, '<B>', ['<B>', '<B>'], forbidden=[['penup', 'ø', 'loop', 'savex'], ['penup', 'ø']], repr='penup'),
    Gamma(olp, '<B>', ['<N>', '<B>', '<B>'], [['1'], ['ø', 'loop'], ['loop']], repr='loop'),
    Gamma([(0.0, 0.0)], '<B>', repr='ø'),
    Gamma(stage, '<SS>', ['<B>'], repr='R')
], type='<SS>')


@njit
def fancysplit(base: int, nsplitmap: np.ndarray, n: int):
    n, op = divmod(n, base)

    ind = 0
    nsplits = nsplitmap[op]
    numbers = [0] * nsplits
    while n > 0:
        n, bit = divmod(n, base)

        nbit, strand = divmod(ind, nsplits)

        # if nbit > strandlimit[strand]:
        #     strand = (strand + 1) % nsplits

        numbers[strand] += bit * base ** nbit

        ind += 1

    return [op] + numbers

@njit
def singlesplit(nsplits: int, n: int):
    ind = 0
    base = 2
    numbers = [0] * nsplits
    while n > 0:
        n, bit = divmod(n, base)

        nbit, strand = divmod(ind, nsplits)
        numbers[strand] += bit * base ** nbit

        ind += 1

    return numbers

@lru_cache(maxsize=1 << 15)
def maketree(G: Gammas, tau: type, n: int) -> Gm:
    Gt, mapping, nsplitmap, natoms, nops = G.views[tau]

    if n < natoms:
        return Gm(mapping[n])

    # underflowing base 2
    if nops < 2:
        head = mapping[natoms]
        tails = singlesplit(nsplitmap[0], n-natoms)
    else:
        head, *tails = fancysplit(nops, nsplitmap, n-natoms)
        head = mapping[head + natoms]

    tailtypes = G[head].tailtypes
    return Gm(head, tuple([maketree(G, tau, n) for tau, n in zip(tailtypes, tails)]))

from gast import parse


def interpret(G, ast):
    # redundant parens
    if len(ast) == 1 and isinstance(ast[0], list):
        return interpret(G, ast[0])

    if not isinstance(ast, list):
        # atom
        if (ind := G.index(ast)) is not None:
            return Gm(ind)

        # hole
        if '<' in ast:
            return ast

        # debruijn
        try:
            if (x := int(ast)) < 0:
                return Gm(x)
            else:
                raise
        except:
            raise ValueError(f'What to do? Op/Atom "{ast}" is not one of {G.gammas}')

    if isinstance(ast, list):
        if (ind := G.index(ast[0])) is not None:
            return Gm(ind, tuple([interpret(G, tail) for tail in ast[1:]]))
        else:
            raise ValueError(f'What to do? Op/Atom "{ast[0]}" is not one of {G.gammas}')

GN = Gammas([
    Gamma(1, '<N>', repr='1'),
    Gamma(add, '<N>', ['<N>', '<N>'], forbidden=[[], []], repr='+'),
], type='<N>')

P = lambda G, s: interpret(G, parse(s))

assert P(GN, "(+ (+ 1 1) (+ 1 1))") == Gm(1, (Gm(1, (Gm(0), Gm(0))), Gm(1, (Gm(0), Gm(0)))))

def everysubtree(t):
    qq = [t]

    while qq:
        n = qq.pop(0)
        qq.extend(n.tails)
        yield n

def isequal(t1, t2):
    return t1.head == t2.head and all(isequal(tail1, tail2) for tail1, tail2 in zip(t1.tails, t2.tails))

def flatten(xs):
    for x in xs:
        yield from x

def allcombinations(xs):
    return flatten(combinations(xs, n) for n in range(len(xs) + 1))

def prerelease(G, t):
    L = len(t.tails)
    trees = []

    for holeinds in allcombinations(range(L)):
        tails = copy(G[t.head].tailtypes)

        tailinds = set(range(L)) - set(holeinds)
        allsubtrees = [prerelease(G, t.tails[tind]) for tind in tailinds]

        for subtrees in itertools.product(*allsubtrees):
            newtails = copy(tails)
            for ind, st in zip(tailinds, subtrees):
                newtails[ind] = st

            g = Gm(t.head, tuple(newtails))
            trees.append(g)

    return trees

def release(G, g):
    if not g.tails:
        return g, G[g.head].type

    ghosttails = [release(G, tail) for tail in g.tails]
    ghosts = [G[g.head].type]

    for subtrees in product(*ghosttails):
        ghosts.append(Gm(g.head, tuple(subtrees)))

    return ghosts

def depth(g):
    if not g.tails:
        return 0

    return 1 + max(depth(tail) for tail in g.tails)

def forcematch(G, st, t):
    if st.head != t.head:
        return []

    if not st.tails or not t.tails:
        return [st]

    matches = set()
    L = len(st.tails)

    # free tails by holes
    for holeinds in allcombinations(range(L)):
        tails = deepcopy(G[st.head].tailtypes)

        tailinds = set(range(L)) - set(holeinds)

        # expect keep tails recursively which are partially equal
        allmatches = [forcematch(G, st.tails[tind], t.tails[tind]) for tind in tailinds]

        for match in itertools.product(*allmatches):
            newtails = deepcopy(tails)
            for m, ind in zip(match, tailinds):
                newtails[ind] = m

            g = Gm(st.head, tuple(newtails))
            matches.add(g)

    return matches

assert len(forcematch(GN, P(GN, "(+ 1 1)"), P(GN, "(+ 1 1)"))) == 4
assert len(forcematch(GN, P(GN, "(+ 1 1)"), P(GN, "(+ 1 (+ 1 1))"))) == 2
assert len(forcematch(GN, P(GN, "(+ (+ 1 1) 1)"), P(GN, "(+ 1 (+ 1 1))"))) == 1

def countmatches(G, matches, st, t):
    if not st.tails or not t.tails:
        return

    if st.head == t.head:
        for match in forcematch(G, st, t):
            matches[match] += 1

    for tail in t.tails:
        countmatches(G, matches, st, tail)

matches = defaultdict(int)
countmatches(GN, matches, P(GN, "(+ 1 (+ 1 1))"), P(GN, "(+ 1 (+ (+ 1 1) (+ 1 1)))"))

assert matches[P(GN, '(+ 1 (+ <N> <N>))')] == 1
assert matches[P(GN, '(+ <N> <N>)')] == 4
assert matches[P(GN, '(+ 1 <N>)')] == 3
assert matches[P(GN, '(+ <N> (+ <N> <N>))')] == 2
assert matches[P(GN, '(+ <N> (+ 1 <N>))')] == 1
assert matches[P(GN, '(+ <N> (+ <N> 1))')] == 1
assert matches[P(GN, '(+ <N> (+ 1 1))')] == 1

@lru_cache
def lent(t):
    "natoms+nops, nargs"
    # types, holes
    if not isinstance(t, Gm):
        return [0, 1]

    return list(reduce(lambda acc, x: [acc[0] + x[0], acc[1] + x[1]], map(lent, t.tails), [1, 0]))

def isequalhole(t1, t2):
    if t1.head != t2.head or len(t1.tails) != len(t2.tails):
        return False

    for tail1, tail2 in zip(t1.tails, t2.tails):
        # hole
        if isinstance(tail1, str) or isinstance(tail2, str):
            continue

        if not isequalhole(tail1, tail2):
            return False

    return True

assert isequalhole(Gm(1), Gm(1))
assert not isequalhole(Gm(1, (Gm(1),)), Gm(1))
assert isequalhole(Gm(1, (Gm(1),)), Gm(1, (Gm(1),)))
assert isequalhole(Gm(1, ('1.t0',)), Gm(1, (Gm(1),)))
assert isequalhole(Gm(1, ('1.t0', Gm(2))), Gm(1, (Gm(1), Gm(2))))
assert isequalhole(Gm(1, (Gm(1, (Gm(2), )), Gm(2))), Gm(1, (Gm(1, (Gm(2), )), Gm(2))))
assert not isequalhole(Gm(1, (Gm(1, (Gm(2), '1.t1')), Gm(2))), Gm(1, (Gm(1, (Gm(1), Gm(2))), Gm(2))))

def rewrite(source, match, target):
    if not source.tails:
        return source

    if isequal(source, match):
        return target

    newtails = [None] * len(source.tails)
    for tind, tail in enumerate(source.tails):
        if isequalhole(tail, match):
            newtails[tind] = target
        else:
            newtails[tind] = rewrite(tail, match, target)

    return source._replace(tails=tuple(newtails))

def forceholes(t, tailtypes=None):
    if not t.tails:
        return t

    if tailtypes is None:
        tailtypes = []

    newtails = [None] * len(t.tails)
    for tind, tail in enumerate(t.tails):
        if isinstance(tail, str):
            tailtypes.append(tail)
            newtails[tind] = Gm(-len(tailtypes))
        else:
            newtails[tind] = forceholes(tail, tailtypes)

    return t._replace(tails=tuple(newtails))

# ■ ~

from fontrender import alfbet
X = alfbet.astype(np.int8)


@dataclass
class Beam:
    x: np.ndarray
    ax: np.ndarray = None
    error: float = np.inf
    tree: Gm = None

class H(NamedTuple):
    nc: int
    count: int
    tree: Gm

    def __lt__(self, o):
        return self.nc < o.nc

def brusheval(G, im, tree):
    im.fill(0)
    evalg(G, (), tree)(im)

@njit
def compare(x, ax):
    return np.abs(x - ax).sum()

def explore(G, X, Z, ns=(0, 10**6)):
    im = zeros(X[0].shape, np.int8)

    stime = time()
    tbar = trange(*ns)
    for n in tbar:
        tree = maketree(G, G.type, n=n)
        brusheval(G, im, tree)

        for z in Z:
            error = compare(z.x, im)

            if error < z.error or (error == z.error and length(tree) < length(z.tree)):
                z.error = error
                z.ax = im.copy()
                z.tree = tree

    return Z

def plotz(Z, ind=0):
    N = len(Z)
    fig, axs = pyplot.subplots(N, 1, figsize=(40, 20))

    for z, ax in zip(Z, axs):
        im = np.hstack((z.x, z.ax))
        ax.imshow(im, cmap='hot', interpolation='bicubic')
        ax.grid(None)
        ax.axis('off')
        ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
        ax.set_title(f"[{z.error}] {z.tree}", size=18, y=1.025, fontfamily='IBM Plex Sans')

    pyplot.subplots_adjust(left=0.0, right=1, hspace=0.4)
    savefig(f"stash/z{ind}.png")


def multicore(nocores, fn, args):
    if ncores == 1:
        return fn(*list(args)[0])

    try:
        pool = mp.Pool(ncores)

        if isinstance(args, list):
            out = pool.map(fn, args)
        else:
            out = pool.starmap(fn, args)
    finally:
        pool.close()
        pool.join()

    return out

def countghosts(trees, alltrees):
    matches = defaultdict(int)

    for ghost in flatten(release(G, tree) for tree in trees):
        if isinstance(ghost, str) or not ghost.tails:
            continue

        for tree in alltrees:
            if isequalhole(ghost, tree):
                matches[ghost] += 1

    return matches

def releasetrees(trees):
    return set(flatten(release(G, tree) for tree in trees))

if __name__ == '__main__':
    shutil.rmtree('stash')
    os.mkdir('stash')

    ncores = 1
    batch = 10**6

    for ind in count():
        size = (ind+1) * batch

        Z = [Beam(x) for x in X]

        nss = []
        for jnd in range(ncores):
            nss.append((jnd * size+1, ((jnd + 1) * size)))

        manyZ = multicore(ncores, explore, zip(repeat(G), repeat(X), repeat(Z), nss))

        if ncores > 1:
            for nZ in manyZ:
                for zind, (nz, z) in enumerate(zip(nZ, Z)):
                    if nz.error < z.error or (nz.error == z.error and length(nz.tree) < length(z.tree)):
                        Z[zind] = nz
        else:
            Z = manyZ

        plotz(Z, ind)

        for z in Z:
            print(f"[{z.error}] ~ {z.tree}")

        trees = list(flatten(map(everysubtree, (z.tree for z in Z))))

        manymatches = multicore(ncores, countghosts, zip(coresplit(ncores, trees), repeat(trees)))
        if ncores > 1:
            matches = reduce(lambda acc, x: acc | x, manymatches)
        else:
            matches = manymatches

        heap = []

        for tree, c in tqdm(matches.items()):
            if not isinstance(tree, Gm):
                continue

            n, nargs = lent(tree)
            if n <= 1:
                continue

            nc = (n - nargs) * c
            h = H(nc=nc, tree=tree, count=c)

            if len(heap) < 10:
                heapq.heappush(heap, h)
            else:
                heapq.heappushpop(heap, h)

        for h in heapq.nlargest(10, heap):
            print(h)

        h = sorted(heap, reverse=True)[0]

        tailtypes = []
        tree = forceholes(h.tree, tailtypes)

        if not tailtypes:
            atom = evalg(G, (), tree)
            g = Gamma(atom, G[tree.head].type, [], repr=f"a{len(G.invented)}")
        else:
            g = Gamma(tree, G[tree.head].type, list(reversed(tailtypes)), repr=f"f{len(G.invented)}")

        print(f'adding {tree=} {tailtypes=} with pleasure of {nc=} & {h.count=}')

        G.add(g)

        print(f'{G=}')

        for g in G.invented:
            print(f'{g} {g.head} : {" -> ".join(g.tailtypes)} -> {g.type}')
