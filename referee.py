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
        # keep the ordering
        self.gammas = self.core + self.invented
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

        self.rbytype = {tau: [self.gammas[gind] for gind in ginds] for tau, ginds in self.bytype.items()}
        self.views = {tau: self.view(tau) for tau in self.bytype.keys()}

    def reset(self):
        self.invented = []
        self.infer()

    def add(self, g):
        self.invented.append(g)
        self.infer()

    def __getitem__(self, ind):
        if isinstance(ind, (int, np.int64)):
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
        Gt = [self.gammas[ind] for ind in inds]

        # recover ind from view to real ind in G
        # sort atoms first than ops
        atominds = [gind for gind, g in zip(inds, Gt) if not g.tailtypes]
        funcinds = [gind for gind, g in zip(inds, Gt) if g.tailtypes]

        natoms = len(atominds)
        nfuncs = len(funcinds)

        mapping = array(atominds + funcinds, dtype=int)

        # the number of splits for each non-atom
        nsplitmap = array([len(self[funcind].tailtypes) for funcind in funcinds])

        strandlimit = []

        for funcind in funcinds:
            # max bound
            limits = zeros(6, dtype=int)

            base = max(2, nfuncs)

            # ...
            for tind, tailtype in enumerate(self[funcind].tailtypes):
                if tailtype.startswith('<N'):
                    natoms_in_n = 9

                    limits[tind] = int(np.ceil(np.log(natoms_in_n) / np.log(base)))
                else:
                    limits[tind] = 10**3

            strandlimit.append(limits)

        return mapping, nsplitmap, array(strandlimit, dtype=int), natoms, nfuncs

    def __hash__(self):
        return len(self.invented)

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

G = Gammas([
    Gamma(0, '<A>', repr='0'),
    Gamma(1, '<N>', repr='1'),
    Gamma(2, '<N>', repr='2'),
    Gamma(3, '<N>', repr='3'),
    Gamma(4, '<N>', repr='4'),
    Gamma(5, '<N>', repr='5'),
    Gamma(6, '<N>', repr='6'),
    Gamma(7, '<N>', repr='7'),
    Gamma(8, '<N>', repr='8'),
    Gamma(9, '<N>', repr='9'),
    Gamma(add, '<N>', ['<N>', '<N>'], [['inf'], ['inf', '1', '2', '3', '4', '5', '6', '7', '8']], repr='+'),
    Gamma(divpi, '<A>', ['<N>', '<N>'], [['inf'], ['1', 'inf']], repr='π'),
    Gamma(neg, '<A>', ['<A>'], [['-', '0']], repr='-'),
    Gamma(omv, '<B>', ['<N>', '<A>', '<B>'], [['inf'], [], []], repr='mv'), # first f must resolve
    Gamma(osx, '<B>', ['<B>', '<B>'], [['ø', 'savex'], ['ø']], repr='savex'),
    Gamma(opu, '<B>', ['<B>', '<B>'], forbidden=[['penup', 'ø', 'loop', 'savex'], ['penup', 'ø']], repr='penup'),
    Gamma(olp, '<B>', ['<N>', '<B>', '<B>'], [['1'], ['ø', 'loop'], ['loop']], repr='loop'),
    Gamma([], '<B>', repr='ø'),
    Gamma(stage, '<SS>', ['<B>'], repr='R')
], type='<SS>')

@lru_cache(maxsize=1 << 15)
def length(g: Gm) -> int:
    if isinstance(g, str):
        return 1

    return 1 + sum(map(length, g.tails))
# ■ ~

# @lru_cache(maxsize=1 << 15)
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

# ■ ~

@njit
def fancysplit(base: int, nsplitmap, strandlimit, n: int):
    n, op = divmod(n, base)

    ind = 0
    nsplits = nsplitmap[op]
    limits = strandlimit[op]

    numbers = [0] * nsplits
    while n > 0:
        n, bit = divmod(n, base)

        nbit, strand = divmod(ind, nsplits)

        if nbit >= limits[strand]:
            strand = (strand + 2) % nsplits

        numbers[strand] += bit * base ** nbit

        ind += 1

    return [op] + numbers

@njit
def singlesplit(nsplits: int, limits, n: int):
    ind = 0
    base = 2
    numbers = [0] * nsplits

    while n > 0:
        n, bit = divmod(n, base)

        nbit, strand = divmod(ind, nsplits)

        if nbit > limits[strand]:
            strand = (strand + 1) % nsplits

        numbers[strand] += bit * base ** nbit

        ind += 1

    return numbers

@lru_cache(maxsize=1 << 15)
def maketree(G: Gammas, tau: type, n: int) -> Gm:
    mapping, nsplitmap, strandlimit, natoms, nops = G.views[tau]

    if n < natoms:
        return Gm(mapping[n])

    # underflowing base 2
    if nops < 2:
        head = mapping[natoms]
        tails = singlesplit(nsplitmap[0], strandlimit[0], n-natoms)
    else:
        head, *tails = fancysplit(nops, nsplitmap, strandlimit, n-natoms)
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

def genrelease(G, g):
    if not g.tails:
        yield g
        yield G[g.head].type
        return

    yield G[g.head].type
    ghosttails = [genrelease(G, tail) for tail in g.tails]

    for subtrees in product(*ghosttails):
        yield Gm(g.head, tuple(subtrees))

def depth(g):
    if not g.tails:
        return 0

    return 1 + max(depth(tail) for tail in g.tails)

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


def extractargs(t, tholed, args):
    if not tholed.tails:
        return

    for tind, (tailholed, tail) in enumerate(zip(tholed.tails, t.tails)):
        if isinstance(tailholed, str):
            args.append(tail)
        else:
            extractargs(tail, tailholed, args)

tr = lambda g: P(G, g)

args = []
extractargs(tr('(mv (+ 1 1) (π 1 2) ø)'), tr('(mv (+ 1 <N>) (π 1 <N>) <B>)'), args)
assert args == [tr('1'), tr('2'), tr('ø')]

def rewrite(source, match, target, withargs=False):
    if not source.tails:
        return source

    if isequalhole(source, match):
        if withargs:
            args = []
            extractargs(source, match, args)

            # again reversed for - access
            return Gm(target.head, tuple(reversed(args)))

        return Gm(target.head)

    newtails = [None] * len(source.tails)
    for tind, tail in enumerate(source.tails):
        if isequalhole(tail, match):
            if withargs:
                args = []
                extractargs(tail, match, args)
                newtails[tind] = Gm(target.head, tuple(reversed(args)))
            else:
                newtails[tind] = target
        else:
            newtails[tind] = rewrite(tail, match, target, withargs=withargs)

    return source._replace(tails=tuple(newtails))

t = P(G, "(+ (+ 1 1) 1)")
nt = rewrite(t, P(G, "1"), P(G, "(+ 1 1)"))
nt = rewrite(nt, P(G, "(+ 1 1)"), P(G, "1"))
assert isequal(t, nt)

nt = rewrite(t, P(G, "(+ 1 <N>)"), Gm(0), withargs=True)

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

from fontrender import alfbet
X = alfbet.astype(np.int8)

iimshow = lambda x: imshow(x, interpolation=None, cmap='hot')

def outcompare(x, ax):
    diff = x - ax

    nnotcovered = len(diff[diff > 0])
    nredundant = len(diff[diff < 0])
    npoints = len(x[x > 0])

    return nnotcovered, nredundant, npoints, diff

def plotz(Z, ind=0):
    canvas = zeros(Z[0].x.shape, np.int8)

    N = len(Z)
    fig, axs = pyplot.subplots(N, 1, figsize=(32, N * 4))

    totalerror = sum(z.error for z in Z)

    for z, ax in zip(Z, axs):
        brusheval(G, canvas, z.g)

        canvas = np.roll(canvas, (3, 4), axis=(0, 1))
        original = np.roll(z.x, (3, 4), axis=(0, 1))

        nnotcovered, nredundant, npoints, diff = outcompare(original, canvas)

        im = np.hstack((canvas, original, diff))

        ax.imshow(im, interpolation=None, cmap='hot', vmin=-1, vmax=1)
        ax.grid(None)
        ax.axis('off')
        ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)

        title = f"covered {npoints - nnotcovered}/{npoints} + extra {nredundant} totaling @{z.error}"
        ax.set_title(f"{title}\n{z.g}", size=24, y=1.05)
        print(title)

    suptitle = f'({ind}) Total error = {totalerror}'
    fig.suptitle(suptitle, y=0.9, size=24)
    print(suptitle)

    pyplot.subplots_adjust(left=0.0, right=1, hspace=0.5)
    savefig(f"stash/z{ind}.png")

@dataclass
class Beam:
    x: np.ndarray
    error: float = np.inf
    g: Gm = None

class H(NamedTuple):
    kalon: int
    count: int
    g: Gm

    def __lt__(self, o):
        return self.kalon < o.kalon

def brusheval(G, canvas, g):
    canvas.fill(0)
    evalg(G, (), g)(canvas)
    return canvas.copy()

@njit
def compare(x, ax):
    return np.abs(x - ax).sum()

def explore(G, X, Z, ns=(0, 10**6)):
    ng = (ns[1] - ns[0]) // 10
    im = zeros(X[0].shape, np.int8)

    c = 0
    stime = time()
    tbar = range(*ns)
    for n in tbar:
        g = maketree(G, G.type, n=n)
        im.fill(0)
        evalg(G, (), g)(im)

        for z in Z:
            error = compare(z.x, im)

            if error < z.error or (error == z.error and length(g) < length(z.g)):
                z.error = error
                z.g = g

        c += 1

        if n % ng == 0:
            print(f'{g} {c / (time() - stime):.0f}/s')

    return Z

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

def isequalholesub(ghost, tree):
    if ghost.head != tree.head:
        return False

    for g, t in zip(ghost.tails, tree.tails):
        if isinstance(g, str):
            continue

        if not isequalholesub(g, t):
            return False

    return True

def biggerlength(g, n):
    qq = [g]

    while qq:
        x = qq.pop(0)
        n -= 1

        if n < 0:
            return True

        if isinstance(x, str):
            continue

        for t in x.tails:
            qq.append(t)

    return False

def countghosts(trees, alltrees):
    matches = defaultdict(int)

    for ghost in releasetrees(trees):
        if isinstance(ghost, str) or not ghost.tails:
            continue

        for tree in alltrees:
            if isequalholesub(ghost, tree):
                matches[ghost] += 1

    return matches

def releasetrees(trees):
    return flatten(genrelease(G, tree) for tree in trees)

# ■ ~

if __name__ == '__main__':
    faststart = False
    ncores = os.cpu_count() // 2
    batch = 1 * 10**7

    if ncores == 4:
        ncores = 1
        batch = 1 * 10**6
    else:
        faststart = False

    if not faststart:
        shutil.rmtree('stash')
        os.mkdir('stash')

    for ind in count():
        if ind > 0 or not faststart:
            size = (ind+1) * batch

            Z = [Beam(x) for x in X]

            nss = []
            for jnd in range(ncores):
                nss.append((jnd * size+1, ((jnd + 1) * size)))

            manyZ = multicore(ncores, explore, zip(repeat(G), repeat(X), repeat(Z), nss))

            if ncores > 1:
                for nZ in manyZ:
                    for zind, (nz, z) in enumerate(zip(nZ, Z)):
                        if nz.error < z.error or (nz.error == z.error and length(nz.g) < length(z.g)):
                            Z[zind] = nz
            else:
                Z = manyZ

            pickle.dump(Z, open(f'stash/Z{ind}.pkl', 'wb'))
        else:
            Z = pickle.load(open(f'stash/Z{ind}.pkl', 'rb'))

        plotz(Z, ind)

        # for R-case solely
        trees = [z.g.tails[0] for z in Z]

        while True:
            Mx = sum(map(length, trees))

            print(f'counting currently at {Mx=}...')

            stime = time()

            subtrees = []
            for st in flatten(map(everysubtree, trees)):
                if biggerlength(st, 1) and not biggerlength(st, 36):
                    subtrees.append(st)

            print(f'total subtrees: {len(subtrees)}')
            manymatches = multicore(ncores, countghosts, zip(coresplit(ncores, subtrees), repeat(trees)))
            if ncores > 1:
                matches = reduce(lambda acc, x: acc | x, manymatches)
            else:
                matches = manymatches

            print(f'took {time() - stime:.0f}s')

            heap = []

            for candidate, c in tqdm(matches.items()):
                if not isinstance(candidate, Gm):
                    continue

                n, nargs = lent(candidate)
                if n <= 1:
                    continue

                Mxg = Mx - c * (n - nargs - 1)
                Mg = n

                kalon = (Mxg + Mg) / Mx

                h = H(kalon=-kalon, g=candidate, count=c)

                if len(heap) < 5:
                    heapq.heappush(heap, h)
                else:
                    heapq.heappushpop(heap, h)

            for h in heapq.nlargest(3, heap):
                print(h)

            h = sorted(heap, reverse=True)[0]

            if h.kalon <= -1:
                break

            tailtypes = []
            gbodywithholes = h.g
            gbody = forceholes(gbodywithholes, tailtypes)

            isatom = not tailtypes

            if isatom:
                atom = evalg(G, (), gbody)
                gamma = Gamma(atom, G[gbody.head].type, [], repr=f"a{len(G.invented)}")
            else:
                gamma = Gamma(gbody, G[gbody.head].type, list(reversed(tailtypes)), repr=f"f{len(G.invented)}")

            print(f'adding {gbody=} {tailtypes=} with pleasure of {h.kalon=} & {h.count=}')
            print(f'{gbodywithholes=} {gbody=}')

            G.add(gamma)
            G.infer()
            gind = len(G)-1

            canvas = zeros(Z[0].x.shape, int)
            outbefore = [brusheval(G, canvas, Gm(G.index('R'), (g,))) for g in trees]

            print('rewriting...')
            trees = [rewrite(g, gbodywithholes, Gm(gind), withargs=not isatom) for g in trees]
            outafter = [brusheval(G, canvas, Gm(G.index('R'), (g,))) for g in trees]

            for ind, (outa, outb) in enumerate(zip(outbefore, outafter)):
                if (outa - outb).sum() != 0:
                    raise ValueError(f'bad rewrite @ {trees[ind]}')

        for g in G.invented:
            print(f'{g} {g.head} : {" -> ".join(g.tailtypes)} -> {g.type}')
