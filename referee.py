from importblob import *
from brush import *
from q import *
from yee import ImageEncoder
from rich import print
from gast import parse

# ■ ~

class Gm(NamedTuple):
    head: int
    tails: tuple = tuple()

    def __repr__(self):
        global G
        return grepr(G, self)

        if self.tails:
            return f"({self.head} {' '.join([repr(t) for t in self.tails])})"

        return str(self.head)

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
        # tau -> [gind]
        self.bytype = defaultdict(list)

        # fill type pool
        for gind, g in enumerate(self):
            self.bytype[g.type].append(gind)

        # reconstruct gammas
        for gind, g in enumerate(self):
            newtailtypes = []

            for tind, (tailtype, forbidden) in enumerate(zip_longest(g.tailtypes, g.forbidden)):
                forbidden = forbidden or []
                generictype = f"{tailtype.split(':')[0]}>" if ':' in tailtype else tailtype
                # sieve forbidden ginds
                ginds = [ind for ind in self.bytype[generictype] if not self[ind].repr in forbidden]
                specifictype = f"{generictype[:-1]}:{g.repr}:{tind}>"

                self.bytype[specifictype] = ginds
                newtailtypes.append(specifictype)

            self.gammas[gind] = Gamma(g.head, g.type, newtailtypes, g.forbidden, g.repr)

        self.rbytype = {tau: [self.gammas[gind] for gind in ginds] for tau, ginds in self.bytype.items()}

    def solder(self, Q=None):
        if Q is None:
            Q = {tau: -np.log2(th.ones(len(gs))/len(gs)) for tau, gs in self.bytype.items()}

        for tau in self.bytype.keys():
            qs, sinds = Q[tau].sort()
            qs = ap(lambda x: round(x, 1), qs.numpy())
            gs = ap(lambda gind: G[gind].repr, np.atleast_1d(array(self.bytype[tau])[sinds]))

            print(f'{tau} ~ {list(zip(gs, qs))}')

        self.views = {tau: makeview(self, Q, tau) for tau in self.bytype.keys()}

    def reset(self):
        self.invented = []
        self.infer()

    def add(self, g):
        self.invented.append(g)
        self.gammas = self.core + self.invented

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

    def __hash__(self):
        return len(self.invented)

def grepr(G, g):
    if not isinstance(g, (Gm, int)):
        return g

    if isinstance(g.head, Gm):
        if g.tails:
            return f"(λ[{g.head}]. {' '.join([grepr(G, t) for t in g.tails])})"

        return f"λ[{g.head}]"

    if isinstance(g.head, int) and g.head < 0:
        return str(g.head)

    if not g.tails:
        return G[g.head].repr

    return f"({G[g.head].repr} {' '.join([grepr(G, t) for t in g.tails])})"

@lru_cache(maxsize=1 << 15)
def length(g: Gm) -> int:
    if isinstance(g, str):
        return 1

    return 1 + sum(map(length, g.tails))

# TODO unhashable type list: tail <B> accepts [(...)]
# @lru_cache(maxsize=1 << 20)
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

        if nbit >= limits[strand]:
            strand = (strand + 1) % nsplits

        numbers[strand] += bit * base ** nbit

        ind += 1

    return numbers

@lru_cache(maxsize=1 << 20)
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

@lru_cache(maxsize=1 << 20)
def growtree(G, tau, n):
    masks, fnumber, nops, natoms, opmapping, atommapping = G.views[tau]

    if n < natoms:
        return Gm(atommapping[n])

    assert nops > 0, f"there are not enough atoms to feed this {n}"

    n -= natoms
    opind = selectmask(nops, n, fnumber)

    op = opmapping[opind]
    ttaus = G[op].tailtypes

    return Gm(op, tuple([growtree(G, ttau, selectmask(nops, n, mask))
                         for mask, ttau in zip(masks[opind], ttaus)]))

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

P = lambda G, s: interpret(G, parse(s))

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

def extractargs(t, tholed, args):
    if not tholed.tails:
        return

    for tind, (tailholed, tail) in enumerate(zip(tholed.tails, t.tails)):
        if isinstance(tailholed, str):
            args.append(tail)
        else:
            extractargs(tail, tailholed, args)

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
# from imagrender import imagrender
# X = imagrender('imagery')

iimshow = lambda x: imshow(x, interpolation=None, cmap='hot')

def outcompare(x, ax):
    diff = x - ax

    nnotcovered = len(diff[diff > 0])
    nredundant = len(diff[diff < 0])
    npoints = len(x[x > 0])

    return nnotcovered, nredundant, npoints, diff

def plotz(G, Z, ind=0):
    canvas = zeros(Z[0].x.shape, np.int8)

    N = len(Z)
    fig, axs = pyplot.subplots(N, 1, figsize=(32, N * 4))

    totalerror = sum(z.error for z in Z)

    for z, ax in zip(Z, axs):
        brusheval(G, canvas, z.g)

        canvas = np.roll(canvas, (2, 2), axis=(0, 1))
        original = np.roll(z.x, (2, 2), axis=(0, 1))

        nnotcovered, nredundant, npoints, diff = outcompare(original, canvas)

        im = np.hstack((canvas, original, diff))

        ax.imshow(im, interpolation=None, cmap='hot', vmin=-1, vmax=1)
        ax.grid(None)
        ax.axis('off')
        ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)

        title = f"covered {npoints - nnotcovered}/{npoints} + extra {nredundant} totaling @{z.error}"
        ax.set_title(f"{title}\n{grepr(G, z.g)}", size=24, y=1.15)
        print(title)

    suptitle = f'Epoch #{ind} Total error = {totalerror}'
    fig.suptitle(suptitle, size=24)
    print(suptitle)

    pyplot.subplots_adjust(left=0.0, right=1, hspace=0.9)
    savefig(f"stash/{ind}.png")

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
        try:
            g = growtree(G, G.type, n=n)
        except RecursionError:
            print(f'too much recur {n=}')
            continue

        im.fill(0)
        evalg(G, (), g)(im)

        for z in Z:
            error = compare(z.x, im)

            if error < z.error or (error == z.error and length(g) < length(z.g)):
                z.error = error
                z.g = g

        c += 1

        if n % ng == 0:
            print(f'{grepr(G, g)} {c / (time() - stime):.0f}/s')

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

def pevalg(G, args, g):
    if isinstance(g.head, int) and g.head < 0:
        return args[g.head]

    if not g.tails:
        return g

    return g._replace(tails=tuple([pevalg(G, args, tg) for tg in g.tails]))

def isnormal(G, g):
    if g.head >= len(G.core):
        return False

    return all(isnormal(G, tg) for tg in g.tails)

def normalize(G, g):
    if g.head >= len(G.core):
        # get hiddentail
        body = deepcopy(G[g.head].head)
        # insert tails
        body = pevalg(G, g.tails, body)

        while not isnormal(G, body):
            body = normalize(G, body)

        return body

    return g._replace(tails=tuple([normalize(G, tg) for tg in g.tails]))

def countghosts(trees, alltrees, G):
    matches = defaultdict(int)

    for ghost in releasetrees(G, trees):
        if isinstance(ghost, str) or not ghost.tails:
            continue

        for tree in alltrees:
            if isequalholesub(ghost, tree):
                matches[ghost] += 1

    return matches

def releasetrees(G, trees):
    return flatten(genrelease(G, tree) for tree in trees)

def everytail(G, spectaus, g):
    "index of ttau in spectaus -> tail.head"
    qq = [g]
    while qq:
        g = qq.pop(0)
        qq.extend(g.tails)
        for ttau, tail in zip(G[g.head].tailtypes, g.tails):
            yield spectaus.index(ttau), tail.head

def dream(G, trees, shape):
    spectaus = [tau for tau in G.bytype.keys()]
    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    enc = ImageEncoder(nin=shape[0], nout=len(spectaus) * len(G)).to(device)
    opt = th.optim.Adam(enc.parameters(), 1e-4)

    bsize = 64

    tbar = trange(32)
    for _ in tbar:
        gs = [growtree(G, G.type, randint(10**4)) for _ in range(bsize - len(trees))] + trees

        canvas = np.zeros((bsize, *shape), dtype=np.float32)
        for ind in range(bsize):
            try:
                evalg(G, (), gs[ind])(canvas[ind])
            except TypeError:
                print(gs[ind])
                raise

        images = as_tensor(canvas, device=device).view(bsize, 1, *shape)
        logits = enc(images).view(bsize, len(G), len(spectaus))
        logp = F.log_softmax(logits, 1)

        slogp = 0
        for imind in range(bsize):
            for ttau, gind in everytail(G, spectaus, gs[imind]):
                slogp += -logp[imind, gind, ttau]

        opt.zero_grad(); slogp.backward(); opt.step()
        tbar.set_description(f'{slogp/bsize=:.2f}')

    xs = X[:-3]
    xs = as_tensor(xs, dtype=th.float32, device=device).view(-1, 1, *shape)

    with th.no_grad():
        logits = enc(xs).view(-1, len(G), len(spectaus))
        logp = F.log_softmax(logits, 1)
        # |X| x |G| x |T|
        logp = logp.mean(0).cpu()

    Q = {}
    for tau, ginds in G.bytype.items():
        ls = th.hstack([logp[gind, spectaus.index(tau)] for gind in ginds])
        Q[tau] = -F.log_softmax(ls, -1)

    return Q

if __name__ == '__main__':
    L = 15
    G = Gammas([
        Gamma(getangle, '<A>', ['<N>'], repr='π'),
        *[Gamma(n, '<N>', repr=repr(n)) for n in range(L+1)],
        Gamma(add, '<N>', ['<N>', '<N>'], [['0'], [repr(n) for n in range(L)]], repr='+'),
        Gamma(omv, '<B>', ['<N>', '<A>', '<B>'], [[], [], []], repr='mv'), # first f must resolve
        Gamma(osx, '<B>', ['<B>', '<B>'], [['ø', 'savex'], ['ø']], repr='savex'),
        Gamma(opu, '<B>', ['<B>', '<B>'], forbidden=[['penup', 'ø', 'loop', 'savex'], ['penup', 'ø']], repr='penup'),
        Gamma(olp, '<B>', ['<N>', '<B>', '<B>'], [['0', '1'], ['ø', 'loop'], ['loop']], repr='loop'),
        Gamma([], '<B>', repr='ø'),
        Gamma(stage, '<SS>', ['<B>'], [['ø']], repr='R')
    ], type='<SS>')

# ■ ~

if __name__ == '__main__':
    G.solder()
    ncores = os.cpu_count() // 2

    if ncores == 4:
        batch = 1 * 10**5
        ncores = 4
        faststart = False
    else:
        batch = 1 * 10**7
        faststart = False

    if not faststart:
        shutil.rmtree('stash')
        os.mkdir('stash')

    for ind in count():
        if faststart and ind == 0:
            Z = pickle.load(open(f'stash/Z{ind}.pkl', 'rb'))
        else:
            size = (3*ind+1) * batch

            Z = [Beam(x) for x in X]

            nss = []
            for jnd in range(ncores):
                nss.append((jnd * size+1, ((jnd + 1) * size)))

            print(f'exploring a forest spanning 10^{int(np.log10(size))} in search for {len(X)} lone pines')
            manyZ = multicore(ncores, explore, zip(repeat(G), repeat(X), repeat(Z), nss))

            if ncores > 1:
                for nZ in manyZ:
                    for zind, (nz, z) in enumerate(zip(nZ, Z)):
                        if nz.error < z.error or (nz.error == z.error and length(nz.g) < length(z.g)):
                            Z[zind] = nz
            else:
                Z = manyZ

            pickle.dump(Z, open(f'stash/Z{ind}.pkl', 'wb'))
            pickle.dump(G, open(f'stash/G{ind}.pkl', 'wb'))

        plotz(G, Z, ind)

        trees = [z.g for z in Z]
        # normalize
        canvas = zeros(X[0].shape)
        outbefore = [brusheval(G, canvas, g) for g in trees]

        ntrees = [normalize(G, g) for g in trees]
        outafter = [brusheval(G, canvas, g) for g in ntrees]

        for ind, (outa, outb) in enumerate(zip(outbefore, outafter)):
            if not np.all(outa == outb):
                raise ValueError(f'bad rewrite @ {trees[ind]} <> {ntrees[ind]}')

        # strip render stage
        trees = [g.tails[0] for g in ntrees]

        for g in trees:
            print(grepr(G, g))

        G.reset()

        while True:
            Mx = sum(map(length, trees))

            print(f'counting currently at {Mx=}...')

            stime = time()

            subtrees = []
            for st in flatten(map(everysubtree, trees)):
                if biggerlength(st, 1) and not biggerlength(st, 36):
                    subtrees.append(st)

            print(f'total subtrees: {len(subtrees)}')
            manymatches = multicore(ncores, countghosts, zip(coresplit(ncores, subtrees), repeat(trees), repeat(G)))
            if ncores > 1:
                matches = reduce(lambda acc, x: acc | x, manymatches)
            else:
                matches = manymatches

            heap = []

            for candidate, c in tqdm(matches.items()):
                if not isinstance(candidate, Gm):
                    continue

                n, nargs = lent(candidate)
                # temporarily disable constants
                if n <= 1 or nargs == 0:
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
                name = f"f{len(G.invented)}"

                gamma = Gamma(gbody, G[gbody.head].type, list(reversed(tailtypes)), repr=f"f{len(G.invented)}")

            print(f'growing new leaf {grepr(G, gbodywithholes)} with tails {tailtypes} @{-h.kalon:.2f} #{h.count}')

            G.add(gamma)
            gind = len(G)-1

            canvas = zeros(Z[0].x.shape, int)
            outbefore = [brusheval(G, canvas, Gm(G.index('R'), (g,))) for g in trees]

            trees = [rewrite(g, gbodywithholes, Gm(gind), withargs=not isatom) for g in trees]
            outafter = [brusheval(G, canvas, Gm(G.index('R'), (g,))) for g in trees]

            for ind, (outa, outb) in enumerate(zip(outbefore, outafter)):
                if (outa - outb).sum() != 0:
                    raise ValueError(f'bad rewrite @ {trees[ind]}')

        for g in G.invented:
            print(f'{g} {grepr(G, g.head)} : {" -> ".join(g.tailtypes)} -> {g.type}')

        G.infer()
        trees = [Gm(G.index('R'), (g,)) for g in trees]

        # i need to sample, but do i need to do it with a uniform logp?
        G.solder()
        Q = dream(G, trees, X[0].shape)
        G.solder(Q)

        progeny = []
        for n in range(1000):
            g = growtree(G, G.type, n)

            for ind, _g in enumerate(progeny):
                if isequal(_g, g):
                    print(f'found copies {ind} & {n} {grepr(G, _g)} == {grepr(G, g)}')

            if n < 100:
                print(grepr(G, g))

            evalg(G, (), g)
