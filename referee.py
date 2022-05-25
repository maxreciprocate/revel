from blob import *
from lang import *

from rollbrush import *
from q import *
from yee import *
from rich import print
from gast import parse


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
def maketree(G: Language, tau: type, n: int) -> T:
    mapping, nsplitmap, strandlimit, natoms, nops = G.views[tau]

    if n < natoms:
        return T(mapping[n])

    # underflowing base 2
    if nops < 2:
        head = mapping[natoms]
        tails = singlesplit(nsplitmap[0], strandlimit[0], n-natoms)
    else:
        head, *tails = fancysplit(nops, nsplitmap, strandlimit, n-natoms)
        head = mapping[head + natoms]

    tailtypes = G[head].tailtypes

    return T(head, tuple([maketree(G, tau, n) for tau, n in zip(tailtypes, tails)]))

@lru_cache(maxsize=1 << 20)
def growtree(G, tau, n):
    masks, fnumber, leapfrom, leapnumber, nops, natoms, opmapping, atommapping = G.views[tau]

    if n < natoms:
        return T(atommapping[n])

    assert nops > 0, f"there are not enough atoms to feed this {n}"

    n -= natoms

    if n >= leapfrom:
        times = n // leapfrom
        n += times * leapnumber

    opind = selectmask(2, n, fnumber)

    op = opmapping[opind]
    ttaus = G[op].tailtypes

    return T(op, tuple([growtree(G, ttau, selectmask(2, n, mask))
                         for mask, ttau in zip(masks[opind], ttaus)]))

def interpret(L, ast):
    # redundant parens
    if len(ast) == 1 and isinstance(ast[0], list):
        return interpret(L, ast[0])

    elif not isinstance(ast, list):
        # atom
        if (ind := L.index(ast)) is not None:
            return T(ind)

        # hole
        if '<' in ast or '?' in ast:
            return ast

        # debruijn
        try:
            if (x := int(ast)) < 0:
                return T(x)
            else:
                raise
        except:
            raise ValueError(f"Sorry, who's {ast}?")

    if isinstance(ast, list):
        term = ast[0]
        if (ind := L.index(term)) is not None:
            return T(ind, tuple([interpret(L, tail) for tail in ast[1:]]))
        elif term.startswith('?'):
            return term
        else:
            raise ValueError(f"Sorry, who's {ast[0]}?")

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
        yield T(g.head, tuple(subtrees))

def depth(g):
    if not g.tails:
        return 0

    return 1 + max(depth(tail) for tail in g.tails)

@lru_cache
def lent(t):
    "natoms+nops, nargs"
    # types, holes
    if not isinstance(t, T):
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
    "in exact places where is hole in a ghost returns what is living in the tree"
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

            args = list(map(lambda g: rewrite(g, match, target, True), args))
            # again reversed for - access
            return T(target.head, tuple(reversed(args)))

        return T(target.head)

    newtails = [None] * len(source.tails)
    for tind, tail in enumerate(source.tails):
        if isequalhole(tail, match):
            if withargs:
                args = []
                extractargs(tail, match, args)
                # rewrite arguments themselves
                # good that it's only one rewrite each for now
                args = list(map(lambda g: rewrite(g, match, target, True), args))

                newtails[tind] = T(target.head, tuple(reversed(args)))
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
            newtails[tind] = T(-len(tailtypes))
        else:
            newtails[tind] = forceholes(tail, tailtypes)

    return t._replace(tails=tuple(newtails))

def outcompare(x, ax):
    diff = x - ax

    nnotcovered = len(diff[diff > 0])
    nredundant = len(diff[diff < 0])
    npoints = len(x[x > 0])

    return nnotcovered, nredundant, npoints, diff

def plotbeams(G, S, ind=0):
    canvas = zeros(S[0].x.shape, np.int8)

    N = len(S)
    fig, axs = pyplot.subplots(N, 1, figsize=(32, N * 4))

    totalerror = sum(sol.error for sol in S)

    for sol, ax in zip(S, axs):
        brusheval(G, canvas, sol.g)

        original = sol.x
        nnotcovered, nredundant, npoints, diff = outcompare(original, canvas)

        filler = 2 * ones(original.shape)
        im = np.hstack((original, filler, canvas, filler, diff))

        ax.imshow(im, interpolation=None, cmap='hot', vmin=-2, vmax=2)
        ax.grid(None)
        ax.axis('off')
        ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)

        title = f"covered {npoints - nnotcovered}/{npoints} + extra {nredundant} totaling @{sol.error}"
        ax.set_title(f"{title}\n{grepr(G, sol.g)}", size=20, y=1.1)
        print(title)

    suptitle = f'Epoch #{ind} Total error = {totalerror}'
    fig.suptitle(suptitle, size=20)
    print(suptitle)

    pyplot.subplots_adjust(left=0.0, right=1, hspace=0.8)
    savefig(f"stash/{ind}.png")

@dataclass
class Beam:
    x: np.ndarray
    error: float = np.inf
    g: T = None

class H(NamedTuple):
    kalon: int
    count: int
    g: T

    def __lt__(self, o):
        return self.kalon < o.kalon

def brusheval(G, canvas, g):
    canvas.fill(0)
    evalg(G, (), g)(canvas)
    return canvas.copy()

def isuseful(moveind, g):
    "is not identity"
    qq = [g]
    while qq:
        g = qq.pop(0)
        if g.head == moveind:
            return True

        qq.extend(g.tails)

    return False

def explore(G, X, S, ns=(0, 10**6)):
    ng = (ns[1] - ns[0]) // 10
    canvas = zeros((1, *X[0].shape), np.int8)

    cerrors = zeros(len(X), np.int64)
    errors = cerrors.copy()
    errors.fill(10000)

    gs = [None for _ in range(len(X))]
    lengths = [0 for _ in range(len(X))]

    c = 0
    stime = time()
    tbar = range(*ns)

    for n in tbar:
        g = growtree(G, G.type, n=n)

        c += 1

        canvas.fill(0)
        evalg(G, (), g)(canvas[0])
        clength = length(g)

        # compare
        np.sum(np.abs(X - canvas), axis=(1, 2), out=cerrors)

        for ind in np.where(cerrors < errors)[0]:
            gs[ind] = g
            lengths[ind] = clength
            errors[ind] = cerrors[ind]

        for ind in np.where(cerrors == errors)[0]:
            if clength < lengths[ind]:
                gs[ind] = g
                lengths[ind] = clength
                errors[ind] = cerrors[ind]

        if n % ng == 0:
            print(f'{grepr(G, g)} {c / (time() - stime):.0f}/s')

    for i, z in enumerate(S):
        z.g = gs[i]
        z.error = errors[i]

    return S

def multicore(ncores, fn, args):
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
    "== while skipping holes"
    if ghost.head != tree.head or len(ghost.tails) != len(tree.tails):
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

def cutoff_at(G, g, n: int = 0):
    "cuts tree at n-level depth"
    if len(g.tails) == 0:
        return g

    tails = list(g.tails)

    if n == 1:
        tails[-1] = T(G.index('ø'))
    else:
        tails[-1] = cutoff_at(G, tails[-1], n-1)

    return g._replace(tails=tuple(tails))

def countghosts(G, alltrees, subtrees):
    matches = defaultdict(int)

    for ghost in releasetrees(G, subtrees):
        if isinstance(ghost, str) or not ghost.tails:
            continue

        c = 0
        for tree in alltrees:
            if isequalholesub(ghost, tree):
                c += 1

        matches[ghost] = c

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

def weave(G, g1, g2):
    tails = list(g1.tails)

    if tails[-1].head == G.index('ø'):
        tails[-1] = g2
    else:
        tails[-1] = weave(G, tails[-1], g2)

    return g1._replace(tails=tuple(tails))

def forceweave(G, iterind, roll, beams):
    backward = defaultdict(list)

    totalen = roll.shape[1]
    canvas = zeros(beams[0].x.shape, int)

    for beam in beams:
        canvas.fill(0)
        evalg(G, (), beam.g)(canvas)

        timings = np.where(canvas)[1]
        if timings.size == 0:
            realength = 1
        else:
            realength = timings.max() + 1

        for eind in range(realength, totalen+realength):
            ox = roll[:, eind-realength:min(totalen, eind)]
            error = np.sum(np.abs(ox - canvas[:, :ox.shape[1]]))

            # no errors allowed
            if error < 10:
                backward[eind].append((beam.g, realength, error))

    weights = [np.inf] * totalen
    weights[0] = 0
    lambdas = [None] * totalen

    for ind in range(1, totalen):
        minweight = np.inf

        for g, realength, error in backward[ind]:
            preind = ind - realength
            weight = length(g) + 10 * error + weights[preind]
            if weight < minweight:
                minweight = weight
                lambdas[ind] = (g, realength, error)

        weights[ind] = minweight

    gs = []
    ind = roll.shape[1]-1
    while ind > 0:
        g, realength, error = lambdas[ind]
        gs.append((g, realength, error))
        ind -= realength

    shards = []
    canvas = zeros(beams[0].x.shape, int)
    totalg = T(G.index('ø'))

    for ind, (_g, realength, error) in enumerate(gs):
        canvas.fill(0)
        _, y = evalg(G, (), _g)(canvas)
        # normalize or not?
        # _g = normalize(G, _g.tails[0])
        _g = _g.tails[0]

        if y != offset:
            # in principle it's not that to tackle this, but let's leave for now
            if (diff := abs(offset - y)) <= maxnumber:
                op = 'down' if y < offset else 'up'
                _g = weave(G, _g, P(G, f'({op} {diff} ø)'))

        totalg = weave(G, _g, totalg)
        shards.append(canvas[:, :realength].copy())

    # display weave

    iimshow(np.hstack(shards[::-1]))
    xcursor = -0.5
    updown = 0 # bottom | -1 is top
    yoffsets = [roll.shape[0] + 0.5, -1]
    for ind, (g, realength, error) in enumerate(gs[::-1]):
        updown = ~updown

        pyplot.text(xcursor, yoffsets[updown], grepr(G, g.tails[0]), c='black', size=12)
        pyplot.axvline(xcursor, alpha=0.7, lw=1, color='orange')
        xcursor += realength
        # if iterind == 0:
        #     yoffsets[updown] += np.sign(yoffsets[updown]) * 1

    savefig(f'stash/S{iterind}.png')

    # display whole render

    totalg = T(G.index('R'), (totalg,))
    rroll = zeros(roll.shape, int)
    evalg(G, (), totalg)(rroll)
    iimshow(rroll)
    pyplot.title(grepr(G, totalg))
    savefig(f'stash/R{iterind}.png')

    return totalg

def unweave(G, g):
    "g -> gs"
    gs = []

    while len(g.tails) > 0:
        tails = list(g.tails)
        tails[-1] = T(G.index('ø'))
        gs.append(g._replace(tails=tails))

        g = g.tails[-1]

    ind = 0
    out = []
    while ind < len(gs):
        g = gs[ind]

        # it's an invention
        if ind+1 < len(gs):
            ng = gs[ind+1]
            while ind+1 < len(gs) and ng.head < len(G.core):
                if ng.head == G.index('tap') and g.head != G.index('tap'):
                    break
                # if ng.head == G.index('loop'):
                #     break

                g = weave(G, g, ng)
                ind += 1
                if ind+1 < len(gs):
                    ng = gs[ind+1]
                else:
                    break

        out.append(g)
        ind += 1

    return out

def dream(G, trees, shape):
    spectaus = [tau for tau in G.bytype.keys()]
    # perfect square
    newshape = [int(2 ** np.ceil(np.log2(shape[0])))] * 2
    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    enc = ImageEncoder(nin=newshape[0], nout=len(spectaus) * len(G)).to(device)
    opt = th.optim.Adam(enc.parameters(), 1e-4)

    bsize = 64

    tbar = trange(32)
    for _ in tbar:
        gs = [growtree(G, G.type, randint(10**4)) for _ in range(bsize - len(trees))] + trees

        canvas = np.zeros((bsize, *newshape), dtype=np.float32)
        for ind in range(bsize):
            try:
                evalg(G, (), gs[ind])(canvas[ind])
            except TypeError:
                print(gs[ind])
                raise

        images = as_tensor(canvas, device=device).view(bsize, 1, *newshape)
        logits = enc(images).view(bsize, len(G), len(spectaus))
        logp = F.log_softmax(logits, 1)

        slogp = 0
        for imind in range(bsize):
            for ttau, gind in everytail(G, spectaus, gs[imind]):
                slogp += -logp[imind, gind, ttau]

        opt.zero_grad(); slogp.backward(); opt.step()
        tbar.set_description(f'{slogp/bsize=:.2f}')

    xs = as_tensor(X[:5], dtype=th.float32, device=device)
    xs = F.pad(xs, (0, newshape[-1]-shape[-1], 0, newshape[-2]-shape[-2]))
    xs = xs.view(-1, 1, *newshape)

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
    maxlen = 15
    rolen = 120
    roll, X, offset = almostchew('tunes/bebop_john_coltrane_lick.mid', rolen=rolen, scale=False, maxlen=maxlen)

    maxnumber = 12
    G = Language([
        Term([], '<B>', repr='ø'),
        *[Term(n, '<N>', repr=repr(n)) for n in range(1, maxnumber+1)],
        Term(add, '<N>', ['<N>', '<N>'], [['0'], [repr(n) for n in range(maxnumber)]], repr='+'),
        Term(move, '<B>', ['<B>'], repr='tap'),
        Term(neg, '<B>', ['<B>'], repr='neg'),
        Term(down, '<B>', ['<N>', '<B>'], [[], ['up', 'down']], repr='down'),
        Term(up, '<B>', ['<N>', '<B>'], [[], ['down', 'up']], repr='up'),
        Term(loop, '<B>', ['<N>', '<B>', '<B>'], [['1'], ['ø'], []], repr='loop'),
        Term(savex, '<B>', ['<B>', '<B>'], [['ø', 'savex'], ['ø']], repr='savex'),
        Term(stagebrushes(offset), '<SS>', ['<B>'], ['ø'], repr='R')
    ], type='<SS>')
# ■ ~

if __name__ == '__main__':
    if ncores == 4:
        batch = 1 * 10**6
        ncores = 4
        faststart = True
        faststart = False
    else:
        batch = 1 * 10**6
        faststart = False

    if not faststart:
        shutil.rmtree('stash', ignore_errors=True)
        os.mkdir('stash')

    for iepoch in count():
        if faststart and iepoch == 0:
            G = pickle.load(open(f'stash/G0.pkl', 'rb'))
            S = pickle.load(open(f'stash/S0.pkl', 'rb'))
        else:
            G.solder()

            progeny = []
            for n in range(1000):
                g = growtree(G, G.type, n)

                for ind, _g in enumerate(progeny):
                    if isequal(_g, g):
                        print(f'@@@ found copies {ind} & {n} {grepr(G, _g)} == {grepr(G, g)}')

                if n < 40:
                    print(grepr(G, g))

                evalg(G, (), g)

            size = (3*iepoch+1) * batch

            S = [Beam(x) for x in X]

            nss = []
            for jnd in range(ncores):
                nss.append((jnd * size, ((jnd + 1) * size) - 1))

            print(f'({iepoch}) exploring a forest spanning 10^{int(np.log10(size))} in search for {len(X)} lone pines')
            manyS = multicore(ncores, explore, zip(repeat(G), repeat(X), repeat(S), nss))

            if ncores > 1:
                for nS in manyS:
                    for zind, (nz, z) in enumerate(zip(nS, S)):
                        if nz.error < z.error or (nz.error == z.error and length(nz.g) < length(z.g)):
                            S[zind] = nz
            else:
                S = manyS

            pickle.dump(S, open(f'stash/S{iepoch}.pkl', 'wb'))
            pickle.dump(G, open(f'stash/G{iepoch}.pkl', 'wb'))

        plotbeams(G, S[:40], iepoch)

        trees = [forceweave(G, iepoch, roll, S)]

        # strip render stage
        trees = [g.tails[0] for g in trees]

        for g in trees:
            print(grepr(G, g))

        # G.reset()

        while True:
            Mx = sum(map(length, trees))

            stime = time()

            subtrees = []
            for st in flatten(map(everysubtree, trees)):
                if biggerlength(st, 1):
                    subtrees.append(cutoff_at(G, st, 5))

            manymatches = multicore(ncores, countghosts, zip(repeat(G), repeat(subtrees), nsplit(ncores, subtrees)))

            if ncores > 1:
                matches = reduce(lambda acc, x: acc | x, manymatches)
            else:
                matches = manymatches

            heap = []

            for ghost, c in matches.items():
                if not isinstance(ghost, T):
                    continue

                n, nargs = lent(ghost)

                tailtypes = []
                forceholes(ghost, tailtypes)

                # temporarily disable constants, andd numbers as arguments
                if n <= 1 or nargs == 0 or '<N>' in tailtypes:
                    continue

                Mxg = Mx - c * (n - nargs - 1)
                Mg = n

                kalon = (Mxg + Mg) / Mx

                h = H(kalon=-kalon, g=ghost, count=c)

                if len(heap) < 5:
                    heapq.heappush(heap, h)
                else:
                    heapq.heappushpop(heap, h)

            for h in heapq.nlargest(5, heap):
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
                gamma = Term(atom, G[gbody.head].type, [], repr=f"a{len(G.invented)}")
            else:
                name = f"f{len(G.invented)}"

                gamma = Term(gbody, G[gbody.head].type, list(reversed(tailtypes)), repr=f"f{len(G.invented)}")

            print(f'growing new leaf {grepr(G, gbodywithholes)} with tails {tailtypes} @{-h.kalon:.2f} #{h.count}')

            G.add(gamma)
            gind = len(G)-1

            canvas = zeros(S[0].x.shape, int)
            outbefore = [brusheval(G, canvas, T(G.index('R'), (g,))) for g in trees]

            trees = [rewrite(g, gbodywithholes, T(gind), withargs=not isatom) for g in trees]
            outafter = [brusheval(G, canvas, T(G.index('R'), (g,))) for g in trees]

            for ind, (outa, outb) in enumerate(zip(outbefore, outafter)):
                if (outa - outb).sum() != 0:
                    raise ValueError(f'bad rewrite @ {trees[ind]}')

        for g in G.invented:
            print(f'{g} {grepr(G, g.head)} : {" -> ".join(g.tailtypes)} -> {g.type}')

        print(trees[0])
        G.infer()

        # show abstractions
        canvas = zeros(roll.shape, int)
        evalg(G, (), T(G.index('R'), (trees[0],)))(canvas)
        iimshow(canvas)

        gs = unweave(G, trees[0])
        brushes = []
        brushes_starts = [0]

        for g in gs:
            bs = evalg(G, (), g)
            x, y = stage(offset, brushes + bs)(canvas)
            brushes.extend(bs)
            brushes_starts.append(x)

        updown = 0 # bottom | -1 is top
        yoffsets = [-2, roll.shape[0]+1]
        for ind, g in enumerate(gs):
            updown = ~updown

            xcursor = brushes_starts[ind] - 0.5
            pyplot.text(xcursor, yoffsets[1], grepr(G, g), c='black', size=12, rotation='vertical')
            pyplot.axvline(xcursor, alpha=0.7, lw=1, color='orange')
            yoffsets[updown] += np.sign(yoffsets[updown]) * 1

        pyplot.text(-0.5, yoffsets[1], grepr(G, trees[0]), size=12)
        for ind, g in enumerate(G.invented):
            pyplot.text(-0.5, yoffsets[1]+1+ind, f'{g.repr} >>> {grepr(G, g.head)}', size=12)

        savefig(f'stash/A{iepoch}.png')

        trees = [T(G.index('R'), (g,)) for g in trees]

        # i need to sample, but do i need to do it with a uniform logp?
        G.solder()
        Q = dream(G, trees, X[0].shape)
        G.solder(Q)
