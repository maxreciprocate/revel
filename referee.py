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
    def __init__(self, gammas, requested_type):
        self.gammas = gammas
        self.requested_type = requested_type

        self.infer()

    def infer(self):
        self.bytype = defaultdict(list)

        for gind, g in enumerate(self.gammas):
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
                    newtype = f"{g.repr}.t{tailind}"

                    self.bytype[newtype] = newbytype
                    newtailtypes.append(newtype)

            self.gammas[gind] = Gamma(g.head, g.type, newtailtypes, forbidden, g.repr)

        self.rbytype = {}
        for tau, ginds in self.bytype.items():
            self.rbytype[tau] = [self.gammas[gind] for gind in ginds]

    def __getitem__(self, ind):
        return self.gammas[ind]

    def __len__(self):
        return len(self.gammas)

    def index(self, repr: str):
        for ind, g in enumerate(self.gammas):
            if g.repr == repr:
                return ind

        return None

    @lru_cache
    def view(self, tau: type):
        "returns sub-G on a type"
        inds = self.bytype[tau]
        mapping = {cind: ind for cind, ind in zip(count(0), inds)}

        Gt = [self.gammas[ind] for ind in inds]
        natoms, nops = getns(Gt)
        # the number of splits for each non-atom
        nsplitmap = [len(self[i].tailtypes) for i in inds[natoms:]]
        return Gt, mapping, nsplitmap

class Gm(NamedTuple):
    head: int
    tails: tuple = tuple()

    def __repr__(self):
        if not self.tails:
            return G[self.head].repr

        return f"({G[self.head].repr} {' '.join(map(str, self.tails))})"

    def __len__(self):
        return 1 + sum(map(len, self.tails))

@lru_cache(maxsize=None)
def evalg(G, args, g):
    # debruijn index
    if isinstance(g.head, int) and g.head < 0:
        return args[g.head]

    # abstraction
    if isinstance(g.head, Gm):
        return evalg(G, g.tails, g.head)

    # atom
    if not g.tails:
        return G[g.head].head

    # application
    gamma = G[g.head]
    tails = tuple([evalg(G, args, tail) for tail in g.tails])
    return gamma.head(*tails)

G = Gammas([
    Gamma(1, int, repr='1'),
    Gamma(add, int, [int, int], forbidden=[[], []], repr='+'),
], requested_type=int)

G = Gammas([
    Gamma(1, int, repr='1'),
    Gamma(add, int, [int, int], forbidden=[[], ['+']], repr='+'),
], requested_type=int)

G = Gammas([
    Gamma(1, int, repr='1'),
    Gamma(π, float, repr='π'),
    Gamma(add, float, [int, float], forbidden=[[], []], repr='+'),
    Gamma(add, int, [int, int], forbidden=[[], []], repr='+'),
], requested_type=float)

G = Gammas([
    Gamma(1, float, repr='1'),
    Gamma(2, float, repr='2'),
    Gamma(add, float, [float, float], [['+'], []], repr='+'),
    Gamma(mul, float, [float, float], [['1'], ['1']], repr='*'),
    Gamma(truediv, float, [float, float], [[], ['1']], repr='÷'),
], requested_type=float)

ss = "ss"
G = Gammas([
    Gamma(1, int, repr='1'),
    Gamma(π/100, 'angle', repr='ε'),
    Gamma(20, int, repr='inf'),
    Gamma(noop, ss, repr='ø'),
    Gamma(divpi, 'angle', [int, int], [['inf'], ['1', 'inf']], repr='π'),
    Gamma(succ, int, [int], [['inf']], repr='S'),
    Gamma(neg, 'angle', ['angle'], [['-']], repr='-'),
    Gamma(mv, ss, [int, 'angle', ss], forbidden=[['-1', '0', 'inf'], [], []], repr='mv'),
    Gamma(se, ss, [ss, ss], forbidden=[['savex', 'ø', 'loop'], ['savex', 'ø']], repr='savex'),
    Gamma(lp, ss, [int, ss, ss], forbidden=[['-1', '0', '1'], ['loop', 'ø'], ['loop']], repr='loop'),
    Gamma(pu, ss, [ss, ss], forbidden=[['penup', 'ø', 'loop', 'savex'], ['penup', 'ø']], repr='penup'),
], requested_type=ss)

def getns(G: List[Gamma]) -> (int, int):
    "gives nops, natoms for G"
    nops, natoms = 0, 0

    for g in G:
        if not g.tailtypes:
            natoms += 1
        else:
            nops += 1

    return natoms, nops

def fancysplit(base, nsplitmap, n):
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

def singlesplit(nsplits, n):
    ind = 0
    base = 2
    numbers = [0] * nsplits
    while n > 0:
        n, bit = divmod(n, base)

        nbit, strand = divmod(ind, nsplits)
        numbers[strand] += bit * base ** nbit

        ind += 1

    return numbers

@lru_cache(maxsize=None)
def maketree(G: Gammas, tau: type, n: int) -> Gm:
    Gt, mapping, nsplitmap = G.view(tau)
    natoms, nops = getns(Gt)
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

G.rbytype

# ■ ~

@dataclass
class Beam:
    x: np.ndarray
    ax: np.ndarray = None
    error: float = np.inf
    tree: Gm = None

from fontrender import alfbet
X = alfbet.astype(np.int8)

def brusheval(G, im, tree):
    im.fill(0)
    state = (im, 0, 0, 0, True)
    out = evalg(G, (), tree)(state)[0]
    return out

@njit
def compare(x, ax):
    return np.abs(x - ax).sum()

def explore(G, X, Z):
    im = zeros(X[0].shape, np.int8)

    tbar = trange(10**5)
    for n in tbar:
        tree = maketree(G, G.requested_type, n=n)
        ax = brusheval(G, im, tree)

        for z in Z:
            error = compare(z.x, ax)

            if error < z.error:
                z.error = error
                z.ax = ax.copy()
                z.tree = tree

    return Z

Z = explore(G, X, [Beam(x) for x in X])

# ■ ~

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
savefig("stash/z.png")

# ■ ~

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
    yield from flatten(combinations(xs, n) for n in range(len(xs) + 1))

@lru_cache(maxsize=None)
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

def countmatches(G, matches, st, t):
    if not st.tails or not t.tails:
        return

    if st.head == t.head:
        for match in forcematch(G, st, t):
            matches[match] += 1

    for tail in t.tails:
        countmatches(G, matches, st, tail)

subtrees = set(flatten(map(everysubtree, (z.tree for z in Z))))

matches = defaultdict(int)
tbar = tqdm(subtrees)
for st in tbar:
    length = len(st)

    for zind, z in enumerate(Z):
        countmatches(G, matches, st, z.tree)
        tbar.set_description(f'[{zind}/{len(Z)}] {length=} {repr(st)[:50]}')

@lru_cache(maxsize=None)
def lent(t):
    # args + types (must be merged)
    if not isinstance(t, Gm):
        return [0, 1]

    return list(reduce(lambda acc, x: [acc[0] + x[0], acc[1] + x[1]], map(lent, t.tails), [1, 0]))

heap = []

class H(NamedTuple):
    size: int
    tree: Gm

    def __lt__(self, o):
        return self.size < o.size

for tree, c in tqdm(matches.items()):
    if not isinstance(tree, Gm):
        continue

    n, nargs = lent(tree)
    if n <= 1:
        continue

    size = (n - nargs) * c
    h = H(size=size, tree=tree)

    if len(heap) < 20:
        heapq.heappush(heap, h)
    else:
        heapq.heappushpop(heap, h)

print(heapq.nlargest(20, heap))
