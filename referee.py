from importblob import *

class Gamma(NamedTuple):
    head: int
    type: type
    tailtypes: list = []
    repr: str = ""
    forbidden: list = []

    def __repr__(self):
        return self.repr

class Gammas:
    def __init__(self, gammas, requested_type):
        self.gammas = gammas
        self.requested_type = requested_type

    def __getitem__(self, ind):
        return self.gammas[ind]

    def __len__(self):
        return len(self.gammas)

    def index(self, repr: str):
        for ind, g in enumerate(self.gammas):
            if g.repr == repr:
                return ind

        return None

    def bytype(self, type):
        return findall(ap(lambda g: g.type == type, self))

    @lru_cache
    def view(self, tau: type):
        "returns sub-G on a type"
        return [self.gammas[ind] for ind in self.bytype(tau)]


class Gm(NamedTuple):
    head: int
    tails: tuple = tuple()

    def __repr__(self):
        if not self.tails:
            return G[self.head].repr

        return f"({G[self.head].repr} {' '.join(map(str, self.tails))})"

def taurepr(G, tau, g):
    Gt = G.view(tau)
    tailtypes = Gt[g.head].tailtypes
    tailreprs = [taurepr(G, tau, tail) for tau, tail in zip(tailtypes, g.tails)]

    if not tailreprs:
        return Gt[g.head].repr

    return f"({Gt[g.head].repr} {' '.join(tailreprs)})"

@lru_cache(maxsize=None)
def evalg(G, args, tau, g):
    # debruijn index
    if isinstance(g.head, int) and g.head < 0:
        return args[g.head]

    # abstraction
    if isinstance(g.head, Gm):
        # what tau means here?
        return evalg(G, g.tails, tau, g.head)

    # atom
    if not g.tails:
        return G.view(tau)[g.head].head

    # application
    gamma = G.view(tau)[g.head]
    tails = tuple([evalg(G, args, tau, tail) for tau, tail in zip(gamma.tailtypes, g.tails)])
    return gamma.head(*tails)

G = Gammas([
    Gamma(1, int, repr='1'),
    Gamma(add, int, [int, int], forbidden=[[], []], repr='+'),
], requested_type=int)

G = Gammas([
    Gamma(1, int, repr='1'),
    Gamma(π, float, repr='π'),
    Gamma(add, float, [int, float], forbidden=[[], []], repr='+'),
    Gamma(add, int, [int, int], forbidden=[[], []], repr='+'),
], requested_type=float)

def getns(G: List[Gamma]) -> (int, int):
    nops, natoms = 0, 0

    for g in G:
        if not g.tailtypes:
            natoms += 1
        else:
            nops += 1

    return (natoms, nops)

def fancysplit(base, n):
    n, op = divmod(n, base)

    ind = 0
    nsplits = 2
    numbers = [0] * nsplits
    while n > 0:
        n, bit = divmod(n, base)

        nbit, strand = divmod(ind, nsplits)
        numbers[strand] += bit * base ** nbit

        ind += 1

    return [op] + numbers

def singlesplit(n):
    ind = 0
    nsplits = 2
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
    natoms, nops = getns(G.view(tau))
    if n < natoms:
        return Gm(n)

    # underflowing base 2
    if nops == 1:
        tails = singlesplit(n-natoms)
        tailtypes = G.view(tau)[natoms].tailtypes
        return Gm(natoms, tuple([maketree(G, tau, n) for tau, n in zip(tailtypes, tails)]))

    head, *tails = fancysplit(nops, n-natoms)
    return Gm(head + natoms, tuple([maketree(tau, nops, natoms, tail) for tail in tails]))

tbar = trange(10**5)
for n in tbar:
    tree = maketree(G, G.requested_type, n=n)
    # print(taurepr(G, G.requested_type, tree))
    evalg(G, (), G.requested_type, tree)
