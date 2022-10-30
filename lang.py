from blob import *
from q import *

class T(NamedTuple):
    head: int
    tails: tuple = tuple()

    def __repr__(self):
        if self.tails:
            return f"({self.head} {' '.join(map(repr, self.tails))})"

        return str(self.head)

class Term(NamedTuple):
    head: int
    type: type
    tailtypes: list = []
    forbidden: list = []
    repr: str = '?'

    def __repr__(self):
        return self.repr

def tolang(L, ast) -> T:
    match ast:
        case [ast] if isinstance(ast, list):
            return tolang(L, ast)
        case [head, *tails] if len(tails):
            return T(L.index(head), tuple(tolang(L, t) for t in tails))
        case [var]:
            return tolang(L, var)
        case hole if hole[0].isupper() or hole[0] == '.' or hole[0] == '$':
            return hole
        case head if (ind := L.index(head)) is not None:
            return T(ind)
        case debruijn if (index := int(debruijn)) < 0:
            return T(index)
        case _:
            raise ValueError(f"{ast}, it's all greek to me")

class Language:
    def __init__(self, core, type=None):
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
                generictype = f"{tailtype.split(':')[0]}" if ':' in tailtype else tailtype
                # sieve forbidden ginds
                ginds = [ind for ind in self.bytype[generictype] if not self[ind].repr in forbidden]
                specifictype = f"{generictype}:{g.repr}:{tind}"

                self.bytype[specifictype] = ginds
                newtailtypes.append(specifictype)

            self.gammas[gind] = Term(g.head, g.type, newtailtypes, g.forbidden, g.repr)

        bytype = {tau: array(ginds) for tau, ginds in self.bytype.items()}
        self.bytype = bytype
        self.rbytype = {tau: [self.gammas[gind] for gind in ginds] for tau, ginds in self.bytype.items()}

    def solder(self, Q=None):
        if Q is None:
            Q = {tau: -np.log2(ones(len(gs))/len(gs)) for tau, gs in self.bytype.items()}

        taus = list(self.bytype.keys())
        for tau in taus:
            qs, sinds = np.sort(Q[tau]), np.argsort(Q[tau])
            qs = ap(lambda x: round(x, 1), qs)
            gs = ap(lambda gind: self[gind].repr, np.atleast_1d(array(self.bytype[tau])[sinds]))
            print(f'{tau} ~ {list(zip(gs, qs))}')

        self.views = multicore(multiview, zip(repeat(self), repeat(Q), nsplit(ncores, taus)))
        if isinstance(self.views, list):
            self.views = reduce(lambda acc, x: acc | x, self.views, {})

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

    def __mod__(self, expr: str):
        # return interpret(self, parse(expr))
        return tolang(self, parse(expr))

    def __call__(self, term: T):
        if isinstance(term, str):
            term = interpret(self, parse(term))

        return evalg(self, (), term)

    def __lt__(self, term: T):
        return grepr(self, term)

def grepr(L, g):
    if not isinstance(g, (T, int)):
        return g

    if isinstance(g.head, T):
        if g.tails:
            return f"(λ[{g.head}]. {' '.join([grepr(L, t) for t in g.tails])})"

        return f"λ[{g.head}]"

    if isinstance(g.head, int) and g.head < 0:
        return str(g.head)

    if not g.tails:
        return L[g.head].repr

    return f"({L[g.head].repr} {' '.join([grepr(L, t) for t in g.tails])})"

def length(g: T) -> int:
    if isinstance(g, (int, str)):
        return 1

    return 1 + sum(map(length, g.tails))

def evalg(G, args, g):
    # debuijn index
    if isinstance(g.head, int) and g.head < 0:
        return args[g.head]

    # atom
    if not g.tails:
        return G[g.head].head

    # application
    gamma = G[g.head]
    tails = tuple([evalg(G, args, tail) for tail in g.tails])

    # abstraction
    if isinstance(gamma.head, T):
        return evalg(G, tails, gamma.head)

    return gamma.head(*tails)

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

def maxdepth(g):
    if isinstance(g, str | int):
        return 0

    if not g.tails:
        return 0

    return 1 + max(maxdepth(tail) for tail in g.tails)

def lent(t: T | str):
    "natoms+nops, nargs"
    # types
    if isinstance(t, str):
        return [0, 1]
    # eclass
    if isinstance(t, int):
        return [1, 0]

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

def parse(s: str):
    "Parse a string into an AST"
    ast = []
    ind = 0

    while ind < len(s):
        if s[ind] == '(':
            nopen = 1
            sind = ind

            while nopen != 0:
                ind += 1
                if s[ind] == ')':
                    nopen -= 1
                if s[ind] == '(':
                    nopen += 1

            ast.append(parse(s[sind+1:ind]))

        else:
            term = []
            while ind < len(s) and not s[ind].isspace():
                term.append(s[ind])
                ind += 1

            if term:
                ast.append(''.join(term))

        ind += 1

    return ast

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
