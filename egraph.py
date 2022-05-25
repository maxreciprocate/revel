from blob import *
from lang import *
from rich import print

# ■ ~

# back then, those were the times
apply = lambda f, *args: f(*args)

EClass = int

class ENode(NamedTuple):
    head: int
    tails: tuple = tuple()

    def __repr__(self):
        if not self.tails:
            return f'<{self.head}>'

        return f'<{self.head} {" ".join(map(repr, self.tails))}>'

class EGraph:
    def __init__(self, cost=lambda egraph, enode: 0, ctxfn=lambda egraph, enode: None):
        self.parents = []
        self.values = []

        self.cost = partial(cost, self)
        self.ctxfn = partial(ctxfn, self)
        self.analyses = []
        self.hashcons = {}

    def add(self, value=None):
        self.parents.append(-1)
        self.values.append({value})
        self.analyses.append([self.cost(value), value, self.ctxfn(value)])
        return len(self) - 1

    def find(self, ind):
        while self.parents[ind] >= 0:
            ind = self.parents[ind]

        return ind

    def union(self, a, b):
        aroot = self.find(a)
        broot = self.find(b)

        if aroot == broot:
            return aroot

        asize = -self.parents[aroot]
        bsize = -self.parents[broot]

        # aset should be the smaller one
        if asize > bsize:
            aroot, broot = broot, aroot
            asize, bsize = bsize, asize

        self.parents[broot] -= asize
        self.parents[aroot] = broot
        self.values[broot] = self.values[broot] | self.values[aroot]
        self.values[aroot] = set()

        # ctx should remain the same
        bctx = self.analyses[broot][2]
        actx = self.analyses[aroot][2]

        ctx = bctx or actx or None

        if self.analyses[broot][0] > self.analyses[aroot][0]:
            self.analyses[broot] = self.analyses[aroot]

        self.analyses[broot][2] = ctx

        if isinstance(self.analyses[aroot][2], int):
            self.analyses[broot][2] = self.analyses[aroot][2]

        return broot

    def canonize(self, f: ENode) -> ENode:
        # cannot cannonize anything with holes
        if any(not isinstance(tail, EClass) for tail in f.tails):
            return None

        return ENode(f.head, tuple(self.find(a) for a in f.tails))

    def gcanonize(self, f: ENode) -> ENode:
        if not isinstance(f, ENode):
            return f

        tails = []
        for t in f.tails:
            if not isinstance(t, EClass):
                tails.append(t)
            else:
                tails.append(self.find(t))

        return ENode(f.head, tuple(tails))

    def merge(self, a: EClass, b: EClass):
        id = self.union(a, b)
        eclass = set()

        for enode in self.values[id]:
            self.hashcons.pop(enode, None)
            enode = self.canonize(enode)
            self.hashcons[enode] = id
            eclass.add(enode)

        self.values[id] = eclass

    def addenode(self, enode: ENode) -> EClass:
        enode = self.canonize(enode)
        if enode in self.hashcons:
            return self.hashcons[enode]

        id = self.add(enode)
        self.hashcons[enode] = id
        return id

    def addexpr(self, expr: T | str, retenode=False) -> EClass:
        if not isinstance(expr, T):
            return expr

        tails = [self.addexpr(t) for t in expr.tails]
        enode = ENode(expr.head, tuple(tails))

        if any(not isinstance(t, EClass) for t in tails):
            retenode = True
        else:
            eclass = self.addenode(enode)

        if retenode:
            return enode
        return eclass

    def addarray(self, xs) -> EClass:
        x, *xs = xs
        tails = (self.addarray(xs),) if xs else tuple()
        return self.addenode(ENode(str(x), tails))

    def congruences(self):
        out = []

        # find e-nodes that changed
        for id, eclass in enumerate(self.values):
            if not eclass: continue

            for enode in eclass:
                newenode = self.canonize(enode)

                if newenode in self.hashcons:
                    newid = self.hashcons[newenode]
                else:
                    newid = self.addenode(newenode)

                if newid != id:
                    out.append((id, newid))

        return out

    def rebuild(self):
        while (cs := self.congruences()):
            for a, b in cs:
                self.merge(a, b)

    def match(self, pattern: ENode | str, enode: ENode):
        if isinstance(pattern, str):
            return {pattern: self.hashcons[enode]}, self.hashcons[enode]

        if pattern.head == enode.head and len(pattern.tails) == len(enode.tails):
            possibletails = [[] for eclass in enode.tails]
            subst = {}

            for ind, (pt, taileclass) in enumerate(zip(pattern.tails, enode.tails)):
                # if a pattern's tail is a hole, store subst and figure out c
                if isinstance(pt, str):
                    # matches are not consistent
                    if pt in subst and subst[pt] != taileclass:
                        return None

                    possibletails[ind].append((None, taileclass))
                    subst[pt] = taileclass
                elif isinstance(pt, EClass):
                    # if not the same eclass
                    if taileclass != pt:
                        return None

                    possibletails[ind].append((None, pt))
                elif isinstance(pt, ENode):
                    # pt has holes
                    # there are a lot of enodes in this eclass
                    for tailenode in self.values[taileclass]:
                        if (sigmacs := self.match(pt, tailenode)):
                            possibletails[ind].extend(sigmacs)

            out = []
            for tails in product(*possibletails):
                newsubst = copy(subst)
                consistent = True

                # make sure substs are consistent
                for tailsubst, _ in tails:
                    if not tailsubst: continue

                    for k, v in tailsubst.items():
                        if k not in newsubst:
                            newsubst[k] = v
                        elif newsubst[k] != v:
                            consistent = False
                            break

                    if not consistent:
                        break

                if not consistent:
                    continue

                newtails = [x[1] for x in tails]
                # who will we become if applied this rewrite?
                c = self.addenode(ENode(pattern.head, tuple(newtails)))
                out.append((newsubst, c))

            return out

    def saturate(self, rws, times=10):
        for _ in range(times):
            saturated = True

            matches = []
            for lhs, rhs in rws:
                for enode in chain(*self.values):
                    if (matchlist := self.match(lhs, enode)):
                        for m in matchlist:
                            matches.append((rhs, *m))

            for rhs, subst, lhseclass in matches:
                match rhs:
                    # introducing new subst in rhs
                    case [rhs, bonus] if isinstance(bonus, dict):
                        for var, (f, args) in bonus.items():
                            for arg, eclass in subst.items():
                                args = args.replace(arg, str(eclass))

                            if (expr := apply(f, *eval(args))) is not None:
                                subst.update({var: expr})

                    case [rhs, condition] if isinstance(condition, str):
                        for arg, eclass in subst.items():
                            state = self.analyses[eclass][2]
                            condition = condition.replace(arg, str(state))

                        if 'None' in condition:
                            continue

                        try:
                            if not eval(condition):
                                continue
                        except ValueError:
                            continue

                    case rhs:
                        pass

                rhseclass = self.rewritesubst(subst, rhs)

                if lhseclass != rhseclass:
                    saturated = False

                self.merge(lhseclass, rhseclass)

            self.rebuild()

            if saturated:
                return len(matches)

    def rewritesubst(self, subst: dict, term: T):
        if isinstance(term, str):
            return subst[term]

        if not term.tails:
            return self.addenode(ENode(term.head, tuple()))

        newtails = list(term.tails)
        for ind, t in enumerate(term.tails):
            if t in subst:
                newtails[ind] = subst[t]
            else:
                newtails[ind] = self.rewritesubst(subst, t)

        return self.addenode(ENode(term.head, tuple(newtails)))

    def extract(self, eclass, fn=T):
        cost, enode, _ = self.analyses[self.find(eclass)]
        tails = [self.extract(eclass, fn=fn)[1] for eclass in enode.tails]
        return cost, fn(enode.head, tails=tuple(tails))

    def __len__(self):
        return len(self.parents)

    def __getitem__(self, ind):
        return self.values[self.find(ind)]

    def __repr__(self):
        print(self.hashcons)
        print(self.values)
        print(self.analyses)
        print(self.parents)
        return ''

def prettyenode(L: Language, G: EGraph, enode: ENode):
    match enode:
        case str(free):
            return free
        case int(eclass):
            return prettyenode(L, G, G.analyses[G.find(eclass)][1])
        case ENode(head, tails):
            head = L[head].repr

            if len(tails) == 0:
                return head
            else:
                return f'({head} {" ".join(map(partial(prettyenode, L, G), tails))})'
        case _:
            raise ValueError(f'malformed part {enode}: {type(enode)}')

def preprocess(pattern, vars=[]):
    if isinstance(pattern, str):
        vars.insert(0, pattern)
        return T(-len(vars))

    if not pattern.tails:
        return pattern

    newtails = list(pattern.tails)
    for ind, tail in enumerate(newtails):
        newtails[ind] = preprocess(tail, vars)

    return T(pattern.head, tuple(newtails))

def termlength(G: EGraph, enode) -> int:
    return 1 + sum(G.analyses[t][0] for t in enode.tails)

def rewrites(L, *rws):
    return [(L% lhs, L% rhs) for lhs, rhs in [rw.split('~>', 1) for rw in rws]]

def cons(x, xs):
    return [x] + xs

def fuse(L: Language, G: EGraph, fusewith: Tuple[int, int], fuseon: int, a: EClass, b: EClass) -> EClass:
    enodes = []

    c, enode, _ = G.analyses[G.find(a)]
    if enode.head not in fusewith:
        return None

    while enode.head != fuseon:
        enodes.append(enode)
        c, enode, _ = G.analyses[G.find(enode.tails[-1])]

    extratail = b
    for enode in reversed(enodes):
        newenode = ENode(enode.head, (*enode.tails[:-1], extratail))
        extratail = G.addenode(newenode)

    return extratail

@lru_cache(maxsize=1<<20)
def ghosterm(L: Language, t: T) -> List[T]:
    if not t.tails:
        return (L[t.head].type, t)

    tails = map(partial(ghosterm, L), t.tails)
    return [L[t.head].type] + ap(partial(T, t.head), product(*tails))

@lru_cache(maxsize=1<<20)
def ghostbust(L: Language, G: EGraph, depth = 4, eclass: EClass = None) -> List[T]:
    if eclass is None:
        return chain(map(partial(ghostbust, L, G, depth - 1), range(len(G))))

    out = set()
    for t in G.values[eclass]:
        out.add(L[t.head].type)

        if depth == 0:
            continue

        if not t.tails:
            out.add(eclass)
            continue

        tails = map(partial(ghostbust, L, G, depth - 1), t.tails)
        out |= set(map(partial(ENode, t.head), product(*tails)))

    return out

def countsimilar(L: Language, G, a, b):
    if isinstance(a, str) or isinstance(b, str):
        return 1

    if a.head != b.head:
        return 0

    return prod([countsimilars(L, G, at, bt)] for at, bt in zip(a.tails, b.tails))

@lru_cache(maxsize=1<<20)
def countsimilars(L: Language, G: EGraph, pattern: T, eclass: EClass = None) -> int:
    if eclass is None:
        return sum(map(partial(countsimilars, L, G, pattern), range(len(G))))

    if isinstance(pattern, str):
        return len(G.values[eclass])

    c = 0
    for f in G.values[eclass]:
        if f.head != pattern.head or len(f.tails) != len(pattern.tails):
            continue

        c = prod(list(starmap(partial(countsimilars, L, G), zip(pattern.tails, f.tails))))

    return int(c)

def lentqenode(G, t):
    if isinstance(t, str):
        return [0, 1]

    if isinstance(t, int):
        x = [G.analyses[t][0], 0]
        return x

    return list(reduce(lambda acc, x: [acc[0] + x[0], acc[1] + x[1]], map(partial(lentqenode, G), t.tails), [1, 0]))

def contract(G: EGraph, enode: ENode):
    tails = []
    for tail in enode.tails:
        if isinstance(tail, EClass | str):
            tails.append(tail)
        elif (newtail := G.canonize(tail)) is not None:
            tails.append(G.hashcons[newtail])
        else:
            x = contract(G, tail)
            if x in G.hashcons:
                tails.append(G.hashcons[x])
            else:
                tails.append(x)

    return ENode(enode.head, tuple(tails))

@lru_cache(maxsize=1 << 30)
def ghostsingle(L: Language, G: EGraph, depth = 3, eclass: EClass = None):
    if eclass is None:
        return chain(map(partial(ghostsingle, L, G, depth - 1), range(len(G))))

    out = set()
    for t in G.values[eclass]:
        out.add(L.gammas[t.head].type)

        if not t.tails:
            out.add(eclass)

        if depth == 0:
            continue

        tails = list(t.tails)
        for ind in range(len(tails)):
            for ghostail in ghostsingle(L, G, depth-1, tails[ind]):
                tails[ind] = ghostail
                nt = t._replace(tails=tuple(tails))
                out.add(contract(G, nt))
                tails[ind] = t.tails[ind]

    return out

@lru_cache(maxsize=1<<20)
def expandroot(G, maxcomp, eclass):
    if len(G[eclass]) == 0:
        raise ValueError(f"missing {eclass=}")

    out = []
    for enode in G[eclass]:
        if not enode.tails:
            out.append(enode)

        tails = [expandroot(G, maxcomp-1, t) for t in enode.tails]
        if len(tails) == 0: continue

        tails = list(product(*tails))
        out.extend([ENode(enode.head, tuple(ts)) for ts in tails])

    out = [f for f in out if np.isfinite(isslimenough(G, f, maxcomp))]
    return out

def isslimenough(G, f, limit):
    if not isinstance(f, ENode):
        return 1

    if not np.isfinite(weightenode(G, ENode(f.head))):
        return np.inf

    l = 1

    for t in f.tails:
        if isinstance(t, EClass):
            l += G.analyses[t][0]
        elif isinstance(t, str):
            l += 1
        elif (ct := contract(G, t)) in G.hashcons:
            l += weightenode(G, ct)
        else:
            l += isslimenough(G, t, limit)

        limit -= l
        if limit < 0:
            return np.inf

    return l

@lru_cache(maxsize=1<<20)
def isequalholenode(G, a, b):
    if isinstance(a, EClass) and isinstance(b, EClass):
        return a == b

    if isinstance(a, EClass):
        return contract(G, b) in G[a]
    if isinstance(b, EClass):
        return contract(G, a) in G[b]

    if a.head != b.head or len(a.tails) != len(b.tails):
        return False

    for atail, btail in zip(a.tails, b.tails):
        # hole
        if isinstance(atail, str) or isinstance(btail, str):
            continue

        if not isequalholenode(G, atail, btail):
            return False

    return True

@lru_cache(maxsize=1<<20)
def countin(G, source, a):
    if not isinstance(source, ENode) or not isinstance(a, ENode):
        return 0

    rest = sum([countin(G, tail, a) for tail in source.tails])

    if isequalholenode(G, source, a):
        return 1 + rest

    return rest

def samplepath(G: EGraph, f: ENode) -> ENode:
    tails = []
    for t in f.tails:
        nt = list(G[t])[randint(len(G[t]))]
        nt = samplepath(G, nt)
        tails.append(nt)

    return ENode(f.head, tuple(tails))

def kkstack(G, xs, ghosts):
    ghostheap = Heap(100)

    xlengths = [lentqenode(G, x)[0] for x in xs]
    offset = np.min(xlengths)

    for ogk, ghost in ghosts:
        n, nargs = lentqenode(G, ghost)
        lghost = n + nargs

        k = 0
        for x, lx in zip(xs, xlengths):
            c = countin(G, x, ghost)
            if c > 0:
                s = c * (n - nargs - 1) - lghost
            else: # we're always taking some extra ghosts
                s = -np.inf

            k += np.exp2(offset - lx + s) * ogk
            # k += np.exp2(+s) * ogk

        ghostheap.push(k, ghost)

    return ghostheap

def kkalon(L, G, source, history, rules, depth, verb=False):
    sroot = G.addexpr(source)
    G.saturate(rules)
    sroot = G.find(sroot)
    ghosts = set(filter(lambda g: length(g) > 1, chain(*ghostsingle(L, G, depth=depth))))
    ghosts = [(1, ghost) for ghost in ghosts]
    if verb: print(f'{len(ghosts)=}')

    xs = list(set([samplepath(G, random.choice(list(G[sroot]))) for _ in range(10**3)]))
    xs.append(G.extract(sroot, ENode)[1])

    heaps = multicore(kkstack, zip(repeat(G), repeat(xs), nsplit(ncores, ghosts)))
    ghosts = [(k, ghost) for k, ghost in chain(*heaps) if length(ghost) > 1 and k > 1e-9]
    lenghosts = len(ghosts)

    if verb:
        print('oneself')
        for ind, (k, ghost) in enumerate(sorted(ghosts, reverse=True, key=lambda x: x[0])):
            if ind < 10:
                print(prettyenode(L, G, ghost), log(k))

    hroot = G.addexpr(history)
    G.saturate(rules)
    hroot = G.find(hroot)
    hxs = list(set([samplepath(G, random.choice(list(G[hroot]))) for _ in range(10**3)]))
    hxs.append(G.extract(sroot, ENode)[1])

    heaps = multicore(kkstack, zip(repeat(G), repeat(hxs), nsplit(ncores, ghosts)))

    if verb: print('history')

    kalon = 0
    for ind, (k, ghost) in enumerate(sorted(chain(*heaps), reverse=True)):
        kalon += k
        if verb and ind < 10:
            print(prettyenode(L, G, ghost), log(k))

    return kalon / len(ghosts), sum(lent(xs[-1]))

# count how much is in the free variable
# check if the body matches, extract eclasses of the match
def ghostpresence(G, ghost, enode) -> int:
    count = 1

    if ghost.head == enode.head and len(ghost.tails) == len(enode.tails):
        for tail, eclass in zip(ghost.tails, enode.tails):
            if isinstance(tail, str) or tail == eclass:
                # count *= len(G.values[eclass])
                pass
            elif isinstance(tail, ENode):
                c = 0
                for tailenode in G.values[eclass]:
                    c += ghostpresence(G, tail, tailenode)

                count *= c
            else:
                return 0

        return count

    return 0

# ought to use forceholes with debrujin instead of dicts and substs
def renameholes(t: ENode, newnames: list):
    if not t.tails:
        return t

    newtails = [None] * len(t.tails)
    for tind, tail in enumerate(t.tails):
        if isinstance(tail, str):
            var = f'V{len(newnames)}'
            newtails[tind] = var
            newnames.append(var)
        elif isinstance(tail, int):
            newtails[tind] = tail
        else:
            newtails[tind] = renameholes(tail, newnames)

    return t._replace(tails=tuple(newtails))

def countails(G: EGraph) -> dict:
    counts = defaultdict(int)
    for n in chain(*G.values):
        for eclass in n.tails:
            counts[eclass] += 1

    return counts

@lru_cache
def nvarios(G: EGraph, eclass: EClass) -> int:
    "the number of ways of expressing eclass"
    c = 0

    for enode in G.values[G.find(eclass)]:
        c += prod(ap(P(nvarios, G), enode.tails))

    return c

def force(L: Language, G: EGraph):
    counts = countails(G)
    maxs, maxghost = 0, None

    for ghost in set(chain(*ghostbust(L, G))):
    # for ghost in set(chain(*ghostsingle(L, G))):
        if isinstance(ghost, str) or length(ghost) == 1: continue

        c = 0
        for eclass in range(len(G)):
            c += counts[eclass] * countsimilars(L, G, ghost, eclass)

        n, nargs = lent(ghost)
        if nargs == 0: continue

        saved = c * (n - nargs) - length(ghost)
        if saved > 0:
            print(ghost, saved)

        if saved > maxs:
            maxs = saved
            maxghost = ghost

    if maxghost is None:
        return maxghost

    name = f'f{len(L.invented)}'
    gamma = Term(maxghost, L[maxghost.head].type, repr=name)
    L.add(gamma)

    vars = []
    ghost = renameholes(maxghost, vars)

    return (ghost, L%f"({name} {' '.join(vars)})")

def forcesingle(L, G, root):
    counts = zeros((len(G), len(G)), int)
    counts[root, root] = 1

    for eclass in reversed(range(len(G))):
        counteclass = counts.sum(0)[eclass]

        for enode in G.values[eclass]:
            for tail in enode.tails:
                counts[eclass, tail] += counteclass

    counts = counts.sum(0)
    counts[counts == 0] = 1

    rules = []
    ghosts = set()
    maxghosts = Heap(3)
    for ghost in tqdm(chain(*ghostsingle(L, G, depth=3))):
        if length(ghost) <= 1: continue
        if ghost in ghosts: continue
        ghosts.add(ghost)

        c = 0
        for eclass, enodes in enumerate(G.values):
            if len(enodes) == 0: continue

            for enode in enodes:
                c += ghostpresence(G, ghost, enode) * counts[eclass]

        n, nargs = lent(ghost)
        saved = c * (n - nargs) - length(ghost)
        maxghosts.push(saved, ghost)

    for saved, ghost in maxghosts:
        print(prettyenode(L, G, ghost), saved)

        vars = []
        ghost = renameholes(ghost, vars)

        name = f'f{len(L.invented)}'
        L.add(Term(ghost, L[ghost.head].type, repr=name))
        rules.append((G.addexpr(ghost, True), L%f"({name} {' '.join(vars)})"))

    print(f'sieved through {len(ghosts)} ghosts, wouh')
    return rules

def optimize(L, rws, expr):
    G = EGraph(termlength)
    id = G.addexpr(L%expr)
    G.saturate(rws)
    cost, term = G.extract(id)
    return L<term

if __name__ == '__main__':
    L = Language([
        Term(0, 'N', repr='0'),
        Term(1, 'N', repr='1'),
        Term(lambda x, y: x + y, 'N', ['N', 'N'], repr='+'),
    ], type='N')

    G = EGraph()
    expr = G.addexpr(L%"(+ 1 (+ 0 1))", True)
    pttn = G.addexpr(L%"(+ B (+ A 1))", True)
    subst, c = G.match(pttn, expr)[0]
    assert subst["A"] == G.hashcons[G.addexpr(L%"0", True)]
    assert subst["B"] == G.hashcons[G.addexpr(L%"1", True)]
    assert c == G.hashcons[G.addexpr(expr, True)]

    G = EGraph()
    expr = G.addexpr(L%"(+ (+ (+ 1 1)) (+ 0 1))", True)
    pttn = G.addexpr(L%"(+ (+ (+ A A)) (+ 0 A))", True)
    subst, c = G.match(pttn, expr)[0]

    assert subst["A"] == G.hashcons[G.addexpr(L%"1", True)]
    assert c == G.hashcons[G.addexpr(expr, True)]

    # not consistent
    G = EGraph()
    expr = G.addexpr(L%"(+ (+ (+ 1 1)) (+ 0 1))", True)
    pttn = G.addexpr(L%"(+ (+ (+ A A)) (+ A 1))", True)
    assert not G.match(pttn, expr)

    # bigger matches
    G = EGraph()
    expr = G.addexpr(L%"(+ (+ (+ 1 1)) (+ 0 (+ 1 0)))", True)
    pttn = G.addexpr(L%"(+ A (+ 0 B))", True)
    subst, c = G.match(pttn, expr)[0]
    assert subst["A"] == G.hashcons[G.addexpr(L%"(+ (+ 1 1))", True)]
    assert subst["B"] == G.hashcons[G.addexpr(L%"(+ 1 0)", True)]
    assert c == G.hashcons[G.addexpr(expr, True)]

    # test fusing and conditional rewrites
    L = Language([
        Term([], '<A>', repr='end'),
        Term([], '<L>', repr='ø'),
        *[Term(n, '<N>', repr=str(n)) for n in range(12)],
        Term(cons, '<L>', ['<N>', '<L>'], repr='+'),
        Term(cons, '<A>', ['<L>', '<A>'], repr='a'),
    ], type='<A>')

    G = EGraph(termlength)

    a = L%"(+ 1 (+ 2 ø))"
    b = L%"(+ 3 (+ 4 ø))"

    eclass = fuse(L, G, (L.index('+'),), L.index('ø'), G.addexpr(a), G.addexpr(b))
    assert eclass == G.addexpr(L%"(+ 1 (+ 2 (+ 3 (+ 4 ø))))")

    fuseind = partial(fuse, L, G, (L.index('+'),), L.index('ø'))
    rules = [
        (G.addexpr(L%"(a A (a B R))"), (L%'(a I R)', {'I': (fuseind, '(A, B)')}))
    ]

    expr = L%"(a (+ 1 (+ 2 (+ 3 ø))) (a (+ 4 ø) end))"
    id = G.addexpr(expr)
    G.saturate(rules)

    c, term = G.extract(id)
    assert G.addexpr(expr) == G.addexpr(L%"(a (+ 1 (+ 2 (+ 3 (+ 4 ø)))) end)")

    #::: Extraction
    L = Language([
        Term('f', 'g', repr='f'),
        Term('g', 'g', repr='g')
    ], 0)

    G = EGraph(termlength)
    root = G.addexpr(L%"(f (g g))")
    G.saturate([(G.addexpr(L%"(g g)", True), L%"f")])
    assert G.extract(root)[1] == L%"(f f)"

    #::: Looping
    L = Language([
        Term(partial(add, 1), 'N', ['N'], repr='s'),
        Term(partial(add, -1), 'N', ['N'], repr='d'),
        *[Term(n, 'N', repr=str(n)) for n in range(21)],
        Term(cons, 'X', ['N', 'X'], repr='.'),
        Term(cons, 'X', ['N', 'X'], repr='~'),
        Term('loop', 'X', ['N', 'F', 'X'], repr='loop'),
        Term([], 'X', repr='ø'),
    ], 'X')

    def ctxfn(enode):
        # delegate context
        if enode.head == L.index('.') or enode.head == L.index('~'):
            return G.analyses[G.find(enode.tails[0])][2]

        if enode.head == L.index('ø'):
            return 0

        return L[enode.head].head

    G = EGraph(termlength, ctxfn)
    root = G.addexpr(L%"(. 1 (. 2 (. 3 (. 4 (. 5 ø)))))")
    root = G.addexpr(L%"(. 10 (. 5 (. 4 (. 3 (. 2 (. 1 ø))))))")

    def increaseloop(L, G, n: EClass):
        enode = G.analyses[G.find(n)][1]
        inc = L[enode.head].head + 1
        return G.addexpr(T(L.index(str(inc))))

    rules = [
        (G.addexpr(L%'(. 1 ø)', True), L%'(~ s ø)'),
        (G.addexpr(L%'(. N1 (. N2 P))', True), (L%'(~ s (. N2 P))', 'N1 == N2 + 1')),
        (G.addexpr(L%'(. N1 (. N2 P))', True), (L%'(~ d (. N2 P))', 'N1 == N2 - 1')),
        (G.addexpr(L%'(~ F R)', True), L%'(loop 1 F R)'),
        (G.addexpr(L%"(~ F (loop N F R))", True), (L%"(loop SN F R)",
            {'SN': (partial(increaseloop, L, G), '(N,)')})),
    ]

    G.saturate(rules)
    c, term = G.extract(root)
    assert term == L%"(. 10 (loop 5 s ø))"

    #::: Abstraction
    L = Language([
        Term('f', '.', repr='f'),
        Term('g', '.', repr='g'),
        Term('app', '.', repr='@')
    ], '.')

    def weightlength(G: EGraph, enode) -> int:
        return Q[enode.head] + sum(G.analyses[t][0] for t in enode.tails)

    G = EGraph(weightlength)
    Q = [1] * len(L)
    g = L%"(@ f (@ g g) (@ g f) (@ g f) (@ g g))"
    # g = L%"(@ f (@ g g) (@ g f) (@ g f) (@ (@ g g) g))"
    # g = L%"(@ f (@ g g) (@ g g))"
    rooteclass = G.addexpr(g)

    size, g = G.extract(rooteclass)

    subtrees = list(everysubtree(g))

    minghost = None
    mink = size + 1

    for ghost, count in countghosts(L, subtrees, subtrees).items():
        n, nargs = lent(ghost)

        k = size - count * (n - nargs - 1) + length(ghost)

        if k < mink:
            mink = k
            minghost = ghost

    ghost = minghost
    name = f'f{len(L.invented)}'
    gamma = Term(ghost, L[ghost.head].type, repr=name)
    L.add(gamma)
    Q.append(1 + nargs)

    rules = []
    vars = []
    ghost = renameholes(ghost, vars)
    rules.append((G.addexpr(ghost, True), L%f"({name} {' '.join(vars)})"))

    print(G.saturate(rules))
    print(L<G.extract(rooteclass)[1])

    force(L, G)
