from importblob import *
from referee import *
from rich import print

class DisjointSet:
    def __init__(self, cost=lambda disjointset, enode: 0):
        self.parents = []
        self.values = []

        self.cost = cost
        self.analyses = []

    def __len__(self):
        return len(self.parents)

    def add(self, value=None):
        self.parents.append(-1)
        self.values.append({value})
        self.analyses.append((self.cost(value), value))
        return len(self) - 1

    def find(self, ind):
        while self.parents[ind] >= 0:
            ind = self.parents[ind]

        return ind

    def merge(self, a, b):
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

        if self.analyses[broot][0] > self.analyses[aroot][0]:
            self.analyses[broot] = self.analyses[aroot]

        return broot

class Enode(NamedTuple):
    head: int
    tails: tuple = tuple()

Eclass = int

class Egraph:
    def __init__(self, cost=lambda disjointset, enode: 0):
        self.eclass = DisjointSet(partial(cost, self))
        self.hashcons = {}

    def __repr__(self):
        print(self.hashcons)
        print(self.eclass.values)
        print(self.eclass.analyses)
        print(self.eclass.parents)
        return ''

def canonize(G, f: Enode) -> Enode:
    # cannot cannonize anything with holes
    if any(isinstance(tail, str) for tail in f.tails):
        return None

    return Enode(f.head, tuple(G.eclass.find(a) for a in f.tails))

def merge(G, a: Eclass, b: Eclass):
    id = G.eclass.merge(a, b)
    eclass = []
    for enode in G.eclass.values[id]:
        G.hashcons.pop(enode)
        enode = canonize(G, enode)
        G.hashcons[enode] = id
        eclass.append(enode)

    G.eclass.values[id] = set(eclass)

def addenode(G, f: Enode) -> Eclass:
    f = canonize(G, f)
    if f in G.hashcons:
        return G.hashcons[f]

    id = G.eclass.add(f)
    G.hashcons[f] = id
    return id

def addexpr(G, f: T, retenode=False) -> Eclass:
    if not isinstance(f, T):
        return None

    tails = [addexpr(G, t) for t in f.tails]
    if any(t is None for t in tails):
        return None

    enode = Enode(f.head, tuple(tails))
    eclass = addenode(G, enode)

    if retenode:
        return enode
    return eclass

def addarray(G, xs) -> Eclass:
    x, *xs = xs
    tails = (addarray(G, xs),) if xs else tuple()
    return addenode(G, Enode(str(x), tails))

def congruences(G):
    out = []

    # find e-nodes that changed
    for id, eclass in enumerate(G.eclass.values):
        if not eclass: continue

        for enode in eclass:
            cenode = canonize(G, enode)
            if cenode in G.hashcons:
                newid = G.hashcons[cenode]

                if newid != id:
                    out.append((id, newid))

    return out

def rebuild(G):
    while (cs := congruences(G)):
        for a, b in cs:
            merge(G, a, b)

# there is some room to rewrite this, but i hate wrapped generators
def match(G, pattern: T, enode: Enode):
    if isinstance(pattern, str):
        return {pattern: G.hashcons[enode]}, G.hashcons[enode]

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
            else:
                # pt has no holes
                pteclass = addexpr(G, pt)
                # if not the same eclass
                if pteclass is not None:
                    if taileclass == pteclass:
                        possibletails[ind].append((None, pteclass))
                        continue
                    else:
                        return None

                # pt has holes
                # there are a lot of enodes in this eclass
                for tailenode in G.eclass.values[taileclass]:
                    if (sigmacs := match(G, pt, tailenode)):
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
            c = addenode(G, Enode(pattern.head, tuple(newtails)))
            out.append((newsubst, c))

        return out

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

def termlength(G: Egraph, f: Enode) -> int:
    l = 1
    for t in f.tails:
        l += G.eclass.analyses[t][0]

    return l

def extract(G, eclass):
    cost, enode = G.eclass.analyses[eclass]
    tails = [extract(G, eclass)[1] for eclass in enode.tails]
    return cost, T(enode.head, tails=tuple(tails))

def rewrites(L, *rws):
    return [(L(lhs), L(rhs)) for lhs, rhs in [rw.split('~>', 1) for rw in rws]]

def rewritesubst(G, subst, term):
    if isinstance(term, str):
        return subst[term]

    if not term.tails:
        return addenode(G, Enode(term.head))

    newtails = list(term.tails)
    for ind, t in enumerate(term.tails):
        if t in subst:
            newtails[ind] = subst[t]
        else:
            newtails[ind] = rewritesubst(G, subst, t)

    return addenode(G, Enode(term.head, tuple(newtails)))

def applyrws(G, rws, times=10):
    for _ in range(times):
        saturated = True

        matches = []
        for lhs, rhs in rws:
            for enode in chain(*G.eclass.values):
                if (matchlist := match(G, lhs, enode)):
                    for m in matchlist:
                        matches.append((rhs, *m))

        print(f'{len(matches)=}')

        for rhs, subst, lhseclass in matches:
            if isinstance(rhs[1], dict):
                rhs, bonus = rhs

                for var, expr in bonus.items():
                    for k, v in subst.items():
                        expr = expr.replace(k, str(v))

                    if (expr := eval(expr)):
                        subst.update({var: expr})

            rhseclass = rewritesubst(G, subst, rhs)

            if lhseclass != rhseclass:
                saturated = False

            merge(G, lhseclass, rhseclass)

        rebuild(G)

        if saturated:
            break

def cons(x, xs):
    return [x] + xs

def fuse(TL: Language, G: Egraph, fusewith: int, fuseon: int, a: Eclass, b: Eclass) -> Eclass:
    enodes = []

    c, enode = G.eclass.analyses[G.eclass.find(a)]
    if enode.head != fusewith:
        return None

    while enode.head != fuseon:
        enodes.append(enode)
        c, enode = G.eclass.analyses[G.eclass.find(enode.tails[-1])]

    extratail = b
    for enode in reversed(enodes):
        newenode = Enode(enode.head, (*enode.tails[:-1], extratail))
        extratail = addenode(G, newenode)

    return extratail

def optimize(L, rws, expr):
    G = Egraph(termlength)
    expr = L(expr)
    addexpr(G, expr)
    applyrws(G, rws)
    cost, term = extract(G, addexpr(G, expr))
    return grepr(L, term)

if __name__ == '__main__':
    L = Language([
        Term(0, 'N', repr='0'),
        Term(1, 'N', repr='1'),
        Term(lambda x, y: x + y, 'N', ['N', 'N'], repr='+'),
    ], type='N')

    G = Egraph()
    expr = L("(+ 1 (+ 0 1))")
    subst, c = match(G, L("(+ ?b (+ ?a 1))"), addexpr(G, expr, True))[0]
    assert subst["?a"] == G.hashcons[addexpr(G, L("0"), True)]
    assert subst["?b"] == G.hashcons[addexpr(G, L("1"), True)]
    assert c == G.hashcons[addexpr(G, expr, True)]

    G = Egraph()
    expr = P(L, "(+ (+ (+ 1 1)) (+ 0 1))")
    subst, c = match(G, L("(+ (+ (+ ?a ?a)) (+ 0 ?a))"), addexpr(G, expr, True))[0]
    assert subst["?a"] == G.hashcons[addexpr(G, L("1"), True)]
    assert c == G.hashcons[addexpr(G, expr, True)]

    # not consistent
    G = Egraph()
    expr = L("(+ (+ (+ 1 1)) (+ 0 1))")
    assert not match(G, L("(+ (+ (+ ?a ?a)) (+ ?a 1))"), addexpr(G, expr, True))

    # bigger matches
    G = Egraph()
    expr = L("(+ (+ (+ 1 1)) (+ 0 (+ 1 0)))")
    subst, c = match(G, L("(+ ?a (+ 0 ?b))"), addexpr(G, expr, True))[0]
    assert subst["?a"] == G.hashcons[addexpr(G, L("(+ (+ 1 1))"), True)]
    assert subst["?b"] == G.hashcons[addexpr(G, L("(+ 1 0)"), True)]
    assert c == G.hashcons[addexpr(G, expr, True)]

    # test fusing and conditional rewrites
    G = Egraph(termlength)
    TL = Language([
        Term([], '<A>', repr='end'),
        Term([], '<L>', repr='ø'),
        *[Term(n, '<N>', repr=str(n)) for n in range(12)],
        Term(cons, '<L>', ['<N>', '<L>'], repr='+'),
        Term(cons, '<A>', ['<L>', '<A>'], repr='a'),
    ], type='<A>')

    # test fuse
    a = TL("(+ 1 (+ 2 ø))")
    b = TL("(+ 3 (+ 4 ø))")

    eclass = fuse(TL, G, TL.index('+'), TL.index('ø'), addexpr(G, a), addexpr(G, b))
    assert eclass == addexpr(G, TL("(+ 1 (+ 2 (+ 3 (+ 4 ø))))"))

    fuseind = partial(fuse, TL, G, TL.index('+'), TL.index('ø'))

    rules = [
        (TL("(a ?1 (a ?2 ?&))"), (TL('(a ?ind ?&)'), {'?ind': 'fuseind(?1, ?2)'}))
    ]

    expr = TL("(a (+ 1 (+ 2 (+ 3 ø))) (a (+ 4 ø) end))")
    addexpr(G, expr)
    applyrws(G, rules)

    c, term = extract(G, addexpr(G, expr))
    assert addexpr(G, expr) == addexpr(G, TL("(a (+ 1 (+ 2 (+ 3 (+ 4 ø)))) end)"))
