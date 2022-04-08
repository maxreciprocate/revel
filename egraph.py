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

class EGraph:
    def __init__(self, cost=lambda egraph, enode: 0, ctxfn=lambda enode: None):
        self.parents = []
        self.values = []

        self.cost = cost
        self.ctxfn = ctxfn
        self.analyses = []
        self.hashcons = {}

    def add(self, value=None):
        self.parents.append(-1)
        self.values.append({value})
        self.analyses.append([self.cost(self, value), value, self.ctxfn(value)])
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

        if self.analyses[broot][0] > self.analyses[aroot][0]:
            self.analyses[broot] = self.analyses[aroot]
            # keep the ctx
            self.analyses[broot][2] = self.analyses[aroot][2]

        if isinstance(self.analyses[aroot][2], int):
            self.analyses[broot][2] = self.analyses[aroot][2]

        return broot

    def canonize(self, f: ENode) -> ENode:
        # cannot cannonize anything with holes
        if any(isinstance(tail, str) for tail in f.tails):
            return None

        return ENode(f.head, tuple(self.find(a) for a in f.tails))

    def merge(self, a: EClass, b: EClass):
        id = self.union(a, b)
        eclass = set()

        for enode in self.values[id]:
            self.hashcons.pop(enode)
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

    def addexpr(self, expr: T, retenode=False) -> EClass:
        if not isinstance(expr, T):
            return None

        tails = [self.addexpr(t) for t in expr.tails]
        if any(t is None for t in tails):
            return None

        enode = ENode(expr.head, tuple(tails))
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

    # there is some room to rewrite this, but i hate wrapped generators
    def match(self, pattern: T, enode: ENode):
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
                else:
                    # pt has no holes
                    pteclass = self.addexpr(pt)

                    if pteclass is not None:
                        # if not the same eclass
                        if G.find(taileclass) == pteclass:
                            possibletails[ind].append((None, pteclass))
                            continue
                        else:
                            return None

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

            print(f'{len(matches)=}')
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
                            state = self.analyses[eclass][-1]
                            condition = condition.replace(arg, str(state))

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
                break

    def rewritesubst(self, subst, term):
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

    def extract(self, eclass):
        cost, enode, _ = self.analyses[self.find(eclass)]
        tails = [self.extract(eclass)[1] for eclass in enode.tails]
        return cost, T(enode.head, tails=tuple(tails))

    def __len__(self):
        return len(self.parents)

    def __repr__(self):
        print(self.hashcons)
        print(self.values)
        print(self.analyses)
        print(self.parents)
        return ''


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

def fuse(L: Language, G: EGraph, fusewith: int, fuseon: int, a: EClass, b: EClass) -> EClass:
    enodes = []

    c, enode, _ = G.analyses[G.find(a)]
    if enode.head != fusewith:
        return None

    while enode.head != fuseon:
        enodes.append(enode)
        c, enode, _ = G.analyses[G.find(enode.tails[-1])]

    extratail = b
    for enode in reversed(enodes):
        newenode = ENode(enode.head, (*enode.tails[:-1], extratail))
        extratail = G.addenode(newenode)

    return extratail

def optimize(L, rws, expr):
    G = EGraph(termlength)
    id = G.addexpr(L%expr)
    G.saturate(rws)
    cost, term = G.extract(id)
    return L<term
# ■ ~

if __name__ == '__main__':
    L = Language([
        Term(0, 'N', repr='0'),
        Term(1, 'N', repr='1'),
        Term(lambda x, y: x + y, 'N', ['N', 'N'], repr='+'),
    ], type='N')

    G = EGraph()
    expr = L%"(+ 1 (+ 0 1))"
    subst, c = G.match(L%"(+ B (+ A 1))", G.addexpr(expr, True))[0]

    assert subst["A"] == G.hashcons[G.addexpr(L%"0", True)]
    assert subst["B"] == G.hashcons[G.addexpr(L%"1", True)]
    assert c == G.hashcons[G.addexpr(expr, True)]

    G = EGraph()
    expr = L%"(+ (+ (+ 1 1)) (+ 0 1))"
    subst, c = G.match(L%"(+ (+ (+ A A)) (+ 0 A))", G.addexpr(expr, True))[0]

    assert subst["A"] == G.hashcons[G.addexpr(L%"1", True)]
    assert c == G.hashcons[G.addexpr(expr, True)]

    # not consistent
    G = EGraph()
    expr = L%"(+ (+ (+ 1 1)) (+ 0 1))"
    assert not G.match(L%"(+ (+ (+ A A)) (+ A 1))", G.addexpr(expr, True))

    # bigger matches
    G = EGraph()
    expr = L%"(+ (+ (+ 1 1)) (+ 0 (+ 1 0)))"
    subst, c = G.match(L%"(+ A (+ 0 B))", G.addexpr(expr, True))[0]
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

    eclass = fuse(L, G, L.index('+'), L.index('ø'), G.addexpr(a), G.addexpr(b))
    assert eclass == G.addexpr(L%"(+ 1 (+ 2 (+ 3 (+ 4 ø))))")

    fuseind = partial(fuse, L, G, L.index('+'), L.index('ø'))
    rules = [
        (L%"(a A (a B R))", (L%'(a I R)', {'I': (fuseind, '(A, B)')}))
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
    G.saturate([(L%"(g g)", L%"f")])
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
        (L%'(. 1 ø)', L%'(~ s ø)'),
        (L%'(. N1 (. N2 P))', (L%'(~ s (. N2 P))', 'N1 == N2 + 1')),
        (L%'(. N1 (. N2 P))', (L%'(~ d (. N2 P))', 'N1 == N2 - 1')),
        (L%'(~ F R)', L%'(loop 1 F R)'),
        (L%"(~ F (loop N F R))", (L%"(loop SN F R)",
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
    g = L%"(@ f (@ f (@ g g) (@ g g)))"
    rooteclass = G.addexpr(g)

    G.addexpr(L%"(@ g g)")

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

    # need to use forceholes with debrujin instead of dicts and substs
    def renameholes(t, newnames):
        if not t.tails:
            return t

        newtails = [None] * len(t.tails)
        for tind, tail in enumerate(t.tails):
            if isinstance(tail, str):
                var = f'V{len(newnames)}'
                newtails[tind] = var
                newnames.append(var)
            else:
                newtails[tind] = renameholes(tail, newnames)

        return t._replace(tails=tuple(newtails))

    rules = []
    vars = []
    ghost = renameholes(ghost, vars)
    rules.append((ghost, L%f"({name} {' '.join(vars)})"))

    G.saturate(rules)
    print(L<G.extract(rooteclass)[1])
