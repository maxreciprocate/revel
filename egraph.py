from blob import *
from rich import print as pprint
from tqdm.rich import tqdm
from enumerate import App, Library

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

    def addexpr(self, expr: App | str, retenode=False) -> EClass:
        if not isinstance(expr, App):
            return expr

        tails = [self.addexpr(t) for t in expr.xs]
        enode = ENode(expr.fn, tuple(tails))

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
        for _ in tqdm(range(times)):
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

    def rewritesubst(self, subst: dict, term: App):
        if isinstance(term, str):
            return subst[term]

        if not term.xs:
            return self.addenode(ENode(term.fn, tuple()))

        newtails = list(term.xs)
        for ind, t in enumerate(term.xs):
            if t in subst:
                newtails[ind] = subst[t]
            else:
                newtails[ind] = self.rewritesubst(subst, t)

        return self.addenode(ENode(term.fn, tuple(newtails)))

    def extract(self, eclass, fn=App):
        cost, enode, _ = self.analyses[self.find(eclass)]
        tails = [self.extract(eclass, fn=fn)[1] for eclass in enode.tails]
        return cost, fn(enode.head, tuple(tails))

    def __len__(self):
        return len(self.parents)

    def __getitem__(self, ind):
        return self.values[self.find(ind)]

    def __repr__(self):
        pprint(self.hashcons)
        pprint(self.values)
        pprint(self.analyses)
        pprint(self.parents)
        return ''

def prettyenode(L: Library, G: EGraph, enode: ENode):
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

def termlength(G: EGraph, enode) -> int:
    return 1 + sum(G.analyses[t][0] for t in enode.tails)

def rewrites(L, *rws):
    return [(L% lhs, L% rhs) for lhs, rhs in [rw.split('~>', 1) for rw in rws]]

def fuse(L: Library, G: EGraph, fusewith: Tuple[int, int], fuseon: int, a: EClass, b: EClass) -> EClass:
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

def optimize(L, rws, expr):
    G = EGraph(termlength)
    id = G.addexpr(L%expr)
    G.saturate(rws)
    cost, term = G.extract(id)
    return L<term

if __name__ == '__main__':
    L = Library([
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

    def cons(x, xs):
        return [x] + xs

    # test fusing and conditional rewrites
    L = Library([
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
    L = Library([
        Term('f', 'g', repr='f'),
        Term('g', 'g', repr='g')
    ], 0)

    G = EGraph(termlength)
    root = G.addexpr(L%"(f (g g))")
    G.saturate([(G.addexpr(L%"(g g)", True), L%"f")])
    assert G.extract(root)[1] == L%"(f f)"

    #::: Looping
    L = Library([
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
    L = Library([
        Term('f', '.', repr='f'),
        Term('g', '.', repr='g'),
        Term('app', '.', repr='@')
    ], '.')

    def weightlength(G: EGraph, enode) -> int:
        return Q[enode.head] + sum(G.analyses[t][0] for t in enode.tails)

    G = EGraph(weightlength)
    Q = [1] * len(L)
    g = L%"(@ f (@ g g) (@ g f) (@ g f) (@ g g))"
    rooteclass = G.addexpr(g)
    size, g = G.extract(rooteclass)

