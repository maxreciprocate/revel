from lang import *
from egraph import *
from roll import *
from decon import *

# ■ ~

L = Language([
    Term([], '<L>', repr='ø'),
    Term(0, '<N>', repr='0'),
    Term('S', '<N>', ['<N>'], repr='S'),
    Term('cons', '<L>', ['<N>', '<L>'], repr='@'),
    Term('loop', '<L>', ['<N>', '<L>', '<L>'], repr='loop'),
    Term('b-12', '<L>', ['<L>'], repr='b-12'),
    Term('e-3', '<L>', [], repr='e-3'),
    Term('+1', '<L>', ['<N>'], repr='+1'),
], '<L>')
L.solder()

numbs = [l for l in L if l.repr.isdigit()]
qstart = L.index(numbs[0].repr)
qend = L.index(numbs[-1].repr)
Q = ones(len(numbs)).cumsum().astype(int)

def ctxfn(G: EGraph, enode: ENode):
    if enode.head == L.index('@'):
        v = G.analyses[enode.tails[0]][2]
        return v

    # do not update context
    if enode.head == L.index('+1'):
        return None
    elif enode.head in range(qstart, qend+1):
        return 0 # Q[enode.head-qstart]
    elif enode.head == L.index('S'):
        return G.analyses[enode.tails[0]][2] + 1

    return None

def incloop(n: EClass) -> EClass:
    nn = G.analyses[n][2] + 1
    return G.addexpr(L%repr(nn))

def toline(L, xs):
    x, *xs = xs
    n = growtree(L, '<N>', x)
    tails = (n, toline(L, xs)) if xs else (n, T(L.index('ø')))
    return T(L.index('@'), tails)

def getghosts(L, G, rules, xs):
    rooteclass = G.addexpr(toline(L, xs))
    G.saturate(rules)
    ghosts = set(filter(lambda g: length(g) > 1, chain(*ghostsingle(L, G, depth=10))))
    print(L<G.extract(rooteclass)[1])
    return G, ghosts


@lru_cache(maxsize=1<<20)
def weightenode(G: EGraph, enode: ENode) -> float:
    if qstart <= enode.head <= qend:
        return Q[enode.head - qstart]

    return 1 + sum(G.analyses[tail][0] for tail in enode.tails)

G = EGraph(weightenode, ctxfn)
sroot = G.addexpr(toline(L, [1, 2, 3]))
G.saturate(rules)
sroot = G.find(sroot)
ghosts = set(filter(lambda g: length(g) > 1, chain(*ghostbust(L, G, depth=10))))
ghosts = [(1, ghost) for ghost in ghosts]

xs = list(set([samplepath(G, random.choice(list(G[sroot]))) for _ in range(1000)]))
xs.append(G.extract(sroot, ENode)[1])

ghostheap = kkstack(G, xs, ghosts)
kalon = 0

ppghosts = []
goodness = []

for ind, (k, ghost) in enumerate(sorted(ghostheap, reverse=True)):
    kalon += k
    print(prettyenode(L, G, ghost), log(k))
    ppghosts.append(prettyenode(L, G, ghost))
    goodness.append(log(k))

# ■ ~

xs = [1, 2, 3]

topn = 4
hsheap = Heap(500)

hsheap
for ls, _ in tqdm(chain(*[np.ndenumerate(empty([topn] * i)) for i in range(3, 7)])):
    history = list(ls) + [2, 3]
    G = EGraph(weightenode, ctxfn)

    rules = [
        (G.addexpr(L%"(@ I2 (@ I1 ...))", True), (L%"(@ I2 (@ +1 ...))", "I1 == I2 + 1")),
        (G.addexpr(L%"(@ F (@ F ...))", True), L%"(loop  F ...)"),
        (G.addexpr(L%"(@ (S 0) (@ (S (S 0)) ...))", True), L%"(b-12 ...)"),
        (G.addexpr(L%"(@ (S (S (S 0))) ø)", True), L%"e-3"),
    ]

    k = kkalon(L, G, toline(L, xs), toline(L, history), rules, depth=10)
    hsheap.push(k, history)

for k, history in sorted(hsheap, reverse=True):
    print(history, log(k))
    G = EGraph(weightenode, ctxfn)
    k = kkalon(L, G, toline(L, xs), toline(L, history), rules, depth=10, verb=True)

L<G.extract(G.addexpr(toline(L, history)))[1]
histories = [(k, d) for k, d in sorted(hsheap)]
