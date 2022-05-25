from egraph import *
from roll import *

def note(root, scale, index, next):
    return [(root + scales[scale][index]) % 12] + next

def multindex(i, nexti):
    return [i-1] + nexti

def totree(L, xs):
    x, *xs = xs
    tails = (totree(L, xs),) if xs else (T(L.index("nø")),)
    return T(L.index(f"+{roots[x]}"), tails)

flatsymbol = '♭'
sharpsymbol = '♯'
upsymbol = '↑'
downsymbol = '↓'

roots = ['C', 'D♭', 'D', 'E♭', 'E', 'F', 'G♭', 'G', 'A♭', 'A', 'B♭', 'B']
intervals = ['1', 'b2', '2', 'b3', '3', '4', 'b5', '5', 'b6', '6', 'b7', '7']
tointerval = {name: i for i, name in enumerate(intervals)}
scalesints = {
    'major': '1 2 3 4 5 6 7',
    'minor': '1 2 b3 4 5 b6 b7',
    'harm-minor': '1 2 b3 4 5 b6 7',
    'major-chord': '1 3 5 7',
    'minor-chord': '1 b3 5 b7',
    'mixolydian': '1 2 3 4 5 6 b7',
    'whole/half': '1 2 b3 4 b5 b6 6 7',
    'bebop-major': '1 2 3 4 5 b6 6 7',
    'locrian': '1 b2 b3 4 b5 b6 b7',
    'lydian': '1 2 3 b5 5 6 7',
    'pentatonic': '1 2 3 5 6',
    'chromatic': '1 b2 2 b3 3 4 b5 5 b6 6 b7 7',
}

scalesmap = {name: ap(lambda s: tointerval[s], ints.split())
             for name, ints in scalesints.items()}
scales = ap(array, scalesmap.values())

def ctxfn(G, enode: ENode):
    if enode.head == L.index('@'):
        v = G.analyses[enode.tails[1]][2]
        return v

    # do not update context
    if enode.head == L.index('+1') or enode.head == L.index('-1'):
        return None

    # return the length of the scale
    if enode.head in scaleinds:
        return len(scales[L[enode.head].head])

    if enode.head in indexinds:
        v = enode.head - min(indexinds) + 1
        return v

    return None

def incloop(G, n: EClass) -> EClass:
    nn = G.analyses[n][2] + 1
    return G.addexpr(L%repr(nn))

def levelterm(L: Language, t: T, end: int) -> List[T]:
    if len(t.tails) == 0:
        return [t]

    out = []

    while t.head != end:
        tails = list(t.tails)
        nt = tails[-1]
        tails[-1] = T(end)
        out.append(t._replace(tails=tuple(tails)))
        t = nt

    return out

def annotatescales(L, name, voice, term):
    annotations = [None] * len(voice)
    ind = 0
    for t in levelterm(L, term, L.index('nø')):
        key = (L<t.tails[0])[1:]
        scale = L<t.tails[1]

        _ind = ind
        ops = []
        for op in reversed(levelterm(L, t.tails[2], L.index('iø'))):
            match op:
                case T(head, (_, T(op), _)) if head == L.index('@'):
                     if op == L.index('+1'):
                         op = upsymbol
                     elif op == L.index('-1'):
                         op = downsymbol
                     else:
                         op = L[op].repr

                     ops.append(op)
                     ind += 1
                case T(head, (T(times), _, T(op), _)) if head == L.index('loop'):
                     ind += L[times].head
                     if op == L.index('+1'):
                         op = upsymbol
                     elif op == L.index('-1'):
                         op = downsymbol
                     else:
                         op = L[op].repr

                     ops.append(f'loop {L[times].repr} {op}')

        annotations[_ind] = (f'{key} {scale}', f'({", ".join(ops)})')

    from music21 import stream, note, duration, converter, metadata
    score = converter.parse(name, quantizePost=False)

    nns = ap(lambda s: s.capitalize(), name.rsplit('/')[-1][:-4].split('-'))
    if nns[-1].isdigit(): nns[-1] = '#'+nns[-1]
    score.metadata = metadata.Metadata(title=' '.join(nns), composer=f'')

    notes = score.flatten().getElementsByClass(note.Note)

    for annotation in annotations:
        n = next(notes)

        while not hasattr(n, 'isNote') or n.tie and n.tie.type == 'stop':
            n = next(notes)

        if annotation:
            n.addLyric('')
            for line in annotation:
                n.addLyric(line)

    score.show('musicxml')

L = Language([
    Term([], '?note', repr='nø'),
    Term([], '?d', repr='dø'),
    Term([], '?index', repr='iø'),
    Term(multindex, '?index', ['?N', '?index'], repr='@'),
    *[Term(n, '?N', repr=str(n)) for n in range(1, max(map(len, scales))+1)],
    *[Term(ind, '?scale', repr=scale) for ind, scale in enumerate(scalesmap.keys())],
    *[Term(n, '?root', repr=':' + roots[n]) for n in range(12)],
    Term(note, '?d', ['?root', '?scale', '?index', '?d'], repr='+'),
    *[Term(f"+{root}", '?note', ['?note'], repr=f"+{root}") for root in roots],
    Term('loop', '?index', ['?N', '?N'], repr='loop'),
    Term('+1', '?N', ['?N'], repr='+1'),
    Term('-1', '?N', ['?N'], repr='-1'),
], type='?d')

lastnum = [l for l in L if l.repr.isdigit()][-1].repr

@lru_cache(1<<20)
def weightenode(G: EGraph, enode: ENode) -> float:
    if L.index('+C') <= enode.head <= L.index('+B'):
        return np.inf

    if L.index('1') <= enode.head <= L.index(lastnum):
        return Q[enode.head - L.index('1')]

    if enode.head == L.index('iø'):
        return 1

    return 1 + sum(G.analyses[tail][0] for tail in enode.tails)

scaleinds = [ind for ind, l in enumerate(L) if l.type == '?scale']
scaleinds = range(min(scaleinds), max(scaleinds) + 1)

indexinds = [ind for ind, l in enumerate(L) if l.type == '?N']
indexinds = range(min(indexinds), max(indexinds) + 1)
Q = ones(12).cumsum().astype(int)

def bakein(midifpath):
    if midifpath.startswith('random'):
        return randint(12, size=20)

    roll = bakeroll(midifpath, quantize=32, staccato=True)
    roll = roll[::-1, :]
    midivoice = roll.argmax(0)
    voice = midivoice[np.nonzero(midivoice)]
    voice = (voice + 1) % 12
    return voice

def stagerules(L: Language, G: EGraph):
    fuseind = partial(fuse, L, G, (L.index('@'), L.index('loop')), L.index('iø'))

    rules = []
    for ri, root in enumerate(roots):
        for scale, tones in scalesmap.items():
            for i, n in enumerate(tones):
                rules.append(f"(+{roots[(n+ri) % 12]} ...) ~> (+ :{roots[ri]} {scale} (@ {scale} {i+1} iø) ...)")

    rules = rewrites(L, *rules)
    global incloop
    lincloop = partial(incloop, G)

    langrws = [
        (G.addexpr(L%"(+ R S I1 (+ R S I2 ...))", True), (L%"(+ R S I ...)", {'I': (fuseind, '(I2, I1)')})),
        (G.addexpr(L%"(@ S I2 (@ S I1 ...))", True), (L%"(@ S +1 (@ S I1 ...))", "I2 == I1 % S + 1")),
        (G.addexpr(L%"(@ S I2 (@ S I1 ...))", True), (L%"(@ S -1 (@ S I1 ...))", "I1 == I2 % S + 1")),
        (G.addexpr(L%"(@ S F (@ S F ...))", True), L%"(loop 2 S F ...)"),
        (G.addexpr(L%"(@ S F (loop N S F ...))", True), (L%"(loop NN S F ...)", {'NN': (lincloop, '(N,)')})),
        (G.addexpr(L%"(+ R S (loop N S F (@ S 1 iø)) ...)", True), (L%"(+ R S (loop NN S F iø) ...)", {'NN': (lincloop, '(N,)')})),
    ]

    rules.extend(langrws)
    return rules

def rendervoice(render, namevoice):
    G = EGraph(weightenode, ctxfn)
    rooteclass = G.addexpr(totree(L, namevoice[1]))

    G.saturate(stagerules(L, G))
    opt = G.extract(rooteclass)[1]
    for t in levelterm(L, opt, L.index('nø')):
        print(grepr(L, t))

    if render:
        annotatescales(L, *namevoice, opt)
    return opt

def getkalon(L, history, source):
    G = EGraph(weightenode, ctxfn)
    return kkalon(L, G, totree(L, source), totree(L, history), stagerules(L, G), depth=3, verb=True)

if __name__ == '__main__':
    sources = [
        'tunes/the-lick.mid',
        'tunes/coltrane.mid',
        'tunes/charlie-parker-1.mid',
        'tunes/charlie-parker-2.mid',
        'tunes/bill-evans-1.mid',
        'tunes/bill-evans-2.mid',
        'tunes/barry-harris.mid',
        'tunes/al-haig.mid',
    ]

    ss = ap(list, ap(bakein, sources))

    H = flatten(ss)
    out = ap(P(rendervoice, True), zip(sources, ss))

    ks = ap(P(getkalon, L, H), ss)

    for (k, ml), source in zip(ks, sources):
        print(f'{log(k)} {ml} {source}')

    kalons = list(zip(ks, sources))
    pickle.dump(kalons, open('photos/kalons.pkl','wb'))

    ks = ap(len, ss)

    lenghts = list(zip(ks, sources))
    pickle.dump(lengths, open('photos/lengths.pkl','wb'))
