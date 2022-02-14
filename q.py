from importblob import *

# ■ ~

def binbase(base, n):
    out = []
    while n > 0:
        n, bit = divmod(n, base)
        out.append(str(bit))

    return ''.join(reversed(out))

lsoft = lambda ps: -F.log_softmax(ps.float(), -1) / np.log(2)
H = lambda ps: -(ps * log(ps) / np.log(2)).sum()

def isgeneric(tau):
    return ':' not in tau

def nless(G, Q, tau, budget, unwrap=False):
    qs, tinds = Q[tau].sort()

    c = 0
    for logp, gind in zip(qs, np.take(G.bytype[tau], tinds)):
        if logp > budget:
            break

        if not G[gind].tailtypes:
            c += 1
        else:
            # TODO of course it's not flatly bound
            c += np.prod([nless(G, Q, tailtau, budget - logp) for tailtau in G[gind].tailtypes])

    return c

def selectmask(base: int, a: int, mask: int) -> int:
    "almost like a & mask, but in a given base and also full shifted to 0b1111"
    base = max(2, base)
    out = 0

    steps = 0
    nbits = 0
    while mask > 0:
        mask, maskbit = divmod(mask, base)
        steps += 1

        if maskbit:
            bit = 0
            for _ in range(steps):
                a, bit = divmod(a, base)

            out += bit * base ** nbits

            if a == 0:
                break

            nbits += 1
            steps = 0

    return out

def growtail(G, Q, base, ind, tails, budget, counts, nbits, sequence, budgets):
    "allocates new bits for $ind if there is enough tails for the given budget"
    c = 0
    for logp, gind in tails[ind]:
        if logp > budget:
            continue

        if (ttaus := G[gind].tailtypes) is None:
            c += 1
        else:
            # iter.product on all tails
            c += np.prod([nless(G, Q, ttau, budget-logp) for ttau in ttaus])

    # need to allocate a new bit?
    while c - counts[ind] > base ** nbits[ind]:
        c -= base ** nbits[ind]
        counts[ind] += base ** nbits[ind]

        sequence.insert(0, ind)
        budgets.insert(0, budget)

        nbits[ind] += 1

def makeview(G, Q, tau: str):
    if isgeneric(tau) and not tau == G.type: return
    print(tau)

    qs, tinds = Q[tau].sort()
    fnumber = 0
    foffset = None

    atommapping = []
    opmapping = []
    fqs = []

    for q, gind in zip(qs, np.take(G.bytype[tau], tinds)):
        if G[gind].tailtypes:
            fqs.append(q)
            opmapping.append(gind)
        else:
            atommapping.append(gind)

    natoms, nops = len(atommapping), len(opmapping)

    fqs = array(fqs)
    masks = []

    for fgind in opmapping:
        tails = []

        for ttau in G[fgind].tailtypes:
            qs, inds = Q[ttau].sort()
            qs = qs.numpy()
            ginds = np.take(G.bytype[ttau], inds)

            tails.append(list(zip(qs, ginds)))

        counts = zeros(len(tails), np.uint64)
        nbits = zeros(len(tails), np.uint64)
        base = max(2, nops)

        sequence = []

        budgetstep = 0.05
        budget = 0
        budgets = []

        sequencelength = np.log(10**9) / np.log(base)
        while len(sequence) < sequencelength:
            budget += budgetstep

            for tind in range(len(tails)):
                growtail(G, Q, base, tind, tails, budget, counts, nbits, sequence, budgets)

        if nops > 1:
            if fnumber == 0:
                # TODO better bound on this q-difference between f
                fdiff = np.diff(array(fqs))[0]
                foffset = findfirst(lambda x: x > fdiff, budgets[::-1])

                if foffset is None:
                    foffset = len(sequence)

            if foffset == 0:
                sequence.append(np.inf)
                budgets.append(fdiff)
            else:
                sequence.insert(-foffset, np.inf)
                budgets.insert(-foffset, fdiff)

        numbers = zeros(len(tails), np.uint64)
        for ind, tail in enumerate(sequence[::-1]):
            if np.isfinite(tail):
                numbers[tail] += base ** ind
            elif fnumber == 0 and nops > 1:
                fnumber = base ** ind

        masks.append(list(numbers))
        print(sequence)

    return [masks, fnumber, nops, natoms, opmapping, atommapping]
