from blob import *

H = lambda ps: -(ps * log(ps) / np.log(2)).sum()

def binbase(base, n, outstr=True):
    out = []
    while n > 0:
        n, bit = divmod(n, base)
        out.append(str(bit) if outstr else bit)

    if outstr:
        return ''.join(reversed(out))

    return out[::-1]

def foldbase(base, bits):
    out = 0
    for ind, bit in enumerate(reversed(bits)):
        out += bit * base ** ind
    return int(out)

def isgeneric(tau):
    return ':' not in tau

def nless(L, Q, tau, budget, unwrap=False):
    qs, tinds = np.sort(Q[tau]), np.argsort(Q[tau])

    c = 0
    for logp, gind in zip(qs, L.bytype[tau][tinds]):
        if logp > budget:
            break

        if not L[gind].tailtypes:
            c += 1
        else:
            # TODO of course it's not flatly bound
            c += np.prod([nless(L, Q, tailtau, budget - logp) for tailtau in L[gind].tailtypes])

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

def growtail(L, Q, base, ind, tails, budget, counts, nbits, sequence, budgets):
    "allocates new bits for $ind if there is enough tails for the given budget"
    c = 0
    for logp, gind in tails[ind]:
        if logp > budget:
            continue

        if (ttaus := L[gind].tailtypes) is None:
            c += 1
        else:
            # iter.product on all tails
            c += np.prod([nless(L, Q, ttau, budget-logp) for ttau in ttaus])

    # need to allocate a new bit?
    while c - counts[ind] > base ** nbits[ind]:
        c -= base ** nbits[ind]
        counts[ind] += base ** nbits[ind]

        sequence.append(ind)
        budgets.append(budget)

        nbits[ind] += 1

def multiview(L, Q, taus):
    "the length i have to go through ^^"
    return {tau: makeview(L, Q, tau) for tau in taus}

def makeview(L, Q, tau: str):
    qs, tinds = np.sort(Q[tau]), np.argsort(Q[tau])

    atommapping = []
    opmapping = []
    fqs = []

    for q, gind in zip(qs, np.atleast_1d(L.bytype[tau][tinds])):
        if L[gind].tailtypes:
            fqs.append(q)
            opmapping.append(gind)
        else:
            atommapping.append(gind)

    ind_fnull = None
    # f_ø just to have the last diff?
    # why the last diff after all? (doesn't it matter?)
    if len(fqs) & 1:
        fqs.append(fqs[-1])
        ind_fnull = len(fqs)

    natoms, nops = len(atommapping), len(opmapping)

    fqs = array(fqs)
    # keep the offsets as in the first f
    foffsets = []
    leapfrom = np.inf
    leapnumber = 0

    masks = []
    for fgind in opmapping:
        fnumber = 0
        foffset = None

        nops_satiated = 0

        tails = []

        for ttau in L[fgind].tailtypes:
            qs, inds = np.sort(Q[ttau]), np.argsort(Q[ttau])
            ginds = L.bytype[ttau][inds]

            tails.append(list(zip(qs, ginds)))

        counts = zeros(len(tails), np.uint64)
        nbits = zeros(len(tails), np.uint64)
        base = 2

        sequence = []

        budgetstep = 3
        budget = 0
        budgets = []

        sequencelength = np.log(10**6) / np.log(base)
        while len(sequence) < sequencelength:
            budget += budgetstep

            for tind in range(len(tails)):
                growtail(L, Q, base, tind, tails, budget, counts, nbits, sequence, budgets)

        if len(foffsets) > 0:
            for foffset in foffsets:
                sequence.insert(foffset, np.inf)
        else:
            fdiff = 0
            while nops > 1 and nops - nops_satiated > 0:
                fdiff = np.diff(fqs[nops_satiated//2:])[0] + fdiff
                foffset = findfirst(lambda x: x > fdiff, budgets)

                if foffset is None:
                    foffset = len(sequence)

                sequence.insert(foffset, np.inf)
                budgets.insert(foffset, fdiff)

                if nops_satiated == 0:
                    nops_satiated = 2
                else:
                    nops_satiated <<= 1

                foffsets.append(foffset)

        numbers = zeros(len(tails), np.uint64)
        for ind, tail in enumerate(sequence):
            if np.isfinite(tail):
                numbers[tail] += base ** ind
            else:
                fnumber += base ** ind

        if nops > 2 and not np.log2(nops).is_integer():
            bits = binbase(2, nops, outstr=False)

            sequence = array(sequence)
            sequence[np.where(np.isfinite(sequence))] = 0
            sequence[np.where(~np.isfinite(sequence))] = bits[::-1]

            leapfrom = foldbase(2, sequence[::-1])
            leapto = int(2 ** np.ceil(np.log2(leapfrom)))
            leapnumber = leapto - leapfrom

        masks.append(list(numbers))

    return [masks, fnumber, leapfrom, leapnumber, nops, natoms, opmapping, atommapping]
