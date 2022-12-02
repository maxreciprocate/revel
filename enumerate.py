import math
import numpy as np
from numba import njit
from typing import NamedTuple
from functools import lru_cache
from itertools import zip_longest
from collections import defaultdict
from copy import copy
from λ import App, Var, Ref, Lam, reduce, parse, decode, encode
from time import time

H = lambda ps: -(ps * np.log2(ps)).sum()

@njit
def collapse(x: int) -> int:
    out = 0
    nth = 0
    while x != 0:
        if x & 1:
            out += 2 ** nth
            nth += 1
        x >>= 1
    return out

@njit
def select_mask(x: int, mask: int) -> int:
    "x & mask, but only on bits which are 1 in the mask"
    out = 0
    nth = 0
    while x != 0:
        if mask & 1:
            if x & 1:
                out += 2 ** nth
            nth += 1
        mask >>= 1
        x >>= 1
    return out

@njit
def inverse_mask(x: int, mask: int):
    "inverse select_mask, so that select_mask(inverse_mask(x, mask), mask) == x"
    if x >= mask:
        return x

    out = 0
    nth = 0
    while x != 0:
        if mask & 1:
            if x & 1:
                out += 2 ** nth
            x >>= 1
        mask >>= 1
        nth += 1
    return out

class Term(NamedTuple):
    fn: any
    type: str
    tailtypes: list = []
    forbidden: list = []
    repr: str = '<>'
    def __repr__(self): return self.repr

class Type(NamedTuple):
    type: str
    parent: str = None
    arg_ix: int = None
    def __repr__(self):
        if self.parent: return f"<{self.type}:{self.parent}:{self.arg_ix}>"
        return f"<{self.type}>"

class TypeMask(NamedTuple):
    natoms: int
    nfuncs: int
    atoms_ixs: list = []
    funcs_ixs: list = []
    func_mask: list = []
    masks: list = []

class Library:
    "An array of abstrations"
    def __init__(self, primitives, type: Type):
        self.primitives = primitives
        self.invented = []
        self.type = type
        self.prepare()

    def add(self, term):
        self.invented.append(term)
        self.terms = self.primitives + self.invented
        self.prepare()

    def index(self, repr: str):
        for ix, term in enumerate(self.terms):
            if term.repr == repr:
                return ix

    def reset(self):
        self.invented = []
        self.prepare()

    def prepare(self):
        self.terms = self.primitives + self.invented
        # mapping type -> [term_ix]
        self.type_to_ixs = defaultdict(list)

        # fill the type mapping
        for ix, term in enumerate(self.terms):
            self.type_to_ixs[term.type].append(ix)

        # update argument types for each term
        # by specifying its arity index and removing forbidden terms from the mapping
        for term_ix, term in enumerate(self.terms):
            updated_tailtypes = []

            for tail_ix, (tailtype, forbidden) in enumerate(zip_longest(term.tailtypes, term.forbidden)):
                forbidden = forbidden or []
                # filter forbidden funcs
                ixs = [ix for ix in self.type_to_ixs[tailtype] if not self.terms[ix].repr in forbidden]
                unique_type = Type(tailtype.type, term.repr, tail_ix)

                self.type_to_ixs[unique_type] = ixs
                updated_tailtypes.append(unique_type)

            self.terms[term_ix] = term._replace(tailtypes=updated_tailtypes)

        self.funcs = [term.fn for term in self.terms]
        self.type_to_ixs = {type: np.array(ixs) for type, ixs in self.type_to_ixs.items()}
        self.masks = create_masks(self)

    def __getitem__(self, ix):
        return self.funcs[ix]

    def __len__(self): return len(self.terms)
    def __repr__(self): return repr(self.terms)
    def __call__(self, s: str): return reduce(encode(parse(s), self), self)
    def __lshift__(self, s: str): return encode(parse(s), self)
    def __hash__(self): return sum(hash(term[0]) for term in self.invented)

def create_mask(bitstring, type, typemask, func=False) -> int:
    mask = 0
    for ix, t in enumerate(bitstring):
        if t == type:
            mask += 2 ** ix

    if typemask.nfuncs == 0:
        last = typemask.natoms
    elif func:
        last = typemask.nfuncs
    else:
        last = mask + 1

    # `last` is the first overflow, on which a jump by `leap = mask - last` is required"
    last = inverse_mask(last, mask)
    if mask == 0:
        leap = 0
    else:
        leap = 2**math.floor(math.log2(mask)+1) - last

    return mask, last, leap

def create_masks(L):
    type_masks = {}
    # segragate atoms/funcs for each type
    for type, ixs in L.type_to_ixs.items():
        atoms_ixs = [ix for ix in ixs if len(L.terms[ix].tailtypes) == 0]
        funcs_ixs = [ix for ix in ixs if len(L.terms[ix].tailtypes) >= 1]
        natoms = len(atoms_ixs)
        nfuncs = len(funcs_ixs)
        type_masks[type] = TypeMask(natoms, nfuncs, atoms_ixs, funcs_ixs)

    # create masks for bitstrings
    for type, typemask in type_masks.items():
        if typemask.nfuncs == 0:
            continue

        bitstring = [0] * math.ceil(np.log2(typemask.nfuncs))
        func_mask = create_mask(bitstring, 0, typemask, func=True)

        max_bit = 32
        masks, leaps = [], []
        for fix in typemask.funcs_ixs:
            tailtypes = L.terms[fix].tailtypes
            func_bitstring = copy(bitstring)

            while len(func_bitstring) < max_bit:
                for tailtype in tailtypes:
                    tail_typemask = type_masks[tailtype]

                    if tail_typemask.nfuncs == 0:
                        nbits_allocated = sum(t == tailtype for t in func_bitstring)
                        if 2 ** nbits_allocated > tail_typemask.natoms:
                            # give room to other types if there are any
                            if len(tailtypes) > 1:
                                continue

                    func_bitstring.append(tailtype)

            fix_masks = [create_mask(func_bitstring, t, type_masks[t]) for t in tailtypes]
            masks.append(fix_masks)

        type_masks[type] = typemask._replace(masks=masks, func_mask=func_mask)

    return type_masks

@lru_cache(maxsize=1<<30)
def growtree(L, type, n: int):
    natoms, nfuncs, atoms_ixs, funcs_ixs, func_mask, masks = L.masks[type]

    if n < natoms:
        return Ref(atoms_ixs[n])

    n -= natoms

    func_mask, func_last, func_leap = func_mask
    if func_mask > 0 and n >= func_last:
        n += (n // func_last) * func_leap

    ix = select_mask(n, func_mask)

    func_ix = funcs_ixs[ix]
    tailtypes = L.terms[func_ix].tailtypes

    tails = []
    for tailtype, (tail_mask, tail_last, tail_leap) in zip(tailtypes, masks[ix]):
        if n >= tail_last:
            n += (n // tail_last) * tail_leap

        tails.append(growtree(L, tailtype, select_mask(n, tail_mask)))

    return App(Ref(func_ix), *tails)

if __name__ == '__main__':
    assert collapse(0b1) == 0b1
    assert collapse(0b10) == 0b1
    assert collapse(0b100) == 0b1
    assert collapse(0b101) == 0b11
    assert collapse(0b111) == 0b111
    assert collapse(0b111000) == 0b111
    assert collapse(0b101010) == 0b111

    assert select_mask(0b11, 0b10) == 0b1
    assert select_mask(0b11, 0b1001) == 0b1
    assert select_mask(0b1000, 0b1111) == 0b1000
    assert select_mask(0b1001, 0b1001) == 0b11

    assert inverse_mask(0b1, 0b1) == 1
    assert inverse_mask(0b01, 0b11) == 1
    assert inverse_mask(0b01, 0b11) == 0b1
    assert inverse_mask(0b10, 0b11) == 0b10
    assert inverse_mask(0b001, 0b101) == 0b1
    assert inverse_mask(0b101, 0b10101) == 0b10001

    mask, last, leap = create_mask([Type('x'), Type('y')], Type('x'), TypeMask(natoms=1, nfuncs=0))
    assert mask == 1
    assert last == 1
    assert leap == 1
    assert select_mask((last+leap), mask) == 0

    mask, last, leap = create_mask([Type('x'), Type('y')], Type('x'), TypeMask(natoms=2, nfuncs=0))
    assert mask == 1
    assert last == 2
    assert leap == 0
    assert select_mask((last+leap), mask) == 0

    mask, last, leap = create_mask([Type('x'), Type('y'), Type('x')], Type('x'), TypeMask(natoms=3, nfuncs=0))
    assert mask == 5
    assert last == 5
    assert leap == 3
    assert select_mask((last+leap), mask) == 0

    TL = Library([
        Term(lambda x, y: x + y, Type('Int'), [Type('Int'), Type('Int')], repr='+'),
        Term(1, Type('Int'), repr='1'),
        Term(10, Type('Int'), repr='10'),
    ], Type('Int'))

    plus10 = Term(encode(parse("λ (+ $0 10)"), TL), Type('Int'), [Type('Int')], repr='+10')
    TL.add(plus10)
    plus20 = Term(encode(parse("λ (+10 (+10 $0))"), TL), Type('Int'), [Type('Int')], repr='+20')
    TL.add(plus20)

    assert TL("(+10 10)") == 20
    assert TL("(+10 ((λ $0) 10))") == 20
    assert TL("((λ ($0 10)) +10)") == 20
    assert TL("(((λ (λ ($1 $0))) +10) 10)") == 20
    assert TL("(((λ (λ ($1 $0))) +20) 10)") == 30

    L = TL
    stime = time()
    trees = set()
    for n in range(5000):
        tree = growtree(L, L.type, n)
        out = reduce(tree, L)
        if n < 100:
            print(f'{n} {out}', decode(tree, L))
        trees.add(tree)

    print(f'{time() - stime:.2f}s')
    print(f'{len(trees)=}')
