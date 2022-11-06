"utils for the bitstring enumeration"
import math
import numpy as np
from data import Type, TypeView
from numba import njit

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

def create_mask(bitstring, type, typeview, func=False) -> int:
    mask = 0
    for ix, t in enumerate(bitstring):
        if t == type:
            mask += 2 ** ix

    if typeview.nfuncs == 0:
        last = typeview.natoms
    elif func:
        last = typeview.nfuncs
    else:
        last = mask + 1

    # `last` is the first overflow, on which a jump by `leap = mask - last` is required"
    last = inverse_mask(last, mask)
    if mask == 0:
        leap = 0
    else:
        leap = 2**math.floor(math.log2(mask)+1) - last

    return mask, last, leap

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

    mask, last, leap = create_mask([Type('x'), Type('y')], Type('x'), TypeView(natoms=1, nfuncs=0))
    assert mask == 1
    assert last == 1
    assert leap == 1
    assert select_mask((last+leap), mask) == 0

    mask, last, leap = create_mask([Type('x'), Type('y')], Type('x'), TypeView(natoms=2, nfuncs=0))
    assert mask == 1
    assert last == 2
    assert leap == 0
    assert select_mask((last+leap), mask) == 0

    mask, last, leap = create_mask([Type('x'), Type('y'), Type('x')], Type('x'), TypeView(natoms=3, nfuncs=0))
    assert mask == 5
    assert last == 5
    assert leap == 3
    assert select_mask((last+leap), mask) == 0

