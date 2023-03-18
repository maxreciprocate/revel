import numpy

from λ import Lam, App, Var, encode, decode, reduce, parse, length, Ref
from enumerate import Library, growtree, Type, Term
from operator import add, mul
from tqdm import trange

L = Library([
    Term("A", Type('L'), repr="A"),
    Term("C", Type('L'), repr="C"),
    Term("T", Type('L'), repr="T"),
    Term("G", Type('L'), repr="G"),
    Term(2, Type('N'), repr="2"),
    Term(3, Type('N'), repr="3"),
    Term(add, Type('L'), [Type('L'), Type('L')], repr="."),
    Term(mul, Type('L'), [Type('L'), Type('N')], repr="*"),
], Type("L"))

# ■ ~
# source = "".join(list(numpy.random.choice(["A", "C", "T", "G"], size=64)))
source = "TTCATCCACAAAATTTAAAAGCTCTGATCGAGTTGCANAGTTCCAGATAATCTGTNNATCAAANAT".replace("N", "A")

def to_primitive(xs):
    x, *xs = xs
    if len(xs) == 1:
        return L.index(xs[0])
    return App(Ref(L.index(".")), Ref(L.index(x)), to_primitive(xs))

original = to_primitive(source)
length(original)
# ■ ~

cutlen = 8
xs = [source[ix:ix+cutlen] for ix in range(0, len(source), cutlen)]
source
programs = [None] * len(xs)
left = len(xs)
length(f)

for n in trange(10**9):
    if left == 0:
        break
    f = growtree(L, L.type, n)
    out = reduce(f, L)

    for ix, x in enumerate(xs):
        if out == x:
            if programs[ix] is None:
                programs[ix] = f
                print(programs)
                left -= 1
            elif length(f) < length(programs[ix]):
                programs[ix] = f
                print(programs)

current = programs[-1]
for p in programs[-2::-1]:
    current = App(Ref(L.index(".")), p, current)

reduce(current, L) == source
length(current)
