from blob import *
from data import Type, TypeView, Term
from bits import create_mask, select_mask

def length(t) -> int:
    if not isinstance(t, tuple):
        return 1
    return sum(map(length, t))

class Language:
    def __init__(self, axioms, type: Type):
        self.axioms = axioms
        self.invented = []
        self.type = type
        self.prepare()

    def prepare(self):
        # keep this ordering
        self.terms = self.axioms + self.invented
        # mapping type -> [term_ix]
        self.bytype = defaultdict(list)

        # fill in type pool
        for ix, term in enumerate(self.terms):
            self.bytype[term.type].append(ix)

        # reconstruct terms
        for term_ix, term in enumerate(self.terms):
            updated_tailtypes = []

            for tail_ix, (tailtype, forbidden) in enumerate(zip_longest(term.tailtypes, term.forbidden)):
                forbidden = forbidden or []
                # sieve forbidden funcs
                ixs = [ix for ix in self.bytype[tailtype] if not self.terms[ix].repr in forbidden]
                unique_type = Type(tailtype.type, term.repr, tail_ix)

                self.bytype[unique_type] = ixs
                updated_tailtypes.append(unique_type)

            self.terms[term_ix] = term._replace(tailtypes=updated_tailtypes)

        self.funcs = [term.head for term in self.terms]
        self.bytype = {type: array(ixs) for type, ixs in self.bytype.items()}
        self.views = make_views(self)

    def add(self, term):
        self.invented.append(term)
        self.terms = self.axioms + self.invented
        self.prepare()

    def index(self, repr: str):
        for ix, term in enumerate(self.terms):
            if term.repr == repr:
                return ix

    def __getitem__(self, ix): # pretend this is list storing lambdas
        return self.funcs[ix]

    def reset(self):
        self.invented = []
        self.prepare()

    def __len__(self): return len(self.terms)
    def __repr__(self): return repr(self.terms)
    def __hash__(self): return sum(hash(term[0]) for term in self.invented)

def make_views(L):
    views = {}
    # segragate atoms/funcs for each type
    for type, ixs in L.bytype.items():
        atoms_ixs = [ix for ix in ixs if len(L.terms[ix].tailtypes) == 0]
        funcs_ixs = [ix for ix in ixs if len(L.terms[ix].tailtypes) >= 1]
        natoms = len(atoms_ixs)
        nfuncs = len(funcs_ixs)
        views[type] = TypeView(natoms, nfuncs, atoms_ixs, funcs_ixs)

    # create masks for bitstrings
    bytype_masks = {}
    for type, typeview in views.items():
        if typeview.nfuncs == 0:
            continue

        bitstring = [0] * math.ceil(np.log2(typeview.nfuncs))
        func_mask = create_mask(bitstring, 0, typeview, func=True)

        max_bit = 32
        masks, leaps = [], []
        for fix in typeview.funcs_ixs:
            tailtypes = L.terms[fix].tailtypes
            func_bitstring = copy(bitstring)

            while len(func_bitstring) < max_bit:
                for tailtype in tailtypes:
                    tail_typeview = views[tailtype]

                    if tail_typeview.nfuncs == 0:
                        nbits_allocated = sum(t == tailtype for t in func_bitstring)
                        if 2 ** nbits_allocated > tail_typeview.natoms:
                            # give room to other type if there are any
                            if len(tailtypes) > 1:
                                continue

                    func_bitstring.append(tailtype)

            fix_masks = [create_mask(func_bitstring, t, views[t]) for t in tailtypes]
            masks.append(fix_masks)

        views[type] = typeview._replace(masks=masks, func_mask=func_mask)

    return views

@lru_cache(maxsize=1<<20)
def maketree(L, type, n):
    natoms, nfuncs, atoms_ixs, funcs_ixs, func_mask, masks = L.views[type]

    if n < natoms:
        return int(-atoms_ixs[n]-1)

    n -= natoms

    func_mask, func_last, func_leap = func_mask
    if func_mask > 0 and n >= func_last:
        n += (n // func_last) * func_leap

    func_ix = select_mask(n, func_mask)

    func = funcs_ixs[func_ix]
    tailtypes = L.terms[func].tailtypes

    tails = []
    for tailtype, (tail_mask, tail_last, tail_leap) in zip(tailtypes, masks[func_ix]):
        if n >= tail_last:
            n += (n // tail_last) * tail_leap

        tails.append(maketree(L, tailtype, select_mask(n, tail_mask)))

    return (int(-func-1), *tails)

def shift(λ, s: int, l: int):
    match λ:
        case ('λ', body): return ('λ', shift(body, s, l+1))
        case (f, x): return (shift(f, s, l), shift(x, s, l))
        case int(ix): return ix if ix < l else ix + s
        case _: return λ

def subst(λ, n: int, e):
    match λ:
        case ('λ', body): return ('λ', subst(body, n+1, shift(e, 1, 0)))
        case (f, *xs): return (subst(f, n, e), *[subst(x, n, e) for x in xs])
        case int(ix): return e if n == ix else ix
        case _: return λ

def redux(λ, L=[]):
    match λ:
        case (('λ', f), x): return shift(subst(f, 0, shift(x, 1, 0)), -1, 0)
        case ('λ', body): return ('λ', redux(body, L))
        case (int(ix), *xs) if ix < 0: # fn from the library
            fn = L[abs(ix+1)]
            if isinstance(fn, tuple):
                return redux((fn, *xs), L)

            args = [reduce(x, L) for x in xs]
            return fn(*args)
        case (f, x): return (redux(f, L), redux(x, L))
        case _: return λ

def reduce(λ, L=[], limit=100):
    while (reduced := redux(λ, L)) != λ and limit:
        λ = reduced
        limit -= 1
    if isinstance(λ, int) and λ < 0:
        return L[abs(λ+1)]
    return λ

def parse(s: str):
    "Parse string into ast"
    ast = []
    ix = 0
    while ix < len(s):
        if s[ix] == '(':
            nopen = 1
            six = ix
            while nopen != 0:
                ix += 1
                if s[ix] == ')':
                    nopen -= 1
                elif s[ix] == '(':
                    nopen += 1
            ast.append(parse(s[six+1:ix]))
        else:
            chars = []
            while ix < len(s) and not s[ix].isspace():
                chars.append(s[ix])
                ix += 1
            if chars:
                ast.append(''.join(chars))
        ix += 1
    return ast

def index(library, name): # covers ValueError and complements index
    try: return -library.index(name)-1
    except: return None

def tolang(ast, L=[]):
    "Convert the raw ast by replacing names with their indices from L"
    match ast:
        case [ast]: return tolang(ast, L)
        case ['λ', body]: return ('λ', tolang(body, L))
        case [*xs]: return tuple(tolang(x, L) for x in ast)
        case name if (ix := index(L, name)) is not None: return ix
        case hole if hole[0] == '?': return hole
        case debruijn if debruijn[0] == '$': return int(debruijn[1:])
        case _: raise ValueError(f"{ast}, it's all greek to me")

def inlang(λ, L=[]):
    match λ:
        case (f, *xs): return f'({inlang(f, L)} {" ".join([inlang(x, L) for x in xs])})'
        case int(ix) if ix < 0: return L.terms[abs(ix+1)].repr
        case _: return repr(λ)

if __name__ == '__main__':
    succ = ('λ', ('λ', ('λ', (1, ((2, 1), 0)))))
    zero = ('λ', ('λ', 0))
    four = ('λ', ('λ', (1, (1, (1, (1, 0))))))
    assert reduce((succ, (succ, (succ, (succ, zero))))) == four

    Y = ('λ', (('λ', (1, (0, 0))), ('λ', (1, (0, 0)))))
    assert redux(redux((Y, Y))) == (Y, redux((Y, Y)))

    cons = ('λ', ('λ', ('λ', ((0, 2), 1))))
    car = ('λ', (0, ('λ', ('λ', 1))))
    cdr = ('λ', (0, ('λ', ('λ', 0))))

    λ = ((cons, four), ((cons, four), four))
    assert reduce((car, λ)) == reduce((car, (cdr, λ))) == reduce((cdr, (cdr, λ)))

    L = [('λ', 0), lambda x: x**2, 10]
    λ = (-2, (-2, (-1, -3)))
    assert reduce(λ, L) == 10000

    assert length((1, 1, 1)) == 3
    assert length((((1,), (1,), (1,)), 1, 1)) == 5

    assert parse("abc (ijk (xyz))") == ['abc', ['ijk', ['xyz']]]
    assert parse("(-111 000 111)") == [['-111', '000', '111']]
    assert parse("(λ 0) (-1 (λ 0))") == [['λ', '0'], ['-1', ['λ', '0']]]

    TL = Language([
        Term(add, Type('Int'), [Type('Int'), Type('Int')], repr='+'),
        Term(1, Type('Int'), repr='1'),
        Term(10, Type('Int'), repr='10'),
    ], Type('Int'))

    plus10 = Term(tolang(parse("λ (+ $0 10)"), TL), Type('Int'), [Type('Int')], repr='+10')
    TL.add(plus10)

    assert reduce(tolang(parse("(+10 10)"), TL), TL) == 20
    assert reduce(tolang(parse("(+10 ((λ $0) 10))"), TL), TL) == 20
    assert reduce(tolang(parse("((λ ($0 10)) +10)"), TL), TL) == 20
    assert reduce(tolang(parse("(((λ (λ ($1 $0))) +10) 10)"), TL), TL) == 20

    SL = Language([
        Term(0, Type('Int'), repr='ø'),
        Term(lambda x: x + 1, Type('Int'), [Type('Int')], repr='S'),
    ], Type('Int'))

    stime = time()
    trees = set()
    for n in range(10000):
        tree = maketree(TL, TL.type, n)
        out = reduce(tree, TL)
        if n < 100:
            print(f'{n} {out}', inlang(tree, TL))
        trees.add(tree)

    print(f'{time() - stime:.2f}s')
    print(f'{len(trees)=}')


