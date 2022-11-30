from blob import *
from data import Type, TypeView, Term
from bits import create_mask, select_mask

@dataclass
class Debruijn:
    ix: int
    def __repr__(self): return f'${self.ix}'
    def __hash__(self): return int(self.ix)

@dataclass
class Reference:
    ix: int
    def __repr__(self): return f'#{self.ix}'
    def __hash__(self): return int(self.ix)

def shift(λ, s: int, l: int):
    match λ:
        case ('λ', body): return ('λ', shift(body, s, l+1))
        case (f, x): return (shift(f, s, l), shift(x, s, l))
        case Debruijn(ix): return λ if ix < l else Debruijn(ix + s)
        case _: return λ

def subst(λ, n: int, e):
    match λ:
        case ('λ', body): return ('λ', subst(body, n+1, shift(e, 1, 0)))
        case (f, *xs): return (subst(f, n, e), *[subst(x, n, e) for x in xs])
        case Debruijn(ix): return e if n == ix else λ
        case _: return λ

def redux(λ, L=[]):
    match λ:
        case (('λ', f), x): return shift(subst(f, 0, shift(x, 1, 0)), -1, 0)
        case ('λ', body): return ('λ', redux(body, L))
        case (f, *xs) if callable(f): return f(*[reduce(x, L) for x in xs])
        case (f, *xs): return (redux(f, L), *[redux(x, L) for x in xs])
        case Reference(ix): return L[ix]
        case _: return λ

def reduce(λ, L=[], limit=100):
    while (reduced := redux(λ, L)) != λ and limit:
        λ = reduced
        limit -= 1
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

def tolang(ast, L=[]):
    "Convert the raw ast by replacing names with their indices from L"
    match ast:
        case [ast]: return tolang(ast, L)
        case ['λ', body]: return ('λ', tolang(body, L))
        case [*xs]: return tuple(tolang(x, L) for x in ast)
        case name if (ix := index(L, name)) is not None: return Reference(ix)
        case hole if hole[0] == '?': return hole
        case debruijn if debruijn[0] == '$': return Debruijn(int(debruijn[1:]))
        case reference if reference[0] == '#': return Reference(int(reference[1:]))
        case _: raise ValueError(f"{ast}, it's all greek to me")

def inlang(λ, L=[]):
    match λ:
        case (f, *xs): return f'({inlang(f, L)} {" ".join([inlang(x, L) for x in xs])})'
        case Debruijn(ix) if ix < 0: return L.terms[abs(ix+1)].repr
        case _: return repr(λ)

class Language:
    "Wrapper over an array of primitives/invented abstrations"
    def __init__(self, primitives, type: Type):
        self.primitives = primitives
        self.invented = []
        self.type = type
        self.prepare()

    def add(self, term):
        self.invented.append(term)
        self.terms = self.axioms + self.invented
        self.prepare()

    def index(self, repr: str):
        for ix, term in enumerate(self.terms):
            if term.repr == repr:
                return ix

    def __getitem__(self, ix):
        return self.funcs[ix]

    def reset(self):
        self.invented = []
        self.prepare()

    def prepare(self):
        self.terms = self.axioms + self.invented
        # mapping type -> [term_ix]
        self.type_to_ixs = defaultdict(list)

        # fill the type mapping
        for ix, term in enumerate(self.terms):
            self.type_to_ixs[term.type].append(ix)

        # update argument types for each term by specifying it arity index
        # and removing forbidden terms from the mapping
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

        self.funcs = [term.head for term in self.terms]
        self.type_to_ixs = {type: array(ixs) for type, ixs in self.type_to_ixs.items()}
        self.views = make_views(self)

    def __len__(self): return len(self.terms)
    def __repr__(self): return repr(self.terms)
    def __call__(self, string): return reduce(tolang(parse(string), self), self)
    def __lshift__(self, string): return tolang(parse(string), self)
    def __hash__(self): return sum(hash(term[0]) for term in self.invented)

def make_views(L):
    views = {}
    # segragate atoms/funcs for each type
    for type, ixs in L.type_to_ixs.items():
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
def growtree(L, type, n: int):
    natoms, nfuncs, atoms_ixs, funcs_ixs, func_mask, masks = L.views[type]

    if n < natoms:
        return int(-atoms_ixs[n]-1)

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

    return (Reference(func_ix), *tails)

def index(library, name): # covers ValueError and complements index
    try: return library.index(name)
    except: return None

def length(t) -> int:
    if not isinstance(t, tuple):
        return 1
    return sum(map(length, t))

if __name__ == '__main__':
    T = lambda x: tolang(parse(x))

    assert redux(T("(λ (λ $0))")) == T("(λ (λ $0))")
    assert redux(T("(λ (λ $2))")) == T("(λ (λ $2))")
    assert redux(T("((λ $0) $1)")) == T("$1")
    assert redux(T("((λ $0) (λ $2))")) == T("(λ $2)")
    assert redux(T("((λ $2) (λ $0))")) == T("$1")
    assert redux(T("((λ $0) (λ $1))")) == T("(λ $1)")
    assert redux(T("((λ ($0 $0)) $1)")) == T("($1 $1)")
    assert redux(redux(T("((λ ($0 $0)) (λ $0))"))) == T("(λ $0)")
    assert redux(redux(T("((λ (λ (λ ($2 $1)))) (λ $0))"))) == T("(λ (λ $1))")
    assert redux(redux(redux(T("(((λ (λ (λ ($2 $1)))) (λ $0)) (λ $1))")))) == T("(λ (λ $2))")
    assert redux(T("((λ (λ $1)) (λ $0))")) == T("(λ (λ $0))")
    assert redux(T("((λ (λ $0)) $1)")) == T("(λ $0)")
    assert redux(T("((λ (λ $1)) (λ $2))")) == T("(λ (λ $3))")
    assert redux(T("((λ (λ $1)) (λ $3))")) == T("(λ (λ $4))")

    I = Debruijn
    succ = ('λ', ('λ', ('λ', (I(1), ((I(2), I(1)), I(0))))))
    zero = ('λ', ('λ', I(0)))
    four = ('λ', ('λ', (I(1), (I(1), (I(1), (I(1), I(0)))))))
    assert reduce((succ, (succ, (succ, (succ, zero))))) == four

    Y = ('λ', (('λ', (I(1), (I(0), I(0)))), ('λ', (I(1), (I(0), I(0))))))
    assert redux(redux((Y, Y))) == (Y, redux((Y, Y)))

    cons = T("(λ (λ (λ (($0 $2) $1))))")
    car = T("(λ ($0 (λ (λ $1))))")
    cdr = T("(λ ($0 (λ (λ $0))))")

    λ = ((cons, four), ((cons, four), four))
    assert reduce((car, λ)) == reduce((car, (cdr, λ))) == reduce((cdr, (cdr, λ)))

    L = [('λ', I(0)), lambda x: x**2, 10]
    λ = T("(#1 (#1 (#0 #2)))")
    assert reduce(λ, L) == 10000

    assert length((1, 1, 1)) == 3
    assert length((((1,), (1,), (1,)), 1, 1)) == 5

    assert parse("abc (ijk (xyz))") == ['abc', ['ijk', ['xyz']]]
    assert parse("(-111 000 111)") == [['-111', '000', '111']]
    assert parse("(λ $0) (#1 (λ $0))") == [['λ', '$0'], ['#1', ['λ', '$0']]]

    TL = Language([
        Term(add, Type('Int'), [Type('Int'), Type('Int')], repr='+'),
        Term(1, Type('Int'), repr='1'),
        Term(10, Type('Int'), repr='10'),
    ], Type('Int'))

    plus10 = Term(tolang(parse("λ (+ $0 10)"), TL), Type('Int'), [Type('Int')], repr='+10')
    TL.add(plus10)
    plus20 = Term(tolang(parse("λ (+10 (+10 $0))"), TL), Type('Int'), [Type('Int')], repr='+20')
    TL.add(plus20)

    assert TL("(+10 10)") == 20
    assert TL("(+10 ((λ $0) 10))") == 20
    assert TL("((λ ($0 10)) +10)") == 20
    assert TL("(((λ (λ ($1 $0))) +10) 10)") == 20
    assert TL("(((λ (λ ($1 $0))) +20) 10)") == 30

    SL = Language([
        Term(0, Type('Int'), repr='ø'),
        Term(lambda x: x + 1, Type('Int'), [Type('Int')], repr='S'),
    ], Type('Int'))

    L = TL
    stime = time()
    trees = set()
    for n in range(20000):
        tree = growtree(L, L.type, n)
        out = reduce(tree, L)
        if n < 100:
            print(f'{n} {out}', inlang(tree, L))
        trees.add(tree)

    print(f'{time() - stime:.2f}s')
    print(f'{len(trees)=}')
