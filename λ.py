from dataclasses import dataclass

@dataclass
class Var:
    ix: int
    def __repr__(self): return f'${self.ix}'
    def __hash__(self): return int(self.ix)

@dataclass
class Ref:
    ix: int
    def __repr__(self): return f'#{self.ix}'
    def __hash__(self): return -int(self.ix)-1

@dataclass
class Lam:
    body: any
    def __repr__(self): return f'(λ {self.body})'
    def __hash__(self): return hash(self.body)

@dataclass
class App:
    __match_args__ = ('f', 'xs')
    def __init__(self, f, *xs):
        self.f = f
        self.xs = xs

    def __repr__(self): return f'({repr(self.f)} {" ".join(repr(x) for x in self.xs)})'
    def __hash__(self): return hash((self.f, self.xs))
    def __eq__(self, λ): return isinstance(λ, App) and self.f == λ.f and self.xs == λ.xs

def shift(λ, s: int, l: int):
    match λ:
        case Lam(body): return Lam(shift(body, s, l+1))
        case App(f, (x,)): return App(shift(f, s, l), shift(x, s, l))
        case Var(ix): return λ if ix < l else Var(ix + s)
        case _: return λ

def subst(λ, n: int, e):
    match λ:
        case Lam(body): return Lam(subst(body, n+1, shift(e, 1, 0)))
        case App(f, xs): return App(subst(f, n, e), *[subst(x, n, e) for x in xs])
        case Var(ix): return e if n == ix else λ
        case _: return λ

def redux(λ, L=[]):
    match λ:
        case App(Lam(f), (x,)): return shift(subst(f, 0, shift(x, 1, 0)), -1, 0)
        case Lam(body): return Lam(redux(body, L))
        case App(f, xs) if callable(f): return f(*[reduce(x, L) for x in xs])
        case App(f, xs): return App(redux(f, L), *[redux(x, L) for x in xs])
        case Ref(ix): return L[ix]
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

def encode(ast, L=[]):
    "Convert the raw ast by replacing names with their indices from L"
    match ast:
        case [ast]: return encode(ast, L)
        case ['λ', body]: return Lam(encode(body, L))
        case [*xs]: return App(*[encode(x, L) for x in ast])
        case hole if hole[0] == '?': return hole
        case debruijn if debruijn[0] == '$': return Var(int(debruijn[1:]))
        case reference if reference[0] == '#': return Ref(int(reference[1:]))
        case name if (ix := L.index(name)) is not None: return Ref(ix)
        case _: raise ValueError(f"{ast}, it's all greek to me")

def decode(λ, L=[]):
    match λ:
        case App(f, xs): return f'({decode(f, L)} {" ".join([decode(x, L) for x in xs])})'
        case Ref(ix): return L.terms[ix].repr
        case _: return repr(λ)

def length(λ) -> int:
    match λ:
        case App(f, xs): return length(f) + sum(length(x) for x in xs)
        case Lam(body): return 1 + length(body)
        case _: return 1

if __name__ == '__main__':
    assert parse("abc (ijk (xyz))") == ['abc', ['ijk', ['xyz']]]
    assert parse("(-111 000 111)") == [['-111', '000', '111']]
    assert parse("(λ $0) (#1 (λ $0))") == [['λ', '$0'], ['#1', ['λ', '$0']]]

    T = lambda s: encode(parse(s))

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

    succ = T("(λ (λ (λ ($1 (($2 $1) $0)))))")
    zero = T("(λ (λ $0))")
    four = T("(λ (λ ($1 ($1 ($1 ($1 $0))))))")
    assert reduce(App(succ, App(succ, App(succ, App(succ, zero))))) == four

    Y = T("(λ ((λ ($1 ($0 $0))) (λ ($1 ($0 $0)))))")
    assert redux(redux(App(Y, Y))) == App(Y, redux(App(Y, Y)))

    cons = T("(λ (λ (λ (($0 $2) $1))))")
    car = T("(λ ($0 (λ (λ $1))))")
    cdr = T("(λ ($0 (λ (λ $0))))")

    λ = App(App(cons, four), App(App(cons, four), four))
    assert reduce(App(car, λ)) == reduce(App(car, App(cdr, λ))) == reduce(App(cdr, App(cdr, λ)))

    L = [T("(λ $0)"), lambda x: x**2, 10]
    λ = T("(#1 (#1 (#0 #2)))")
    assert reduce(λ, L) == 10000
    assert length(λ) == 4

