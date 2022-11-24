from blob import *
from data import Type, TypeView, Term, T
from bits import create_mask, select_mask

class Language:
    def __init__(self, axioms, type: Type):
        self.axioms = axioms
        self.invented = []
        self.type = type
        self.update_types()

    def update_types(self):
        # keep this ordering
        self.terms = self.axioms + self.invented
        # mapping type -> [term_ix]
        self.bytype = defaultdict(list)

        # fill in type pool
        for ix, term in enumerate(self):
            self.bytype[term.type].append(ix)

        # reconstruct terms
        for term_ix, term in enumerate(self):
            updated_tailtypes = []

            for tail_ix, (tailtype, forbidden) in enumerate(zip_longest(term.tailtypes, term.forbidden)):
                forbidden = forbidden or []
                # sieve forbidden funcs
                ixs = [ix for ix in self.bytype[tailtype] if not self[ix].repr in forbidden]
                unique_type = Type(tailtype.type, term.repr, tail_ix)

                self.bytype[unique_type] = ixs
                updated_tailtypes.append(unique_type)

            self.terms[term_ix] = term._replace(tailtypes=updated_tailtypes)

        self.bytype = {type: array(ixs) for type, ixs in self.bytype.items()}
        self.rbytype = {type: [self.terms[ix] for ix in ixs] for type, ixs in self.bytype.items()}
        self.views = make_views(self)

        if os.environ.get('debug'):
            for type, view in self.views.items():
                print(type)
                pprint(view._asdict())
                for mask in view.masks:
                    for (mask, last, leap) in mask:
                        print(f'{bin(mask)=}, {bin(last)=}, {bin(leap)=}')

    def reset(self):
        self.invented = []
        self.update_types()

    def add(self, term):
        self.invented.append(term)
        self.terms = self.axioms + self.invented

    def __getitem__(self, x):
        if isinstance(x, (int, np.int64)):
            assert x >= 0
            return self.terms[x]

        if (ix := self.index(x)) is None:
            raise ValueError(f'there is no {x} in {self.terms}')
        return self.terms[ix]

    def __len__(self):
        return len(self.terms)

    def __repr__(self):
        return repr(self.terms)

    def index(self, repr: str):
        for ix, t in enumerate(self.terms):
            if t.repr == repr:
                return ix

        return None

    def __hash__(self):
        return sum(hash(term[0]) for term in self.invented)

    def __mod__(self, expr: str):
        return tolang(self, parse(expr))

    def __call__(self, term: T):
        if isinstance(term, str):
            term = interpret(self, parse(term))

        return evaltree(self, term)

    def __lt__(self, term: T):
        return grepr(self, term)


def tolang(L: Language, ast: list) -> T:
    match ast:
        case [ast] if isinstance(ast, list):
            return tolang(L, ast)
        case [*ast] if len(ast):
            return T(tolang(L, a) for a in ast)
        case head if (ix := L.index(head)) is not None:
            return ix
        case hole if hole[0].isupper() or hole[0] == '.' or hole[0] == '$':
            return hole
        case debruijn if (ix := int(debruijn)) < 0:
            return ix
        case _:
            raise ValueError(f"{ast}, it's all greek to me")

def make_views(L):
    views = {}
    # segragate atoms/funcs for each type
    for type, ixs in L.bytype.items():
        atoms_ixs = [ix for ix in ixs if len(L[ix].tailtypes) == 0]
        funcs_ixs = [ix for ix in ixs if len(L[ix].tailtypes) >= 1]
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
            tailtypes = L[fix].tailtypes
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
        return atoms_ixs[n]

    n -= natoms

    func_mask, func_last, func_leap = func_mask
    if func_mask > 0 and n >= func_last:
        n += (n // func_last) * func_leap

    func_ix = select_mask(n, func_mask)

    func = funcs_ixs[func_ix]
    tailtypes = L[func].tailtypes

    tails = []
    for tailtype, (tail_mask, tail_last, tail_leap) in zip(tailtypes, masks[func_ix]):
        if n >= tail_last:
            n += (n // tail_last) * tail_leap

        tails.append(maketree(L, tailtype, select_mask(n, tail_mask)))

    return (func, *tails)

def grepr(L, t):
    if not isinstance(t, (T, int, np.int64)):
        return t

    if not isinstance(t, T):
        t = T((t,))

    head, *tails = t

    if isinstance(head, (int, np.int64)) and head < 0:
        return str(t)

    if isinstance(head, T):
        if tails:
            return f"(λ[{head}]. {' '.join([grepr(L, t) for t in tails])})"

        return f"λ[{head}]"

    if not tails:
        return L[head].repr

    return f"({L[head].repr} {' '.join([grepr(L, t) for t in tails])})"

def length(t: T) -> int:
    if not isinstance(t, tuple):
        return 1

    return sum(map(length, t))

# this is overly complex because of some idiosyncraties i'm lazy to clean up
def eval(L: Language, t: T, args=()):
    if isinstance(t, (np.int64, int)):
        if t < 0: # debruijn
            if len(args) >= abs(t):
                return args[t]
            else:
                return t

        return L[t].head
    else:
        head, *tails = t
        if isinstance(head, (np.int64, int)):
            if head < 0: # debruijn
                if len(args) == 0:
                    return t
                if len(args) >= abs(head):
                    head = args[head]
                else:
                    head = head
            else:
                head = L[head].head

        if len(tails) == 0: # atom
            return head

    tails = tuple([eval(L, tail, args) for tail in tails])

    if isinstance(head, (np.int64, int)):
        return (head, *tails)
    if isinstance(head, T): # abstraction
        return eval(L, head, tails)

    return head(*tails)

def isequal(t1, t2):
    return t1.head == t2.head and all(isequal(tail1, tail2) for tail1, tail2 in zip(t1.tails, t2.tails))

def rewrite(source, match, target, withargs=False):
    if not source.tails:
        return source

    if isequalhole(source, match):
        if withargs:
            args = []
            extractargs(source, match, args)

            args = list(map(lambda g: rewrite(g, match, target, True), args))
            # again reversed for - access
            return T(target.head, tuple(reversed(args)))

        return T(target.head)

    newtails = [None] * len(source.tails)
    for tind, tail in enumerate(source.tails):
        if isequalhole(tail, match):
            if withargs:
                args = []
                extractargs(tail, match, args)
                # rewrite arguments themselves
                # good that it's only one rewrite each for now
                args = list(map(lambda g: rewrite(g, match, target, True), args))

                newtails[tind] = T(target.head, tuple(reversed(args)))
            else:
                newtails[tind] = target
        else:
            newtails[tind] = rewrite(tail, match, target, withargs=withargs)

    return source._replace(tails=tuple(newtails))

def pevalg(L, t, args):
    if isinstance(t.head, int) and t.head < 0:
        return args[t.head]

    if not t.tails:
        return t

    return t._replace(tails=tuple([pevalg(L, tail, args) for tail in t.tails]))

def isnormal(L, t):
    if t.head >= len(L.axioms):
        return False

    return all(isnormal(L, tail) for tail in t.tails)

def normalize(L, t):
    if t.head >= len(L.axioms):
        # get hiddentail
        body = deepcopy(L[t.head].head)
        # insert tails
        body = pevalg(L, body, t.tails)

        while not isnormal(L, body):
            body = normalize(L, body)

        return body

    return t._replace(tails=tuple([normalize(L, tail) for tail in t.tails]))

def parse(s: str):
    "Parse a string into an AST"
    ast = []
    ind = 0

    while ind < len(s):
        if s[ind] == '(':
            nopen = 1
            sind = ind

            while nopen != 0:
                ind += 1
                if s[ind] == ')':
                    nopen -= 1
                if s[ind] == '(':
                    nopen += 1

            ast.append(parse(s[sind+1:ind]))

        else:
            term = []
            while ind < len(s) and not s[ind].isspace():
                term.append(s[ind])
                ind += 1

            if term:
                ast.append(''.join(term))

        ind += 1

    return ast

def flatten(xs):
    for x in xs:
        yield from x

def everysubtree(t):
    qq = [t]

    while qq:
        n = qq.pop(0)
        qq.extend(n.tails)
        yield n

def lent(t: T | str):
    "natoms+nops, nargs"
    # types
    if isinstance(t, str):
        return [0, 1]
    # eclass
    if isinstance(t, int):
        return [1, 0]

    return list(reduce(lambda acc, x: [acc[0] + x[0], acc[1] + x[1]], map(lent, t.tails), [1, 0]))

if __name__ == '__main__':
    assert length((1, 1, 1)) == 3
    assert length((((1,), (1,), (1,)), 1, 1)) == 5

    assert parse("(abc (ijk (xyz)))") == [['abc', ['ijk', ['xyz']]]]
    assert parse("(-111 000 +111)") == [['-111', '000', '+111']]
    assert parse("(λ (x) (x x))") == [['λ', ['x'], ['x', 'x']]]


    TL = Language([
        Term(add, Type('Int'), [Type('Int'), Type('Int')], repr='+'),
        Term(1, Type('Int'), repr='1'),
        Term(10, Type('Int'), repr='10'),
    ], Type('Int'))


    a0 = (0, 1, 1)
    assert tolang(TL, parse("(+ 1 1)")) == a0
    a1 = (0, a0, 1)
    assert tolang(TL, parse("(+ (+ 1 1) 1)")) == a1
    a2 = (0, a0, a0)
    assert tolang(TL, parse("(+ (+ 1 1) (+ 1 1))")) == a2

    am0 = (0, -1, 1)
    assert tolang(TL, parse("(+ -1 1)")) == am0
    am1 = (am0, a0)
    assert tolang(TL, parse("((+ -1 1) (+ 1 1))")) == am1
    am2 = (am0, am0)
    assert tolang(TL, parse("((+ -1 1) (+ -1 1))")) == am2


    assert eval(TL, TL%"(+ 1 1)") == 2
    assert eval(TL, TL%"(+ (+ 1 1) 1)") == 3
    assert eval(TL, TL%"(+ (+ 1 1) (+ 1 1))") == 4
    assert eval(TL, TL%"((+ -1 1) 1)") == 2
    assert eval(TL, TL%"((+ -1 -1) 1)") == 2
    assert eval(TL, TL%"((+ -1 ((+ -1 -1) 1)) 10)") == 12
    # hof
    assert eval(TL, TL%"((-3 -2 -1) + 10 1)") == 11
    assert eval(TL, TL%"((-3 -2 (-3 -1 -1)) + 10 1)") == 12
    assert eval(TL, TL%"((-3 -2 (-3 -1 -1)) + ((-1 10 10) +) 1)") == 22
    assert eval(TL, TL%"((-1) (-1))") == (-1,)
    assert eval(TL, TL%"(((-1) (-1)) (-2))") == (-2,)


    L = Language([
        Term(0, Type('N'), repr='ø'),
        Term('S', Type('N'), [Type('N')], [[]], repr='S'),
    ], Type('N'))

    stime = time()
    trees = set()
    for n in range(1000):
        tree = maketree(L, L.type, n)
        if n < 100:
            print(f'***** {n}', L<tree)
        trees.add(tree)

    print(f'{time() - stime:.2f}s')
    print(f'{len(trees)=}')

