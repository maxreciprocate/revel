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

    def __getitem__(self, ix):
        if isinstance(ix, (int, np.int64)):
            return self.terms[ix]

        return self.terms[self.index(ix)]

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
        return sum(hash(term.head) for term in self.invented)

    def __mod__(self, expr: str):
        return tolang(self, parse(expr))

    def __call__(self, term: T):
        if isinstance(term, str):
            term = interpret(self, parse(term))

        return evaltree(self, term)

    def __lt__(self, term: T):
        return grepr(self, term)

def tolang(L: Language, ast) -> T:
    match ast:
        case [ast] if isinstance(ast, list):
            return tolang(L, ast)
        case [head, *tails] if len(tails):
            return T(L.index(head), tuple(tolang(L, t) for t in tails))
        case [var]:
            return tolang(L, var)
        case head if (ix := L.index(head)) is not None:
            return T(ix)
        case hole if hole[0].isupper() or hole[0] == '.' or hole[0] == '$':
            return hole
        case debruijn if (ix := int(debruijn)) < 0:
            return T(ix)
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

@lru_cache(maxsize=None)
def maketree(L, type, n):
    natoms, nfuncs, atoms_ixs, funcs_ixs, func_mask, masks = L.views[type]

    if n < natoms:
        return T(atoms_ixs[n])

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

    return T(func, tuple(tails))

def grepr(L, t):
    if not isinstance(t, (T, int)):
        return t

    if isinstance(t.head, T):
        if t.tails:
            return f"(λ[{t.head}]. {' '.join([grepr(L, t) for t in t.tails])})"

        return f"λ[{t.head}]"

    if isinstance(t.head, int) and t.head < 0:
        return str(t.head)

    if not t.tails:
        return L[t.head].repr

    return f"({L[t.head].repr} {' '.join([grepr(L, t) for t in t.tails])})"

def length(t: T) -> int:
    if isinstance(t, (int, str)):
        return 1

    return 1 + sum(map(length, t.tails))

def evaltree(L, t, args=()):
    # debuijn index
    if isinstance(t.head, int) and t.head < 0:
        return args[t.head]

    # atom
    if not t.tails:
        return L[t.head].head

    # application
    f = L[t.head].head
    tails = tuple([evaltree(L, tail, args) for tail in t.tails])

    # abstraction
    if isinstance(f, T):
        return evaltree(L, f, tails)

    return f(*tails)

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
    assert parse("(abc (ijk (xyz)))") == [['abc', ['ijk', ['xyz']]]]
    assert parse("(-111 000 +111)") == [['-111', '000', '+111']]
    assert parse("(λ (x) (x x))") == [['λ', ['x'], ['x', 'x']]]

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

