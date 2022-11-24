from typing import NamedTuple

T = tuple

class Term(NamedTuple):
    head: int
    type: type
    tailtypes: list = []
    forbidden: list = []
    repr: str = '?'

    def __repr__(self):
        return self.repr

class Type(NamedTuple):
    type: str
    parent_func: str = None
    arg_ix: int = None

    def __repr__(self):
        if self.parent_func:
            return f"<{self.type}:{self.parent_func}:{self.arg_ix}>"
        return f"<{self.type}>"

class TypeView(NamedTuple):
    natoms: int
    nfuncs: int
    atoms_ixs: list = []
    funcs_ixs: list = []
    func_mask: list = []
    masks: list = []

