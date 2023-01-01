use babble::{
    ast_node::{combine_exprs, Arity, AstNode, Expr, Precedence, Printable, Printer, Pretty},
    learn::{LibId, ParseLibIdError},
    experiments::Experiments,
    sexp::Program,
    teachable::{BindingExpr, DeBruijnIndex, Teachable},
};
use std::{
    fs,
    convert::{TryInto, Infallible},
    fmt::{self, Display, Formatter, Write},
    str::FromStr,
};

use egg::Symbol;

/// List operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ListOp {
    /// Add an element to the front of a list
    Cons,
    /// A boolean literal
    Bool(bool),
    /// A conditional expression
    If,
    /// An integer literal
    Int(i32),
    /// A function application
    Apply,
    /// A de Bruijn-indexed variable
    Var(DeBruijnIndex),
    /// An identifier
    Ident(Symbol),
    /// An anonymous function
    Lambda,
    /// A library function binding
    Lib(LibId),
    /// A reference to a lib var
    LibVar(LibId),
    /// A list
    List,
    /// A shift
    Shift,
    /// Variable arity is problematic
    Rotate,
    Move,
    Penup,
}

impl Arity for ListOp {
    fn min_arity(&self) -> usize {
        match self {
            Self::Bool(_)
            | Self::Int(_)
            | Self::Var(_)
            | Self::Ident(_)
            | Self::LibVar(_)
            | Self::List => 0,
            Self::Lambda | Self::Shift => 1,
            Self::Cons | Self::Apply | Self::Lib(_) => 2,
            Self::Rotate | Self::Move | Self::Penup => 2,
            Self::If => 3,
        }
    }

    fn max_arity(&self) -> Option<usize> {
        match self {
            Self::List => None,
            other => Some(other.min_arity()),
        }
    }
}

impl Display for ListOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Rotate => "rotate",
            Self::Move => "move",
            Self::Penup => "penup",
            Self::Cons => "cons",
            Self::If => "if",
            Self::Apply => "@",
            Self::Lambda => "λ",
            Self::Shift => "shift",
            Self::List => "list",
            Self::Lib(ix) => {
                return write!(f, "lib {}", ix);
            }
            Self::LibVar(ix) => {
                return write!(f, "{}", ix);
            }
            Self::Bool(b) => {
                return write!(f, "{}", b);
            }
            Self::Int(i) => {
                return write!(f, "{}", i);
            }
            Self::Var(index) => {
                return write!(f, "{}", index);
            }
            Self::Ident(ident) => {
                return write!(f, "{}", ident);
            }
        };
        f.write_str(s)
    }
}

impl FromStr for ListOp {
    type Err = Infallible;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let op = match input {
            "rotate" => Self::Rotate,
            "move" => Self::Move,
            "penup" => Self::Penup,
            "cons" => Self::Cons,
            "if" => Self::If,
            "shift" => Self::Shift,
            "apply" | "@" => Self::Apply,
            "lambda" | "λ" => Self::Lambda,
            "list" => Self::List,
            input => input
                .parse()
                .map(Self::Bool)
                .or_else(|_| input.parse().map(Self::Var))
                .or_else(|_| input.parse().map(Self::Int))
                .or_else(|_| input.parse().map(Self::LibVar))
                .or_else(|_| {
                    input
                        .strip_prefix("lib ")
                        .ok_or(ParseLibIdError::NoLeadingL)
                        .and_then(|x| x.parse().map(Self::Lib))
                })
                .unwrap_or_else(|_| Self::Ident(input.into())),
        };
        Ok(op)
    }
}

impl Teachable for ListOp {
    fn from_binding_expr<T>(binding_expr: BindingExpr<T>) -> AstNode<Self, T> {
        match binding_expr {
            BindingExpr::Lambda(body) => AstNode::new(Self::Lambda, [body]),
            BindingExpr::Apply(fun, arg) => AstNode::new(Self::Apply, [fun, arg]),
            BindingExpr::Var(index) => AstNode::leaf(Self::Var(index)),
            BindingExpr::Lib(ix, bound_value, body) => {
                AstNode::new(Self::Lib(ix), [bound_value, body])
            }
            BindingExpr::LibVar(ix) => AstNode::leaf(Self::LibVar(ix)),
            BindingExpr::Shift(body) => AstNode::new(Self::Shift, [body]),
        }
    }

    fn as_binding_expr<T>(node: &AstNode<Self, T>) -> Option<BindingExpr<&T>> {
        let binding_expr = match node.as_parts() {
            (Self::Lambda, [body]) => BindingExpr::Lambda(body),
            (Self::Apply, [fun, arg]) => BindingExpr::Apply(fun, arg),
            (&Self::Var(index), []) => BindingExpr::Var(index),
            (Self::Lib(ix), [bound_value, body]) => BindingExpr::Lib(*ix, bound_value, body),
            (Self::LibVar(ix), []) => BindingExpr::LibVar(*ix),
            (Self::Shift, [body]) => BindingExpr::Shift(body),
            _ => return None,
        };
        Some(binding_expr)
    }

    fn list() -> Self {
        Self::List
    }
}

// only care about s-exprs, not pretty print
impl Printable for ListOp {
    fn precedence(&self) -> Precedence {
        0
    }

    fn print_naked<W: Write>(_expr: &Expr<Self>, printer: &mut Printer<W>) -> fmt::Result {
        write!(printer.writer, "")
    }
}

fn main() {
    let input = fs::read_to_string("src_expr").expect("src_expr please");

    let prog: Vec<Expr<ListOp>> = Program::parse(&input)
        .expect("Failed to parse program")
        .0
        .into_iter()
        .map(|x| {
            x.try_into()
                .expect("Input is not a valid list of expressions")
        })
        .collect();

    let exps = Experiments::gen(
        prog,
        vec![],
        vec![],
        vec![5],
        vec![5],
        5,
        vec![false],
        vec![],
        (),
        false,
        Some(3)
    );
    exps.run("res_list.csv");
}
